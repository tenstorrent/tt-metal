# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""FF1-shaped determinism ladder: the didt FF1 matmul with one factor changed per arm.

On a box where test_ff1_matmul goes non-deterministic and test_nd_stage stays clean, the two differ in
almost everything. FF1 runs a production ttnn.matmul (2D multicast, bfp8 weights streamed from DRAM, LoFi,
16-bit DEST, packer L1 accumulation, subblock 1x8) for about a millisecond, then idles while the host syncs,
a thousand times per run, with a full readback every tenth iteration. The nd_stage probes run their own
kernels (operands in local L1, HiFi4, fp32 DEST) as one long loop per arm. Each arm here is FF1 with one of
those factors changed, so an arm that goes clean names a factor the fault needs:

  ff1        FF1 as test_ff1_matmul runs it: same seed and draw order, shapes, program and compute config
  hifi4      math fidelity HiFi4 instead of LoFi
  no_l1acc   packer L1 accumulation off: partials are spilled to L1 and unpacked back into DEST
  sub1x4     output subblock 1x4 instead of 1x8, half the DEST tiles per subblock
  bw2        in0_block_w 2 instead of 3: 99 K blocks instead of 66, more packer L1 accumulate passes
  in1_bf16   bf16 weights instead of bfp8, at in0_block_w 2 (a bf16 in1 buffer at 3 does not fit L1)
  back2back  the host syncs only before a check, so nine matmuls run back to back between checks
  idle_gap   10 ms idle after every iteration's sync, a longer idle -> load step than FF1's
  ff1_end    FF1 again, to tell a factor from drift over the run

fp32 DEST is not an arm: with packer L1 accumulation its partials buffer is Float32, 288 tiles x 4 KB =
1.18 MB per core, which does not fit next to the operands.

Each arm builds its own reference: every activation runs twice, and an element where the two runs differ
is reported as reference ND and settled by a third run. Every check compares each device's output with the
reference bit for bit; a mismatch is localized to cores (the output is block sharded: core x = column / 2304,
core y = row / 128) and to bits (flip histogram, bits set vs cleared).

Env:
  ND_LADDER_ARMS         comma list of arms (default all, in the order above)
  ND_LADDER_ITERS        iterations per arm (default 1000)
  ND_LADDER_CHECK_EVERY  iterations between checks (default 10)
  ND_LADDER_GRID         core grid (default 11x10, the grid the FF1 Galaxy runs use)
  ND_LADDER_ACTS         activations cycled between checks (default 10)
  ND_REPORT_DIR          where the JSON report goes (default generated/nd_stage)
"""

import json
import os
import socket
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from tests.didt.nd_stage.nd_stage_lib import device_identity, where

PER_CORE_M = 4  # tiles
PER_CORE_N = 72  # tiles
SHARD_W = 576  # in0 shard width in elements; K = SHARD_W * grid x
ROWS_PER_CORE = 32 * PER_CORE_M
COLS_PER_CORE = 32 * PER_CORE_N
MAX_SAMPLES = 4
IN1_DTYPES = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


@dataclass(frozen=True)
class Variant:
    fidelity: str = "LoFi"
    l1_acc: bool = True
    subblock_w: int = 8
    block_w: int = 3
    in1: str = "bfp8"
    sync: str = "each"  # each: FF1's mesh syncs after every iteration; check: only before a check
    gap_ms: float = 0.0


ARMS = {
    "ff1": Variant(),
    "hifi4": Variant(fidelity="HiFi4"),
    "no_l1acc": Variant(l1_acc=False),
    "sub1x4": Variant(subblock_w=4),
    "bw2": Variant(block_w=2),
    "in1_bf16": Variant(in1="bf16", block_w=2),
    "back2back": Variant(sync="check"),
    "idle_gap": Variant(gap_ms=10.0),
    "ff1_end": Variant(),
}


def _bf16(bits):
    return torch.tensor([bits], dtype=torch.int16).view(torch.bfloat16).item()


class Ladder:
    def __init__(self, mesh, gx, gy, n_acts, pool):
        self.mesh, self.gx, self.gy, self.pool = mesh, gx, gy, pool
        self.device_ids = list(mesh.get_device_ids())
        self.mapper = ttnn.ReplicateTensorToMesh(mesh)
        torch.manual_seed(1234)  # test_ff1_matmul's seed and draw order, so the ff1 arm runs on its data
        acts = [torch.randn([1, 1, ROWS_PER_CORE * gy, SHARD_W * gx]) for _ in range(n_acts)]
        self.w = torch.randn([1, 1, SHARD_W * gx, COLS_PER_CORE * gx])
        self.acts = [self._upload(a, ttnn.bfloat16) for a in acts]
        self.in1 = {}
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        shard = ttnn.ShardSpec(crs, [ROWS_PER_CORE, SHARD_W], ttnn.ShardOrientation.ROW_MAJOR)
        self.in0_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard)
        self.out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    def _upload(self, x, dtype):
        return ttnn.from_torch(
            x,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=self.mesh,
            mesh_mapper=self.mapper,
        )

    def weights(self, kind):
        if kind not in self.in1:
            self.in1[kind] = self._upload(self.w, IN1_DTYPES[kind])
        return self.in1[kind]

    def configs(self, v):
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(self.gx, self.gy),
            in0_block_w=v.block_w,
            out_subblock_h=1,
            out_subblock_w=v.subblock_w,
            per_core_M=PER_CORE_M,
            per_core_N=PER_CORE_N,
            transpose_mcast=False,
            fused_activation=None,
        )
        compute = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, v.fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=v.l1_acc,
        )
        return program, compute

    def matmul(self, act, w, program, compute):
        return ttnn.matmul(
            act,
            w,
            program_config=program,
            memory_config=self.out_mem,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
        )

    def to_l1(self, i):
        return ttnn.to_memory_config(self.acts[i], self.in0_mem)

    def read(self, out):
        """Each device's output as int16 bit patterns, in mesh device order."""
        return list(self.pool.map(lambda d: ttnn.to_torch(d).view(torch.int16), ttnn.get_device_tensors(out.cpu())))

    def once(self, a, w, program, compute):
        act = self.to_l1(a)
        out = self.matmul(act, w, program, compute)
        got = self.read(out)
        out.deallocate(True)
        act.deallocate(True)
        return got

    def diff(self, got, ref):
        """None when the two are bit-identical, else which cores and which bits differ."""
        if torch.equal(got, ref):
            return None
        g = got.reshape(self.gy, ROWS_PER_CORE, self.gx, COLS_PER_CORE)
        r = ref.reshape(self.gy, ROWS_PER_CORE, self.gx, COLS_PER_CORE)
        per_core = (g != r).sum(dim=(1, 3))
        cores = [[x, y, int(per_core[y, x])] for y, x in per_core.nonzero().tolist()]
        flat_g, flat_r = got.reshape(-1), ref.reshape(-1)
        bad = (flat_g != flat_r).nonzero().reshape(-1)
        gb, rb = flat_g[bad].int() & 0xFFFF, flat_r[bad].int() & 0xFFFF
        xor, set_ = gb ^ rb, gb & ~rb
        err = (flat_g[bad].view(torch.bfloat16).float() - flat_r[bad].view(torch.bfloat16).float()).abs()
        n_cols = got.shape[-1]
        samples = []
        for i in bad[:MAX_SAMPLES].tolist():
            row, col = divmod(i, n_cols)
            gv, rv = int(flat_g[i]), int(flat_r[i])
            samples.append(
                {
                    "row": row,
                    "col": col,
                    "core": [col // COLS_PER_CORE, row // ROWS_PER_CORE],
                    "got": f"{gv & 0xFFFF:#06x}",
                    "ref": f"{rv & 0xFFFF:#06x}",
                    "got_val": _bf16(gv),
                    "ref_val": _bf16(rv),
                }
            )
        return {
            "n_bad": int(len(bad)),
            "cores": cores,
            "bit_flips": [int(((xor >> b) & 1).sum()) for b in range(16)],
            "bits_set": [int(((set_ >> b) & 1).sum()) for b in range(16)],
            "max_abs_err": float(torch.nan_to_num(err, nan=float("inf")).max()),
            "samples": samples,
        }

    def references(self, w, program, compute):
        refs, ref_nd = [], []
        for a in range(len(self.acts)):
            r0 = self.once(a, w, program, compute)
            r1 = self.once(a, w, program, compute)
            infos = list(self.pool.map(self.diff, r1, r0))
            for d, info in enumerate(infos):
                if info:
                    ref_nd.append({"act": a, "device": self.device_ids[d], **info})
            if any(infos):
                r2 = self.once(a, w, program, compute)
                r0 = [torch.where(x == y, x, z) for x, y, z in zip(r0, r1, r2)]
            refs.append(r0)
        return refs, ref_nd

    def run_arm(self, v, iters, check_every):
        w = self.weights(v.in1)
        program, compute = self.configs(v)
        t0 = time.perf_counter()
        refs, ref_nd = self.references(w, program, compute)
        ref_s = time.perf_counter() - t0

        nd, walls, checks = [], [], 0
        nd_checks = Counter()
        a = 0
        act = self.to_l1(a)
        t0 = time.perf_counter()
        for i in range(iters):
            ti = time.perf_counter()
            check = i % check_every == 0
            out = self.matmul(act, w, program, compute)
            if v.sync == "each" or check or i + 1 == iters:
                for _ in self.device_ids:  # FF1 syncs the whole mesh once per device id
                    ttnn.synchronize_device(self.mesh)
            if v.gap_ms:
                time.sleep(v.gap_ms / 1000)
            if check:
                checks += 1
                for d, info in enumerate(self.pool.map(self.diff, self.read(out), refs[a])):
                    if info:
                        nd_checks[self.device_ids[d]] += 1
                        nd.append({"iter": i, "act": a, "device": self.device_ids[d], **info})
                act.deallocate(True)
                a = (a + 1) % len(self.acts)
                act = self.to_l1(a)
            out.deallocate(True)
            if not check:
                walls.append(1000 * (time.perf_counter() - ti))
        loop_s = time.perf_counter() - t0
        act.deallocate(True)

        walls.sort()
        return {
            "ref_s": round(ref_s, 1),
            "loop_s": round(loop_s, 1),
            "iters": iters,
            "checks": checks,
            "iter_ms": {q: _pct(walls, f) for q, f in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))},
            "nd_checks": dict(nd_checks),
            "ref_nd": ref_nd,
            "nd": nd,
        }


def _pct(sorted_values, q):
    if not sorted_values:
        return None
    return round(sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))], 3)


def _grid(mesh_device):
    spec = os.environ.get("ND_LADDER_GRID", "11x10")
    gx, gy = (int(v) for v in spec.lower().split("x"))
    full = mesh_device.compute_with_storage_grid_size()
    assert gx <= full.x and gy <= full.y, f"ND_LADDER_GRID {spec} exceeds the {full.x}x{full.y} grid"
    return gx, gy


def _cores_hit(records):
    """{device: {"x,y": checks that core failed in}}"""
    hit = {}
    for r in records:
        per = hit.setdefault(r["device"], Counter())
        per.update(f"{x},{y}" for x, y, _ in r["cores"])
    return {d: dict(c.most_common()) for d, c in hit.items()}


@skip_for_wormhole_b0("Targets the Blackhole FF1 non-determinism")
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="galaxy")], indirect=["mesh_device"])
def test_nd_ff1_ladder(mesh_device):
    gx, gy = _grid(mesh_device)
    iters = int(os.environ.get("ND_LADDER_ITERS", 1000))
    check_every = int(os.environ.get("ND_LADDER_CHECK_EVERY", 10))
    n_acts = int(os.environ.get("ND_LADDER_ACTS", 10))
    names = [k.strip() for k in os.environ.get("ND_LADDER_ARMS", ",".join(ARMS)).split(",") if k.strip()]
    unknown = [n for n in names if n not in ARMS]
    assert names and not unknown, f"ND_LADDER_ARMS: unknown {unknown}, choose from {list(ARMS)}"
    device_ids = list(mesh_device.get_device_ids())
    identity = device_identity(device_ids)
    logger.info(
        f"ff1 ladder: {len(device_ids)} devices, grid {gx}x{gy}, arms {names}, {iters} iters, check every {check_every}"
    )

    arms = []
    with ThreadPoolExecutor(len(device_ids)) as pool:
        ladder = Ladder(mesh_device, gx, gy, n_acts, pool)
        for name in names:
            v = ARMS[name]
            rec = {"arm": name, "variant": asdict(v)}
            try:
                rec |= ladder.run_arm(v, iters, check_every)
            except Exception as e:  # an arm that cannot run (e.g. L1) must not stop the ladder
                logger.exception(f"{name}: failed to run")
                rec["error"] = f"{type(e).__name__}: {e}"
            arms.append(rec)
            if "error" in rec:
                continue
            rec["cores_hit"] = _cores_hit(rec["nd"])
            rec["ref_cores_hit"] = _cores_hit(rec["ref_nd"])
            logger.info(
                f"{name:10s} ref {rec['ref_s']:6.1f} s  loop {rec['loop_s']:6.1f} s  iter median "
                f"{rec['iter_ms']['median']} ms  ND checks {rec['nd_checks'] or 0}/{rec['checks']}  "
                f"ref ND {len(rec['ref_nd'])}  cores {rec['cores_hit'] or '-'}"
            )

    logger.info("ladder summary (ND checks per device / checks, reference ND records):")
    for rec in arms:
        if "error" in rec:
            logger.error(f"  {rec['arm']:10s} ERROR {rec['error']}")
        else:
            logger.info(
                f"  {rec['arm']:10s} {rec['nd_checks'] or 0}/{rec['checks']}  ref ND {len(rec['ref_nd'])}  "
                f"cores {rec['cores_hit'] or '-'}"
            )

    hit = sorted({d for r in arms for key in ("cores_hit", "ref_cores_hit") for d in r.get(key) or {}})
    for d in hit:
        logger.error(f"  ND on dev {d:2d} | {where(identity, d)}")

    report = {
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "devices": device_ids,
        "identity": identity,
        "grid": [gx, gy],
        "iters": iters,
        "check_every": check_every,
        "activations": n_acts,
        "arms": arms,
    }
    out_dir = Path(os.environ.get("ND_REPORT_DIR", "generated/nd_stage"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"nd_ff1_ladder_{socket.gethostname()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1))
    logger.info(f"report: {path}")

    errors = [r["arm"] for r in arms if "error" in r]
    failing = [r["arm"] for r in arms if r.get("nd") or r.get("ref_nd")]
    assert not errors and not failing, f"ND in arms {failing}, arms that did not run {errors}; see {path}"
