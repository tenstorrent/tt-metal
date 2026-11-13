# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Host side of the ND stage probes: probe table, inputs and goldens, program building, result analysis.

Every probe is a small kernel set that loads one part of the Tensix pipeline (or L1, or the NoC) at high
duty and checks its own output run-to-run on the core. A core that fails some probes and passes others
narrows the fault to the stages the failing probes share and the passing ones do not touch (see
STAGES and localize()).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml
from loguru import logger

import ttnn

KERNEL_DIR = "tests/didt/nd_stage/kernels"
COMPUTE_KERNEL = f"{KERNEL_DIR}/nd_stage_compute.cpp"
CHECKER_KERNEL = f"{KERNEL_DIR}/nd_stage_checker.cpp"
L1_KERNEL = f"{KERNEL_DIR}/nd_stage_l1_march.cpp"
NOC_KERNEL = f"{KERNEL_DIR}/nd_stage_noc_mcast.cpp"

CB_IN0, CB_IN1, CB_OUT, CB_LOOP = 0, 1, 16, 24
NUM_CBS = 64  # host descriptors carry the maximum CB count (Blackhole)
RT, CT, KT = 2, 2, 8  # matmul block: 2x2 output tiles fill half-sync fp32 DEST, 8 k-tiles per sweep
DST_TILES = 4

# Result words; mirrors kernels/nd_stage_result.h.
MAGIC_CHECKER, MAGIC_L1, MAGIC_NOC, MAGIC_DEST = 0x4E445354, 0x4E444C31, 0x4E444E43, 0x4E444453
NO_ITER = 0xFFFFFFFF
INJECT_XOR = 1 << 20  # fp32 mantissa bit 20; bit 4 of the upper element of a bf16 pair
(
    W_MAGIC,
    W_ITERS_DONE,
    W_ITERS_CHECKED,
    W_BAD_ITERS,
    W_BAD_WORDS,
    W_FIRST_BAD_ITER,
    W_FIRST_BAD_WORD,
    W_FIRST_GOT,
    W_FIRST_REF,
    W_LAST_BAD_ITER,
    W_XOR_UNION,
    W_ROW_MASK,
    W_COL_MASK,
    W_TILE_MASK,
    W_ARC_START,
    W_ARC_END,
    W_CYCLES_LO,
    W_CYCLES_HI,
    W_N_SAMPLES,
    W_CORE_ID,
    W_BIT_HIST,  # 8 words: a saturating u8 count per element bit position
) = range(21)
W_SAMPLES = W_BIT_HIST + 8
MAX_SAMPLES = 4
W_COUNT = W_SAMPLES + 4 * MAX_SAMPLES
# Slot 0: the probe's checker, or the DM probe; slot 1: the compute checkers in L1/NoC arms; slot 2: the
# DEST readback of MM_DESTRD.
RESULT_SLOT_BYTES = 256
RESULT_BYTES = 3 * RESULT_SLOT_BYTES
DEST_SLOT = 2

L1_MARCH_BYTES = 512 * 1024
NOC_BUF_BYTES = 64 * 1024

# Stages a probe exercises. A probe that passes clears only the stages in its `clears` set: small L1
# regions (CBs, NoC buffers) do not clear L1 as a whole, only the march does. Names follow the Blackhole
# Tensix coprocessor docs (tt-isa-documentation, BlackholeA0/TensixTile/TensixCoprocessor).
UNPACK0_SRCA = "unpacker 0 -> SrcA"
UNPACK1_SRCB = "unpacker 1 -> SrcB"
UNPACK_DEST = "unpacker 0 -> DEST (32-bit)"
FPU_MVMUL = "FPU MVMUL (multiplier array + adder tree)"
FPU_DATACOPY = "FPU datacopy (SrcA -> DEST, MOVA2D)"
FPU_ELWADD = "FPU ELWADD (adder + DEST accumulate)"
MOVD2A = "DEST -> SrcA move (MOVD2A)"
MOVD2B = "DEST -> SrcB move (MOVD2B)"
SFPU_LDST = "SFPU load/store + LREGs"
SFPU_MAD = "SFPU fp32 MAD (SFPMAD)"
SFPU_INT = "SFPU int path (SFPMUL24, IADD, SHFT, LOADMACRO)"
DEST = "DEST register file"
DEST_RD = "DEST RISC read port"
PACK = "packer (DEST -> L1)"
PACK_L1ACC = "packer L1 accumulate (L1-port adder)"
L1 = "L1 SRAM"
NOC0 = "NoC0 multicast"
NOC1 = "NoC1 multicast"


@dataclass(frozen=True)
class Fmt:
    ttnn_dtype: object
    torch_dtype: torch.dtype
    tile_bytes: int
    elem_shift: int  # 1: two 16-bit elements per 32-bit word


FMTS = {
    "bf16": Fmt(ttnn.bfloat16, torch.bfloat16, 2048, 1),
    "f32": Fmt(ttnn.float32, torch.float32, 4096, 0),
    "int32": Fmt(ttnn.int32, torch.int32, 4096, 0),
}


@dataclass(frozen=True)
class Probe:
    name: str
    probe_id: int  # ND_PROBE in nd_stage_compute.cpp
    in0: str
    in1: str | None
    out: str
    in0_tiles: int
    in1_tiles: int
    reps: int  # inner passes per iteration (runtime arg)
    pushes_per_iter: int
    check_last: int  # pushes per iteration the checker compares
    check_every: int  # checked iterations; the rest are popped unread so the checker keeps up
    unpack_to_dest: bool  # in0 (and the loop CB) unpack straight to DEST
    loop_cb: bool
    stages: frozenset
    what: str
    full_sync: bool = False  # DEST in full-sync mode, so the four tiles sit at offset 0 every iteration
    dest_check: bool = False  # the math RISC also checks DEST before the pack (ND_PROBE MM_DESTRD)

    @property
    def push_bytes(self):
        return DST_TILES * FMTS[self.out].tile_bytes

    @property
    def ref_bytes(self):
        return self.check_last * self.push_bytes

    @property
    def dest_ref_bytes(self):
        return DST_TILES * 4096 if self.dest_check else 0


_COMPUTE = frozenset({DEST, PACK, L1})
_MM = _COMPUTE | {UNPACK0_SRCA, UNPACK1_SRCB, FPU_MVMUL}
_ELT = _COMPUTE | {UNPACK0_SRCA, FPU_DATACOPY, FPU_ELWADD}
PROBES = {
    p.name: p
    for p in [
        Probe(
            "MM_DEST", 0, "bf16", "bf16", "f32", RT * KT, KT * CT, 28, 1, 1, 1, False, False,
            _MM,
            "matmul, K=7168 accumulated in DEST, one pack",
        ),
        Probe(
            "MM_L1ACC", 1, "bf16", "bf16", "f32", RT * KT, KT * CT, 28, 28, 2, 1, False, False,
            _MM | {PACK_L1ACC},
            "matmul, K=256 per block, partials summed by packer L1 accumulation",
        ),
        Probe(
            "MM_DESTRD", 6, "bf16", "bf16", "f32", RT * KT, KT * CT, 28, 1, 1, 4, False, False,
            _MM,
            "MM_DEST in full-sync DEST, DEST read back by the math RISC before the pack",
            full_sync=True, dest_check=True,
        ),
        Probe(
            "FPU_ELTWISE", 2, "bf16", "bf16", "f32", DST_TILES, 2, 64, 1, 1, 2, False, False,
            _ELT | {UNPACK1_SRCB, MOVD2A},
            "DEST += v, DEST -= v alternating, v via unpacker 1 into SrcB, DEST -> SrcA move",
        ),
        Probe(
            "ELTWISE_SRCB", 7, "bf16", "bf16", "f32", DST_TILES, 2, 64, 1, 1, 2, False, False,
            _ELT | {MOVD2B},
            "DEST += v, DEST -= v alternating, v via unpacker 0 into SrcA, DEST -> SrcB move",
        ),
        Probe(
            "SFPU_INT", 3, "int32", None, "int32", DST_TILES, 0, 32, 1, 1, 2, True, False,
            _COMPUTE | {UNPACK_DEST, SFPU_LDST, SFPU_INT},
            "int32 x *= c (odd) in DEST on the SFPU",
        ),
        Probe(
            "SFPU_FP", 9, "f32", None, "f32", DST_TILES, 0, 32, 1, 1, 2, True, False,
            _COMPUTE | {UNPACK_DEST, SFPU_LDST, SFPU_MAD},
            "fp32 x *= 2, x *= 0.5 in DEST on the SFPU",
        ),
        Probe(
            "SFPU_LDST", 8, "int32", None, "int32", DST_TILES, 0, 32, 1, 1, 2, True, False,
            _COMPUTE | {UNPACK_DEST, SFPU_LDST},
            "int32 tiles copied DEST -> LREG -> DEST on the SFPU, no arithmetic",
        ),
        Probe(
            "UNPACK_PACK", 4, "int32", None, "int32", 2 * DST_TILES, 0, 16, 2, 2, 4, True, True,
            _COMPUTE | {UNPACK_DEST},
            "int32 tiles cycled L1 -> DEST -> L1, no arithmetic",
        ),
        Probe(
            "COPY_SRCA", 4, "bf16", None, "bf16", 2 * DST_TILES, 0, 16, 2, 2, 4, False, True,
            _COMPUTE | {UNPACK0_SRCA, FPU_DATACOPY},
            "bf16 tiles cycled L1 -> SrcA -> DEST -> L1 (FPU datacopy)",
        ),
        Probe(
            "PACK_L1ACC", 5, "f32", None, "f32", DST_TILES, 0, 32, 32, 2, 4, True, False,
            _COMPUTE | {UNPACK_DEST, PACK_L1ACC},
            "one DEST packed 32x with L1 accumulation, FPU idle",
        ),
    ]
}  # fmt: skip
NOC_EXTRAS = {"NOC_MCAST": ttnn.NOC.NOC_0, "NOC1_MCAST": ttnn.NOC.NOC_1}
DESTRD_DEST = "MM_DESTRD:dest"  # the DEST readback record of MM_DESTRD, analyzed as its own probe
DM_PROBES = {"L1_MARCH": frozenset({L1}), "NOC_MCAST": frozenset({NOC0, L1}), "NOC1_MCAST": frozenset({NOC1, L1})}
VIRTUAL_PROBES = {DESTRD_DEST: (PROBES["MM_DESTRD"].stages - {PACK, L1}) | {DEST_RD}}


def stages_of(probe_name):
    if probe_name in DM_PROBES:
        return DM_PROBES[probe_name]
    if probe_name in VIRTUAL_PROBES:
        return VIRTUAL_PROBES[probe_name]
    return PROBES[probe_name].stages


def clears(probe_name, failed=()):
    """Stages a pass of this probe rules out. MM_DESTRD checks a quarter as often as MM_DEST and stalls the
    FPU for each DEST readback, so its passes clear nothing alone; its DEST record clears the stages before
    the pack only when its packed record failed on the same checked iterations."""
    if probe_name == "MM_DESTRD" or (probe_name == DESTRD_DEST and "MM_DESTRD" not in failed):
        return frozenset()
    stages = stages_of(probe_name)
    return stages if probe_name == "L1_MARCH" else stages - {L1}


# ---------------------------------------------------------------------------------------------- data


def tilize(x):
    """[R*32, C*32] -> flat elements in tile order (row-major tiles, four 16x16 faces each row-major)."""
    r, c = x.shape[0] // 32, x.shape[1] // 32
    return x.reshape(r, 32, c, 32).permute(0, 2, 1, 3).reshape(r, c, 2, 16, 2, 16).permute(0, 1, 2, 4, 3, 5).reshape(-1)


def _bytes(t):
    return t.contiguous().view(torch.uint8).numpy().copy()


def _gen(shape, dtype, mode, gen):
    if mode == "int":
        return torch.randint(-3, 4, shape, generator=gen).to(dtype)
    return torch.randn(shape, generator=gen).to(dtype)


def make_inputs(probe, mode, seed=0):
    """(in0 bytes, in1 bytes or None, golden bytes of the checked region or None).

    The golden is exact for the int32 and copy probes always, and for the float probes in int mode
    (small integers keep every partial sum exact); otherwise the cross-core consensus stands in.
    """
    gen = torch.Generator().manual_seed(seed)
    n = probe.name
    golden = None
    if n in ("MM_DEST", "MM_L1ACC", "MM_DESTRD"):
        a = _gen((RT * 32, KT * 32), torch.bfloat16, mode, gen)
        b = _gen((KT * 32, CT * 32), torch.bfloat16, mode, gen)
        in0, in1 = _bytes(tilize(a)), _bytes(tilize(b))
        if mode == "int":
            scale = probe.reps // 2 if n == "MM_L1ACC" else probe.reps
            c = tilize((a.double() @ b.double()) * scale).to(torch.float32)
            golden = _bytes(c.repeat(probe.check_last))
    elif n in ("FPU_ELTWISE", "ELTWISE_SRCB"):
        x = _gen((DST_TILES * 1024,), torch.bfloat16, mode, gen)
        v = _gen((1024,), torch.bfloat16, mode, gen)
        in0, in1 = _bytes(x), _bytes(torch.cat([v, -v]))
        if mode == "int":
            golden = _bytes(x.to(torch.float32))
    elif n == "SFPU_INT":
        x = torch.randint(-(2**31), 2**31, (DST_TILES * 1024,), generator=gen, dtype=torch.int64)
        x[3 * 1024 :] |= 1  # tile 3 is the multiplier: odd, so the product map is a bijection
        in0 = _bytes(x.to(torch.int32))
        u = x.numpy().astype(np.uint64) & 0xFFFFFFFF
        out, c = u.copy(), u[3 * 1024 :]
        for _ in range(probe.reps):
            out[: 3 * 1024] = (out[: 3 * 1024] * np.tile(c, 3)) & 0xFFFFFFFF
        golden = out.astype(np.uint32).view(np.uint8).copy()
        in1 = None
    elif n == "SFPU_LDST":
        # Tiles 2, 3 start as copies of 0, 1, so every round of copies leaves all four unchanged.
        x = torch.randint(-(2**31), 2**31, (2 * 1024,), generator=gen, dtype=torch.int64).to(torch.int32)
        in0 = golden = _bytes(torch.cat([x, x]))
        in1 = None
    elif n == "SFPU_FP":
        # x * 2 * 0.5 is exact for every normal fp32, so the input is the golden in both data modes.
        in0 = golden = _bytes(_gen((DST_TILES * 1024,), torch.float32, mode, gen))
        in1 = None
    elif n == "UNPACK_PACK":
        x = torch.randint(-(2**31), 2**31, (2 * DST_TILES * 1024,), generator=gen, dtype=torch.int64).to(
            torch.int32
        )
        in0 = golden = _bytes(x)
        in1 = None
    elif n == "COPY_SRCA":
        in0 = golden = _bytes(torch.randn(2 * DST_TILES * 1024, generator=gen).to(torch.bfloat16))
        in1 = None
    elif n == "PACK_L1ACC":
        v = _gen((DST_TILES * 1024,), torch.float32, mode, gen)
        in0, in1 = _bytes(v), None
        if mode == "int":
            golden = _bytes((v * (probe.reps // 2)).repeat(probe.check_last))
    else:
        raise ValueError(n)
    assert len(in0) == probe.in0_tiles * FMTS[probe.in0].tile_bytes, n
    assert in1 is None or len(in1) == probe.in1_tiles * FMTS[probe.in1].tile_bytes, n
    assert golden is None or len(golden) == probe.ref_bytes, n
    return in0, in1, golden


# ------------------------------------------------------------------------------------------- arms


@dataclass
class Arm:
    """One program run. `groups` maps each compute probe to its cores; `extra` names a DM probe (L1_MARCH,
    NOC_MCAST, NOC1_MCAST: BRISC, with or without MM aggressors on the same cores)."""

    name: str
    groups: dict = field(default_factory=dict)  # probe name -> [(x, y)]
    extra: str | None = None
    loaded: bool = False


def plan_arms(cores, gx, only=None):
    quiet = [Arm(f"{p}:quiet", {p: cores}) for p in PROBES]
    loaded = []
    for p in PROBES:
        if p.startswith("MM_"):  # the MM probes are the aggressors
            continue
        for parity in (0, 1):
            victims = [c for c in cores if (c[0] + c[1]) % 2 == parity]
            aggressors = [c for c in cores if (c[0] + c[1]) % 2 != parity]
            loaded.append(Arm(f"{p}:loaded{parity}", {p: victims, "MM_DEST": aggressors}, loaded=True))
    dm = [
        Arm("L1_MARCH:quiet", extra="L1_MARCH"),
        Arm("L1_MARCH:loaded", {"MM_DEST": cores}, extra="L1_MARCH", loaded=True),
    ]
    if gx > 1:
        for name in NOC_EXTRAS:
            dm += [
                Arm(f"{name}:quiet", extra=name),
                Arm(f"{name}:loaded", {"MM_DEST": cores}, extra=name, loaded=True),
            ]
    arms = quiet + loaded + dm
    if only:
        keys = [k.strip() for k in only.split(",") if k.strip()]
        arms = [a for a in arms if any(a.name == k or a.name.startswith(k + ":") for k in keys)]
    return arms


@dataclass(frozen=True)
class Inject:
    """A flipped output bit on one core, in every arm where `probe` runs there, on every device."""

    probe: str
    core: tuple
    iteration: int
    pack_only: bool = False  # MM_DESTRD:pack, the flip lands after DEST so only the packed record sees it

    def expected(self):
        """(probe record, core) pairs the injection must flag."""
        if PROBES[self.probe].dest_check and not self.pack_only:
            return {(self.probe, self.core), (DESTRD_DEST, self.core)}
        return {(self.probe, self.core)}


def parse_inject(spec):
    """ "PROBE@x,y" or "PROBE@x,y@iter". The default iteration is the first checked one after 0; 0 corrupts
    the reference copy, which the host catches by consensus (and golden) instead of run-to-run.
    MM_DESTRD flips a DEST bit before the pack; MM_DESTRD:pack flips the packed copy only."""
    if not spec:
        return None
    parts = spec.split("@")
    name, _, variant = parts[0].partition(":")
    assert (
        len(parts) in (2, 3) and name in PROBES and variant in ("", "pack")
    ), f"ND_INJECT={spec!r}: want COMPUTE_PROBE[:pack]@x,y[@iter]"
    assert variant == "" or PROBES[name].dest_check, f"ND_INJECT={spec!r}: ':pack' only applies to MM_DESTRD"
    x, y = (int(v) for v in parts[1].split(","))
    it = int(parts[2]) if len(parts) == 3 else PROBES[name].check_every
    return Inject(name, (x, y), it, variant == "pack")


# --------------------------------------------------------------------------------------- program


def _crs(cores):
    """CoreRangeSet of the cores, merging runs along x within a row."""
    ranges, cs = [], sorted(cores, key=lambda c: (c[1], c[0]))
    i = 0
    while i < len(cs):
        x0, y = cs[i]
        j = i
        while j + 1 < len(cs) and cs[j + 1] == (cs[j][0] + 1, y):
            j += 1
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y), ttnn.CoreCoord(cs[j][0], y)))
        i = j + 1
    return ttnn.CoreRangeSet(ranges)


def _sharded(mesh, grid_crs, rows):
    """HEIGHT_SHARDED uint8 L1 tensor, one row per core of grid_crs in row-major core order."""
    shard_bytes = rows.shape[1]
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid_crs, [1, shard_bytes], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch.from_numpy(rows),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _pad64(n):
    return max(64, (n + 63) // 64 * 64)


class Runner:
    """Holds the grid and the per-probe settings; builds and runs arms."""

    def __init__(self, mesh, gx, gy, mode="random", fidelity="HiFi4", inject=None):
        self.mesh = mesh
        self.inject = inject
        self.gx, self.gy = gx, gy
        self.cores = [(x, y) for y in range(gy) for x in range(gx)]  # row-major = shard order
        self.index = {c: i for i, c in enumerate(self.cores)}
        self.grid_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        self.mode = mode
        self.fidelity = getattr(ttnn.MathFidelity, fidelity)
        self.inputs = {p: make_inputs(PROBES[p], mode) for p in PROBES}
        self.virt = {c: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c)) for c in self.cores}

    # -- compute groups

    def _compute_config(self, probe):
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=self.fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=probe.full_sync,
        )
        if probe.unpack_to_dest:
            modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
            modes[CB_IN0] = ttnn.UnpackToDestMode.UnpackToDestFp32
            if probe.loop_cb:
                modes[CB_LOOP] = ttnn.UnpackToDestMode.UnpackToDestFp32
            cfg.unpack_to_dest_mode = modes
        return cfg

    def _group(self, probe, cores, iters, t, result_off):
        crs = _crs(cores)
        f0, fo = FMTS[probe.in0], FMTS[probe.out]
        cbs = []
        in0 = ttnn.cb_descriptor_from_sharded_tensor(
            CB_IN0, t["in0"], address_offset=0, total_size=probe.in0_tiles * f0.tile_bytes, core_ranges=crs
        )
        in0.format_descriptors = [ttnn.CBFormatDescriptor(CB_IN0, f0.ttnn_dtype, f0.tile_bytes)]
        cbs.append(in0)
        if probe.in1:
            f1 = FMTS[probe.in1]
            in1 = ttnn.cb_descriptor_from_sharded_tensor(
                CB_IN1, t["in1"], address_offset=0, total_size=probe.in1_tiles * f1.tile_bytes, core_ranges=crs
            )
            in1.format_descriptors = [ttnn.CBFormatDescriptor(CB_IN1, f1.ttnn_dtype, f1.tile_bytes)]
            cbs.append(in1)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * DST_TILES * fo.tile_bytes,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_OUT, fo.ttnn_dtype, fo.tile_bytes)],
            )
        )
        if probe.loop_cb:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=2 * DST_TILES * f0.tile_bytes,
                    core_ranges=crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_LOOP, f0.ttnn_dtype, f0.tile_bytes)],
                )
            )
        compute_rt, checker_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        ref_addr, res_addr = t["ref"].buffer_address(), t["result"].buffer_address() + result_off
        dest_res_addr = t["result"].buffer_address() + DEST_SLOT * RESULT_SLOT_BYTES
        for x, y in cores:
            inj = self.inject
            hit = inj is not None and inj.probe == probe.name and inj.core == (x, y)
            # MM_DESTRD injects into DEST (both records see it) unless the target is ":pack" (checker only).
            hit_dest = hit and probe.dest_check and not inj.pack_only
            hit_pack = hit and not hit_dest
            compute_rt[x][y] = [
                iters,
                probe.reps,
                probe.check_every,
                ref_addr + probe.ref_bytes,
                dest_res_addr,
                self.index[(x, y)],
                *([inj.iteration, 0, INJECT_XOR] if hit_dest else [NO_ITER, 0, 0]),
            ]
            checker_rt[x][y] = [
                iters,
                probe.pushes_per_iter,
                DST_TILES,
                probe.check_last,
                fo.tile_bytes,
                probe.check_every,
                probe.in0_tiles,
                probe.in1_tiles,
                ref_addr,
                res_addr,
                self.index[(x, y)],
                *([inj.iteration, 0, INJECT_XOR] if hit_pack else [NO_ITER, 0, 0]),
            ]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE_KERNEL,
                core_ranges=crs,
                compile_time_args=[CB_IN0, CB_IN1, CB_OUT, CB_LOOP, RT, CT, KT, probe.in0_tiles, probe.in1_tiles],
                defines=[("ND_PROBE", str(probe.probe_id))],
                runtime_args=compute_rt,
                config=self._compute_config(probe),
            ),
            ttnn.KernelDescriptor(
                kernel_source=CHECKER_KERNEL,
                core_ranges=crs,
                compile_time_args=[CB_IN0, CB_IN1, CB_OUT, fo.elem_shift],
                runtime_args=checker_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
        ]
        return kernels, cbs

    # -- DM probes (BRISC, NOC_0)

    def _l1_kernel(self, passes, t):
        rt = ttnn.RuntimeArgs()
        base, res = t["march"].buffer_address(), t["result"].buffer_address()
        for x, y in self.cores:
            rt[x][y] = [base, L1_MARCH_BYTES, passes, res, self.index[(x, y)]]
        return ttnn.KernelDescriptor(
            kernel_source=L1_KERNEL,
            core_ranges=self.grid_crs,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        )

    def _noc_kernel(self, rounds, t, noc, check_every=8):
        rt = ttnn.RuntimeArgs()
        base, res = t["noc"].buffer_address(), t["result"].buffer_address()
        pattern, rx = base, base + 2 * NOC_BUF_BYTES
        assert self.gx > 1, "the row multicast needs at least one receiver per row"
        for y in range(self.gy):
            recv = [self.virt[(x, y)] for x in range(1, self.gx)]
            x0, x1 = min(v.x for v in recv), max(v.x for v in recv)
            y0, y1 = min(v.y for v in recv), max(v.y for v in recv)
            assert y0 == y1, f"logical row {y} spans virtual rows {y0}..{y1}; the row multicast assumes one"
            s = self.virt[(0, y)]
            # The sender must sit outside the rectangle: it waits for exactly gx-1 acks and keeps its own copy.
            assert not x0 <= s.x <= x1, f"row {y}: sender x={s.x} inside the multicast rectangle {x0}..{x1}"
            # NoC1 runs in the opposite direction to NoC0, so its multicast rectangle is given end-first.
            r0, r1 = ((x0, y0), (x1, y1)) if noc == ttnn.NOC.NOC_0 else ((x1, y1), (x0, y0))
            for x in range(self.gx):
                rt[x][y] = [
                    int(x == 0), rounds, NOC_BUF_BYTES, pattern, rx, self.gx - 1,
                    *r0, *r1, s.x, s.y, check_every, res, self.index[(x, y)],
                ]  # fmt: skip
        # BRISC; the checker of a loaded arm reads on NCRISC (ReaderConfigDescriptor), so either NoC is free.
        return ttnn.KernelDescriptor(
            kernel_source=NOC_KERNEL,
            core_ranges=self.grid_crs,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=noc),
        )

    # -- tensors

    def _tensors(self, arm):
        n = len(self.cores)
        t = {}
        group_of = {c: p for p, cs in arm.groups.items() for c in cs}
        if arm.groups:
            for key, idx in (("in0", 0), ("in1", 1)):
                width = max(
                    (len(self.inputs[p][idx]) for p in arm.groups if self.inputs[p][idx] is not None), default=0
                )
                if width == 0:
                    continue
                rows = np.zeros((n, _pad64(width)), np.uint8)
                for c, p in group_of.items():
                    data = self.inputs[p][idx]
                    if data is not None:
                        rows[self.index[c], : len(data)] = data
                t[key] = _sharded(self.mesh, self.grid_crs, rows)
            ref_w = max(PROBES[p].ref_bytes + PROBES[p].dest_ref_bytes for p in arm.groups)
            t["ref"] = _sharded(self.mesh, self.grid_crs, np.zeros((n, _pad64(ref_w)), np.uint8))
        t["result"] = _sharded(self.mesh, self.grid_crs, np.zeros((n, RESULT_BYTES), np.uint8))
        if arm.extra == "L1_MARCH":
            t["march"] = _sharded(self.mesh, self.grid_crs, np.zeros((n, L1_MARCH_BYTES), np.uint8))
        if arm.extra in NOC_EXTRAS:
            gen = np.random.default_rng(1)
            p0 = gen.integers(0, 2**32, NOC_BUF_BYTES // 4, dtype=np.uint64).astype(np.uint32)
            row = np.concatenate([p0, ~p0, np.zeros_like(p0)]).view(np.uint8)
            t["noc"] = _sharded(self.mesh, self.grid_crs, np.tile(row, (n, 1)))
        return t

    # -- run

    def run(self, arm, iters, t=None):
        """iters: {probe name or extra: count}. Returns (tensors, wall seconds)."""
        t = t or self._tensors(arm)
        kernels, cbs, semaphores = [], [], []
        checker_slot = RESULT_SLOT_BYTES if arm.extra else 0  # the DM probe owns slot 0 when present
        for p, cores in arm.groups.items():
            k, c = self._group(PROBES[p], cores, iters[p], t, checker_slot)
            kernels += k
            cbs += c
        if arm.extra == "L1_MARCH":
            kernels.append(self._l1_kernel(iters["L1_MARCH"], t))
        elif arm.extra in NOC_EXTRAS:
            kernels.append(self._noc_kernel(iters[arm.extra], t, NOC_EXTRAS[arm.extra]))
            semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.grid_crs, initial_value=0) for i in (0, 1)]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
        io = list(t.values())
        start = time.perf_counter()
        ttnn.generic_op(io, program)
        ttnn.synchronize_device(self.mesh)
        wall = time.perf_counter() - start
        return t, wall

    def calibrate(self, arm, key, seconds, min_wall=0.3):
        """Iterations of `key` that fill `seconds` when run alone (quiet arm)."""
        t = self._tensors(arm)
        keys = list(arm.groups) + ([arm.extra] if arm.extra else [])
        self.run(arm, dict.fromkeys(keys, 2), t)  # compile + warm-up
        n = 8
        while True:
            _, wall = self.run(arm, dict.fromkeys(keys, n), t)
            if wall >= min_wall or n >= 1 << 26:
                break
            n *= 4
        free(t)
        rate = n / wall
        return max(4, min(int(rate * seconds), 0xFFFFFFF0)), rate


def free(t):
    for v in t.values():
        ttnn.deallocate(v)


# -------------------------------------------------------------------------------------- readback


def device_shards(tensor, nbytes):
    """[(shard index, np.uint8[ncores, nbytes])] for every device of the mesh."""
    out = []
    for i, d in enumerate(ttnn.get_device_tensors(tensor)):
        a = ttnn.to_torch(d).numpy().astype(np.uint8)
        out.append((i, a.reshape(a.shape[0], -1)[:, :nbytes]))
    return out


@dataclass
class CoreResult:
    words: np.ndarray  # uint32[W_COUNT]

    @property
    def magic(self):
        return int(self.words[W_MAGIC])

    def __getitem__(self, w):
        return int(self.words[w])

    @property
    def nd(self):
        return self[W_BAD_ITERS] > 0

    @property
    def cycles(self):
        return self[W_CYCLES_LO] | (self[W_CYCLES_HI] << 32)

    def bit_hist(self):
        """{element bit: times it flipped}, each count saturating at 255."""
        words = self.words[W_BIT_HIST : W_BIT_HIST + 8]
        counts = {b: int(words[b // 4]) >> (8 * (b % 4)) & 0xFF for b in range(32)}
        return {b: n for b, n in counts.items() if n}

    def samples(self):
        return [
            tuple(int(v) for v in self.words[W_SAMPLES + 4 * i : W_SAMPLES + 4 * i + 4])
            for i in range(min(self[W_N_SAMPLES], MAX_SAMPLES))
        ]


def parse_results(rows, slot):
    words = rows[:, slot * RESULT_SLOT_BYTES : slot * RESULT_SLOT_BYTES + 4 * W_COUNT].copy().view(np.uint32)
    return [CoreResult(words[i]) for i in range(words.shape[0])]


# --------------------------------------------------------------------------------------- signature


def bit_fields(xor_union, fmt):
    """Which fields of the element format the flipped bits fall in."""
    if xor_union == 0:
        return "-"
    bits = [b for b in range(32) if xor_union >> b & 1]
    if fmt == "f32":
        layout = [("sign", 31, 31), ("exp", 23, 30), ("mant", 0, 22)]
    elif fmt == "bf16":
        layout = [("sign", 15, 15), ("exp", 7, 14), ("mant", 0, 6)]
    else:
        return f"bits {_ranges(bits)}"
    parts = []
    for name, lo, hi in layout:
        hit = [b - lo for b in bits if lo <= b <= hi]
        if hit:
            parts.append(name if name == "sign" else f"{name}[{_ranges(hit)}]")
    return " ".join(parts)


def _ranges(v):
    v = sorted(v)
    out, start = [], None
    for i, b in enumerate(v):
        if start is None:
            start = b
        if i + 1 == len(v) or v[i + 1] != b + 1:
            out.append(str(start) if start == b else f"{start}-{b}")
            start = None
    return ",".join(out)


def mask_list(m):
    return _ranges([b for b in range(32) if m >> b & 1]) or "-"


# --------------------------------------------------------------------------------------- analysis


@dataclass
class Finding:
    """One failing (device, core) in one arm."""

    arm: str
    probe: str
    loaded: bool
    device: int  # physical device id
    shard: int
    core: tuple
    kind: str  # "ND" (run-to-run), "golden" or "consensus" (stable but wrong)
    detail: dict


def analyze_arm(runner, arm, t, device_ids):
    """Findings for one finished arm, plus per-core health records for the summary."""
    findings, notes = [], []
    res_rows = dict(device_shards(t["result"], RESULT_BYTES))
    ref_rows = dict(device_shards(t["ref"], t["ref"].shape[-1])) if "ref" in t else {}

    def check_magic(shard, i, r, magic, probe):
        if r.magic != magic:
            notes.append(
                f"{arm.name}: dev {device_ids[shard]} core {runner.cores[i]} {probe}: no result (magic {r.magic:#x})"
            )
            return False
        if r[W_CORE_ID] != i:
            notes.append(f"{arm.name}: dev {device_ids[shard]} shard row {i} reports core id {r[W_CORE_ID]}")
        if (r[W_ARC_START] | r[W_ARC_END]) & 1:
            notes.append(f"{arm.name}: dev {device_ids[shard]} core {runner.cores[i]} {probe}: ARC throttle flag set")
        return True

    def nd_finding(shard, i, r, probe, fmt):
        return Finding(
            arm.name, probe, arm.loaded, device_ids[shard], shard, runner.cores[i], "ND",
            {
                "iters_checked": r[W_ITERS_CHECKED], "bad_iters": r[W_BAD_ITERS], "bad_words": r[W_BAD_WORDS],
                "first_bad_iter": r[W_FIRST_BAD_ITER], "first_word": r[W_FIRST_BAD_WORD],
                "first_got": f"{r[W_FIRST_GOT]:#010x}", "first_ref": f"{r[W_FIRST_REF]:#010x}",
                "bits": bit_fields(r[W_XOR_UNION], fmt), "xor_union": f"{r[W_XOR_UNION]:#010x}",
                "tiles": mask_list(r[W_TILE_MASK]), "rows": mask_list(r[W_ROW_MASK]), "cols": mask_list(r[W_COL_MASK]),
                "face_cols": _ranges({c % 16 for c in range(32) if r[W_COL_MASK] >> c & 1}) or "-",
                "bit_hist": r.bit_hist(),
                "samples": [[it, w, f"{g:#010x}", f"{e:#010x}"] for it, w, g, e in r.samples()],
            },
        )  # fmt: skip

    # Compute probes: run-to-run from the on-core checker, golden and consensus from the iteration-0 copy.
    slot = 1 if arm.extra else 0
    for p, cores in arm.groups.items():
        probe = PROBES[p]
        golden = runner.inputs[p][2]
        idx = [runner.index[c] for c in cores]
        refs = {}
        for shard, rows in res_rows.items():
            results = parse_results(rows, slot)
            for i in idx:
                r = results[i]
                if not check_magic(shard, i, r, MAGIC_CHECKER, p):
                    continue
                if r.nd:
                    findings.append(nd_finding(shard, i, r, p, probe.out))
            ref = ref_rows[shard]
            for i in idx:
                refs[(shard, i)] = ref[i, : probe.ref_bytes]
            if probe.dest_check:
                for i, r in enumerate(parse_results(rows, DEST_SLOT)):
                    if i in idx and check_magic(shard, i, r, MAGIC_DEST, DESTRD_DEST) and r.nd:
                        findings.append(nd_finding(shard, i, r, DESTRD_DEST, probe.out))
        if probe.dest_check:
            findings += _dest_consensus(arm, p, probe, refs, ref_rows, device_ids, runner, notes)
        if golden is not None:  # make_inputs only returns a golden when it is exact
            g32 = golden.view(np.uint32)
            for (shard, i), ref in refs.items():
                bad = np.nonzero(ref.view(np.uint32) != g32)[0]
                if len(bad):
                    findings.append(
                        _stable_finding(arm, p, "golden", device_ids, shard, runner.cores[i], ref, g32, bad, probe)
                    )
        hashes = {k: hashlib.blake2b(v.tobytes(), digest_size=16).digest() for k, v in refs.items()}
        if hashes:
            top, count = Counter(hashes.values()).most_common(1)[0]
            if count < len(hashes):
                major = next(refs[k] for k, h in hashes.items() if h == top).view(np.uint32)
                for k, h in hashes.items():
                    if h != top:
                        bad = np.nonzero(refs[k].view(np.uint32) != major)[0]
                        findings.append(
                            _stable_finding(
                                arm,
                                p,
                                "consensus",
                                device_ids,
                                k[0],
                                runner.cores[k[1]],
                                refs[k],
                                major,
                                bad,
                                probe,
                                count,
                                len(hashes),
                            )
                        )

    # DM probes: self-checking, no host data to compare.
    if arm.extra:
        magic = MAGIC_L1 if arm.extra == "L1_MARCH" else MAGIC_NOC
        for shard, rows in res_rows.items():
            for i, r in enumerate(parse_results(rows, 0)):
                if arm.extra in NOC_EXTRAS and runner.cores[i][0] == 0:
                    continue  # senders do not check
                if check_magic(shard, i, r, magic, arm.extra) and r.nd:
                    findings.append(nd_finding(shard, i, r, arm.extra, "u32"))
    return findings, notes


def _dest_consensus(arm, p, probe, refs, ref_rows, device_ids, runner, notes):
    """Cross-core check of iteration 0's DEST copy, which the run-to-run check cannot judge. Also notes how
    many cores' DEST copy equals their packed copy byte for byte: when that holds on a healthy grid, a core
    whose two copies differ went wrong between DEST and L1."""
    dest = {k: ref_rows[k[0]][k[1], probe.ref_bytes : probe.ref_bytes + probe.dest_ref_bytes] for k in refs}
    same = sum(np.array_equal(dest[k], refs[k]) for k in refs)
    notes.append(f"{arm.name}: DEST copy equals the packed copy on {same}/{len(refs)} cores")
    hashes = {k: hashlib.blake2b(v.tobytes(), digest_size=16).digest() for k, v in dest.items()}
    top, count = Counter(hashes.values()).most_common(1)[0]
    if count == len(hashes):
        return []
    major = next(dest[k] for k, h in hashes.items() if h == top).view(np.uint32)
    out = []
    for k, h in hashes.items():
        if h != top:
            bad = np.nonzero(dest[k].view(np.uint32) != major)[0]
            f = _stable_finding(
                arm, DESTRD_DEST, "consensus", device_ids, k[0], runner.cores[k[1]], dest[k], major, bad, probe
            )
            f.detail |= {
                "majority": f"{count}/{len(hashes)}",
                "dest_equals_packed": bool(np.array_equal(dest[k], refs[k])),
            }
            out.append(f)
    return out


def _stable_finding(arm, p, kind, device_ids, shard, core, ref, want32, bad, probe, majority=None, total=None):
    got32 = ref.view(np.uint32)
    xor = int(np.bitwise_or.reduce(got32[bad] ^ want32[bad])) if len(bad) else 0
    if probe.out == "bf16":
        xor = (xor & 0xFFFF) | (xor >> 16)
    words_per_tile = FMTS[probe.out].tile_bytes // 4
    detail = {
        "bad_words": int(len(bad)),
        "first_word": int(bad[0]),
        "first_got": f"{int(got32[bad[0]]):#010x}",
        "first_want": f"{int(want32[bad[0]]):#010x}",
        "bits": bit_fields(xor, probe.out),
        "tiles": _ranges(sorted({int(w) // words_per_tile for w in bad})),
    }
    if majority is not None:
        detail["majority"] = f"{majority}/{total}"
    return Finding(arm.name, p, arm.loaded, device_ids[shard], shard, core, kind, detail)


def localize(findings, ran):
    """Per (device, core): which probes failed, which passed, and the stages left standing.

    ran: {probe name: {"quiet", "loaded"}} conditions each probe ran in. Suspects are the stages every
    failing probe shares, minus what a passing probe clears. A probe that failed anywhere does not clear.
    """
    by_core = {}
    for f in findings:
        by_core.setdefault((f.device, f.core), []).append(f)
    verdicts = []
    for (dev, core), fs in sorted(by_core.items()):
        failed = {}
        for f in fs:
            failed.setdefault(f.probe, set()).add("loaded" if f.loaded else "quiet")
        passed = [p for p in ran if p not in failed]
        common = set.intersection(*(set(stages_of(p)) for p in failed))
        cleared = set().union(*(clears(p, failed) for p in passed)) if passed else set()
        suspects = common - cleared
        if not suspects:
            suspects = set().union(*(set(stages_of(p)) for p in failed)) - cleared
        # Every stage cleared, e.g. MM_DEST fails while MM_L1ACC passes: no single block, but the failing
        # probes' operating point (accumulation depth, duty cycle, format) is the lead.
        specific_to = [] if suspects else sorted(failed)
        didt_only = all(failed[p] == {"loaded"} and "quiet" in ran.get(p, ()) for p in failed)
        kinds = sorted({f.kind for f in fs})
        # Every later iteration disagreeing with iteration 0, while iteration 0 disagrees with the other
        # cores: the reference copy was the bad one, not the rest of the run.
        stable = {(f.arm, f.probe) for f in fs if f.kind != "ND"}
        iter0 = sorted(
            {
                f.probe
                for f in fs
                if f.kind == "ND" and f.detail["bad_iters"] == f.detail["iters_checked"] and (f.arm, f.probe) in stable
            }
        )
        verdicts.append(
            {
                "device": dev,
                "core": list(core),
                "failed": {p: sorted(c) for p, c in failed.items()},
                "passed": sorted(passed),
                "kinds": kinds,
                "suspects": sorted(suspects),
                "specific_to": specific_to,
                "didt_only": didt_only,
                "iter0_outlier": iter0,
                "dest_vs_pack": _dest_vs_pack(failed, ran),
            }
        )
    return verdicts


def _dest_vs_pack(failed, ran):
    """Where MM_DESTRD's two records (DEST before the pack, the packed output) put the fault."""
    if "MM_DESTRD" not in ran:
        return None
    packed, dest = "MM_DESTRD" in failed, DESTRD_DEST in failed
    if packed and dest:
        return "in DEST before the pack (unpack, FPU, DEST)"
    if packed:
        return "after DEST (packer, L1 write, checker read)"
    if dest:
        return "DEST RISC read path only"
    return "MM_DESTRD clean"


def env_seconds(default=20.0):
    return float(os.environ.get("ND_PROBE_SECONDS", default))


# Blackhole Galaxy: the high nibble of a chip's PCIe bus names its UBB tray slot, the low nibble (1-8) its place
# on the tray. Same numbering as tt-metal's control plane and syseng's sys-triage; slot N's FRU is UBB{N-1}_FRU.
UBB_SLOT = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}


def parse_fru(text):
    """{FRU name: {field: value}} from `ipmitool fru print` output."""
    frus, cur = {}, None
    for line in text.splitlines():
        key, sep, val = line.partition(":")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if key == "FRU Device Description":
            cur = frus.setdefault(val.split(" (")[0], {})
        elif cur is not None:
            cur[key] = val
    return frus


def _read_fru():
    """(fru text, source). The UBB FRUs need root: ND_FRU_FILE is a saved `ipmitool fru print` from the host."""
    path = os.environ.get("ND_FRU_FILE")
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read(), f"{path} ({(time.time() - os.path.getmtime(path)) / 60:.0f} min old)"
    try:
        r = subprocess.run(["sudo", "-n", "ipmitool", "fru", "print"], capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "", f"none ({type(e).__name__})"
    # ipmitool exits non-zero when any one FRU is unreadable, so judge by the output.
    if "FRU Device Description" in r.stdout:
        return r.stdout, "sudo ipmitool fru print"
    return "", f"none ({(r.stderr.strip().splitlines() or ['no output'])[-1]})"


def identity_from_descriptor(desc, device_ids, frus):
    """{device id: location and ASIC ID} from a parsed cluster descriptor (all values strings) and parse_fru()."""
    pcie = {k: v for d in desc.get("chips_with_mmio") or [] for k, v in d.items()}
    out = {}
    for dev in device_ids:
        k = str(dev)
        rec = {"pcie_dev": int(pcie[k]) if k in pcie else None, "bdf": (desc.get("chip_pci_bdfs") or {}).get(k)}
        uid = (desc.get("chip_unique_ids") or {}).get(k)
        rec["asic_id"] = f"{int(uid):016X}" if uid is not None else None
        bus = (desc.get("chip_to_bus_id") or {}).get(k)
        if bus is not None:
            bus = int(bus, 16)
            slot = UBB_SLOT.get(bus & 0xF0)
            rec |= {"bus": f"{bus:#04x}", "ubb": slot, "tray_pos": bus & 0x0F}
            rec["tray_serial"] = frus.get(f"UBB{slot - 1}_FRU", {}).get("Board Serial") if slot else None
        out[dev] = rec
    return out


def device_identity(device_ids):
    """{device id: PCIe address, UBB tray slot and serial, place on the tray, ASIC ID}; {} when unavailable.

    Device ids and PCIe numbers are enumeration order and change between boots and hosts. The ASIC ID is unique per
    die and the tray serial per tray, so those two name the hardware for syseng. Never raises: identity is
    reporting only.
    """
    try:
        with open(ttnn.cluster.serialize_cluster_descriptor()) as f:
            # BaseLoader keeps every value a string; YAML 1.1 would read a BDF like 0000:01:00.0 as a number.
            desc = yaml.load(f, Loader=yaml.BaseLoader)
        text, source = _read_fru()
        identity = identity_from_descriptor(desc, device_ids, parse_fru(text))
    except Exception as e:
        logger.warning(f"device identity unavailable: {type(e).__name__}: {e}")
        return {}
    logger.info(f"device identity (tray serials from {source}):")
    for dev, r in identity.items():
        logger.info(f"  dev {dev:2d} pcie {str(r['pcie_dev']):>2}  {where(identity, dev)}")
    return identity


def where(identity, dev):
    """One-line location of a device for logs, e.g. UBB3 ASIC6 tray T38S7TBA... 0000:c6:00.0 ASIC ID 09D1..."""
    r = identity.get(dev)
    if not r:
        return ""
    parts = [f"UBB{r['ubb']} ASIC{r['tray_pos']}" if r.get("ubb") else None]
    parts += [f"tray {r['tray_serial']}" if r.get("tray_serial") else None, r.get("bdf")]
    parts += [f"ASIC ID {r['asic_id']}" if r.get("asic_id") else None]
    return " ".join(p for p in parts if p)
