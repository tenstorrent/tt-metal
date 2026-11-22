# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.59 (owner 2026-11-14): the flat routed expert's x snake pieces, EXPERT_X_PIECES legacy vs min8k, at this model's
shapes on the whole mesh.

Terms:
  x block   = one sub-block's x tiles for one K block (MT row tiles x 8 K tiles, bfp8: MT x 8 x 1088 B);
  pieces    = the cut-through pieces the gate/up cores forward an x block in along their snake (legacy: 4 whenever
              aligned; min8k = FlatLayoutOptions.x_min_piece_bytes 8192, Marko Staletovic 1e2ab798da1: the most of
              4 / 2 / 1 of >= 8 KB each);
  M         = tokens of every one of a chip's 12 local experts (every chip the same counts);
  flat ms   = the flat_routed_expert program's device time (real-time profiler, slowest chip), median of REPS calls.

The plan: the routed experts' as the model's MoE block runs them - H, I, 12 experts a chip, bfp4 weights, weights_nk, the
core cap (MAX_WORKER_CORES), capacity max(chunk, 256), and the block's own arena budget (the combine fits the flat plan
into it). Both sides share one plan and one set of weight tensors: only the program's piece count differs, so the
outputs must be bit-identical.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_expert_x_pieces.py
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec

S = spec()
MS = (64, 256, 512, 1024)  # tokens an expert
REPS = 5
LEAD = 40  # matmuls (11 x 10 cores, rule 18) ahead of each timed window (DS41-248)
K_FLAT = "flat_routed_expert/device/kernels"
SIDES = {"legacy": 0, "min8k": 8192}
MESH = pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
BOX = pytest.mark.parametrize("device_params", [harness.device_params(S)], indirect=True, ids=["box"])


def pieces(x_bytes: int, min_piece: int) -> int:
    """The program factory's choice (flat_routed_expert_program_factory.cpp)."""
    aligned = lambda n: x_bytes % n == 0 and (x_bytes // n) % 16 == 0  # noqa: E731
    if min_piece == 0:
        return 4 if aligned(4) else (2 if aligned(2) else 1)
    return next((n for n in (4, 2) if aligned(n) and x_bytes // n >= min_piece), 1)


@BOX
@MESH
@pytest.mark.timeout(3600)
def test_expert_x_pieces(mesh_device):
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt import cores
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings
    from models.demos.moe_block_main.tt.flat_gather import flat_weights_module
    from models.demos.moe_block_main.tt.moe_block import MoEBlock
    from tt_py_test_utils_common.realtime_profiler_utils import profile_realtime_program_merged

    t_start = time.time()
    mb, fw = ad._mb(), flat_weights_module()
    shape = tuple(mesh_device.shape)
    N = shape[0] * shape[1]
    cores.init_caps(mesh_device)
    ad._apply_core_cap(mesh_device)
    mcfg = hooks._config(S)
    chunk = int(S.get("target.chunk"))
    H, I, NE = int(mcfg.dim), int(mcfg.inter), int(mcfg.n_experts)
    E = NE // N
    dims, act_name = ad.block_dims(mcfg)
    act = fw.ACTS[act_name]
    dtypes = ad.block_dtypes()
    wdtype = "bf8" if dtypes.routed_weights == ttnn.bfloat8_b else "bf4"
    # the model's block (for its arena budget and capacity), as moe_block_main_adapter.block builds it
    bcfg = mb.MoEBlockConfig(chunk_tokens=chunk, num_links=ad.LINKS, combine_snap=ad.COMBINE_SNAP,
                             combine_snap_split=settings.get("COMBINE_SPLIT") == "on", persist_l1_small=True)  # fmt: skip
    res = mb.MoEBlockResources(mesh_device, dims, bcfg, dtypes)
    blk = MoEBlock(mesh_device, res, dtypes=dtypes, activation=act, max_tokens_per_expert=max(chunk, 256),
                   layout=mb.FlatLayoutOptions(weights_nk=True))  # fmt: skip
    m_cap = int(blk.expert_kw["max_tokens_per_expert"])
    sh = blk.arena_spec.memory_config.shard_spec.shape
    arena = int(sh[0] * 64)  # (moe_block_main_adapter's log: the arena's bytes a core)
    rec: dict = {"H": H, "I": I, "E": E, "m": m_cap, "arena_bytes": arena, "cap": str(mb.max_grid())}

    g = torch.Generator().manual_seed(59)
    local = [
        (
            torch.randn(I, H, generator=g) * H**-0.5,
            torch.randn(I, H, generator=g) * H**-0.5,
            torch.randn(H, I, generator=g) * I**-0.5,
        )
        for _ in range(E)
    ]  # fmt: skip  ([N, K] as weights_nk takes them)
    gids = [list(range(k * E, (k + 1) * E)) for k in range(N)]

    def expert(min_piece: int, arena_bytes: int, weights):
        return fw.FlatRoutedExpert(
            mesh_device, weights, m=m_cap, H=H, I=I, gids=gids, n_global=NE, wdtype=wdtype, act=act_name, pin=1,
            layout=dict(weights_nk=True, x_min_piece_bytes=min_piece, arena_bytes=arena_bytes),
            keep_packed=not isinstance(weights, str),
        )  # fmt: skip

    # the plan at the block's arena (what the combine runs) and at the whole L1 (the standalone op's own)
    fe = {"legacy": expert(0, arena, [local] * N)}
    fe["min8k"] = expert(SIDES["min8k"], arena, "uninitialized")
    fe["min8k"].w_gu, fe["min8k"].w_d = fe["legacy"].w_gu, fe["legacy"].w_d  # (one set of weights: same plan)
    assert fe["legacy"].lay == fe["min8k"].lay, "x_min_piece_bytes changed the plan"
    for name, arena_b in (("block_arena", arena), ("whole_l1", -1)):
        lay = fe["legacy"].lay if arena_b == arena else expert(0, -1, "uninitialized").lay
        xb = int(lay["mt"]) * 8 * 1088
        rec[f"plan_{name}"] = {"mt": int(lay["mt"]), "x_block_bytes": xb,
                               **{s: {"pieces": pieces(xb, v), "piece_bytes": xb // pieces(xb, v)}
                                  for s, v in SIDES.items()}}  # fmt: skip
        logger.info(f"XP plan {name}: {rec[f'plan_{name}']}")

    lead = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096, generator=g) * 0.01, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )  # fmt: skip
    rows = E * max(MS) + 64
    x = ttnn.from_torch(
        torch.randn(rows, H, generator=g).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )  # fmt: skip
    kw = dict(down_fp32=True)

    def window(fn) -> list:
        def go():
            for _ in range(LEAD):
                ttnn.deallocate(ttnn.matmul(lead, lead, core_grid=ttnn.CoreGrid(y=10, x=11)))
            fn()

        _, per = profile_realtime_program_merged(mesh_device, go)
        return [
            e["duration_ns"] / 1e6
            for e in per.values()
            if any(K_FLAT in s.replace("\\", "/") for s in e["kernel_sources"])
        ]

    fails: list[str] = []
    table = {}
    for M in MS:
        c_ = torch.zeros(N, NE, dtype=torch.int32)
        r_ = torch.zeros(N, NE, dtype=torch.int32)
        for k in range(N):
            for j in range(E):
                c_[k, k * E + j], r_[k, k * E + j] = M, j * (-(-M // 32) * 32)
        up = lambda t: ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,  # noqa
                                       device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0))  # fmt: skip
        cd, rd = up(c_), up(r_)
        ys = {}
        for s in SIDES:
            y = fe[s](x, cd, rd, **kw)  # (warm: compiles)
            ys[s] = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            ttnn.deallocate(y)
        # (the rows the experts write: expert j's M rows from its region; the rest of y is never written)
        mp = -(-M // 32) * 32
        yl = {s: ys[s].reshape(N, -1, ys[s].shape[-1]) for s in SIDES}
        valid = torch.zeros(yl["legacy"].shape[1], dtype=torch.bool)
        for j in range(E):
            valid[j * mp : j * mp + M] = True
        a, b = (yl[s][:, valid] for s in SIDES)
        same = torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
        diff = float((a.float() - b.float()).abs().max())
        logger.info(f"XP M {M}: y {tuple(ys['legacy'].shape)}, {int(valid.sum())} rows a chip compared, same {same}")
        if not same:
            fails.append(f"M {M}: outputs differ, max |diff| {diff}")
        t = {}
        for rep in range(2):  # (two windows, sides alternating: drift shows as a side-order effect)
            for s in SIDES if rep == 0 else reversed(list(SIDES)):

                def run(s=s):
                    for _ in range(REPS):
                        ttnn.deallocate(fe[s](x, cd, rd, **kw))

                w = window(run)
                assert len(w) == REPS, f"{len(w)} flat programs, {REPS} calls"
                t.setdefault(s, []).extend(w)
        med = {s: statistics.median(v) for s, v in t.items()}
        table[M] = {"legacy_ms": round(med["legacy"], 4), "min8k_ms": round(med["min8k"], 4),
                    "delta_pct": round(100 * (med["min8k"] / med["legacy"] - 1), 2), "bit_identical": same,
                    "spread_ms": {s: round(max(v) - min(v), 4) for s, v in t.items()}}  # fmt: skip
        logger.info(f"XP M {M}: {table[M]}")
        metrics.record(f"x_pieces_flat_ms_legacy_M{M}", table[M]["legacy_ms"])
        metrics.record(f"x_pieces_flat_ms_min8k_M{M}", table[M]["min8k_ms"])
        metrics.record(f"x_pieces_flat_delta_pct_M{M}", table[M]["delta_pct"])
        ttnn.deallocate(cd)
        ttnn.deallocate(rd)
    rec["table"] = {str(k): v for k, v in table.items()}
    rec["fails"] = fails
    rec["wall_s"] = round(time.time() - t_start, 1)
    metrics.record("x_pieces_bit_identical", int(not fails))
    metrics.record_json("x_pieces", rec)
    logger.info(f"XP summary: {rec}")
    assert not fails, fails
