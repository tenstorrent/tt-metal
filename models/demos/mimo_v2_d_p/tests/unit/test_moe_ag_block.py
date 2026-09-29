# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Determinism and device perf of the all-gather MoE block (tt/moe_ag.py) on the QuietBox 2x2.

* Block programs (route plan, the two local-reduce phases, the TP add + tilize) on random routing / expert outputs
  (y stands in for the experts, row-major bf16 as the flat expert writes it): N launches compared on device against
  the first (bitwise; markers folded on device, one read at the end), and the device time of each program.
* The MoE block end to end (TtMoE: router, gathers, route plan, flat routed expert, reduce, send-back, TP all-reduce;
  layer 1's real weights on real-token embeddings): the same determinism check on its output, and its device time
  (sum of its programs, median over calls).

Perf baselines: _PERF_EXPECTED_NS, from the "RT-CAL" lines (BH p150 QuietBox, 2026-09-29).
"""

import statistics

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.ffn import moe_capacity_factor
from models.demos.mimo_v2_d_p.tt.moe_ag import MoeAgBlock
from tests.ttnn.profiling.realtime_profiler_utils import (
    assert_op_duration_merged,
    profile_realtime_program_merged,
    require_realtime_profiler,
)

E, K, H = 256, 8, 4096
SEQS = [640, 2048]  # tokens per chip: the Galaxy chunk (5120 / SP8) and the QuietBox 4k chunk
DET_ITERS = 50


def _marker(reference, actual):
    """Non-zero on a chip iff any element of actual differs from reference (exact; uint32 compares exactly)."""
    tile = lambda t: t if t.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    return ttnn.max(ttnn.ne(tile(reference), tile(actual), dtype=ttnn.bfloat16))


def _fold(marker, new):
    return new if marker is None else ttnn.maximum(marker, new)


def _mismatch(marker):
    return [float(ttnn.to_torch(s).item()) for s in ttnn.get_device_tensors(ttnn.from_device(marker))]


def _block(mesh_device, S, seed=3):
    """A MoeAgBlock with gathered random tokens / routing and a random row-major y."""
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc = E // n_dev
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]
    _, _, buf_rows, _ = compute_constants(S, E, K, n_dev, rows, moe_capacity_factor(K, E, n_dev))
    blk = MoeAgBlock.get(mesh_device, chunk_size_per_chip=S, hidden=H, k=K, n_global=E, gids=gids, buf_rows=buf_rows)
    gen = torch.Generator().manual_seed(seed)
    T = rows * S
    row_shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, None))
    dev = lambda t, dtype, layout, mapper: ttnn.from_torch(
        t, device=mesh_device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )
    x = dev(torch.randn(rows, 1, S, H, generator=gen), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_shard)
    it = torch.rand(T, E, generator=gen).argsort(-1)[:, :K]
    idx = dev(it.reshape(rows, 1, S, K).to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, row_shard)
    w = dev(torch.rand(rows, 1, S, K, generator=gen), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, row_shard)
    y = dev(
        torch.randn(rows, cols, buf_rows, H, generator=gen),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    x_rm = blk.to_rm(x)
    blk.gather(x_rm, idx, w)
    return blk, y


def _moe(mesh_device, S):
    """TtMoE for layer 1 (real weights) and its input: real-token embeddings through the post-attention norm."""
    from models.demos.mimo_v2_d_p.reference import hf
    from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
    from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
    from models.demos.mimo_v2_d_p.tt.ffn import TtMoE, TtRMSNorm

    cfg = MiMoTextConfig.from_json()
    sp, tp = tuple(mesh_device.shape)
    sd = layer_state(1, cfg)
    moe = TtMoE(
        mesh_device,
        {k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")},
        cfg,
        seq_len_per_chip=S,
        num_links=3,
        cache_prefix="L1",
    )
    norm = TtRMSNorm(mesh_device, sd["post_attention_layernorm.weight"], cfg.layernorm_epsilon)
    x = ttnn.from_torch(
        global_state()["embed_tokens.weight"][hf.tokenize_prompt(S * sp)].float()[None, None],
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, None)),
    )
    h = norm(x)
    x.deallocate(True)
    return moe, h


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("S", SEQS, ids=[f"S{s}" for s in SEQS])
def test_moe_ag_programs_determinism(mesh_device, device_params, S):
    blk, y = _block(mesh_device, S)
    plan = blk.plan()
    ref_plan = [ttnn.clone(t) for t in plan]
    out = blk.reduce(y)
    ref_parts = [ttnn.clone(blk.lreduce.other), ttnn.clone(blk.lreduce.own), out]
    names = ["counts", "regions", "token_index", "y_slot", "reduce phase 1", "reduce phase 2", "tp add"]
    markers = [None] * len(names)
    for _ in range(DET_ITERS):
        plan = blk.plan()
        for i, (r, t) in enumerate(zip(ref_plan, plan)):
            markers[i] = _fold(markers[i], _marker(r, t))
        out = blk.reduce(y)
        for i, (r, t) in enumerate(zip(ref_parts, [blk.lreduce.other, blk.lreduce.own, out])):
            markers[4 + i] = _fold(markers[4 + i], _marker(r, t))
        out.deallocate(True)
    bad = {n: m for n, m in zip(names, (_mismatch(m) for m in markers)) if any(m)}
    assert not bad, f"S{S}: non-deterministic over {DET_ITERS} launches (per-chip markers): {bad}"


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("S", SEQS, ids=[f"S{s}" for s in SEQS])
def test_moe_block_determinism(mesh_device, device_params, S):
    moe, h = _moe(mesh_device, S)
    ref = moe(h)
    marker = None
    for _ in range(DET_ITERS):
        out = moe(h)
        marker = _fold(marker, _marker(ref, out))
        out.deallocate(True)
    m = _mismatch(marker)
    assert not any(m), f"S{S}: MoE block output non-deterministic over {DET_ITERS} calls (per-chip markers {m})"


# Device ns (max over chips), median of 3, BH p150 QuietBox 2x2 (median of 3 calibration runs, spread <= 2%); recalibrate from the "RT-CAL" lines.
_PERF_EXPECTED_NS = {
    "S640_route_plan": 29_610,
    "S640_reduce_phase1": 58_397,
    "S640_reduce_phase2": 77_515,
    "S640_tp_add_tilize": 62_392,
    "S640_moe_block": 2_854_509,  # sum of its 15 programs
    "S2048_route_plan": 53_533,
    "S2048_reduce_phase1": 154_816,
    "S2048_reduce_phase2": 228_691,
    "S2048_tp_add_tilize": 143_305,
    "S2048_moe_block": 4_729_987,
}
_PERF_MARGIN = 0.05


def _check(label, missing):
    expected = _PERF_EXPECTED_NS.get(label)
    if expected is None:
        missing.append(label)
    return dict(expected_ns=expected or 1, margin=_PERF_MARGIN if expected else float("inf"), label=f'"{label}"')


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("S", SEQS, ids=[f"S{s}" for s in SEQS])
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_moe_ag_programs_perf(mesh_device, device_params, S):
    require_realtime_profiler("MoE all-gather block perf checks")
    blk, y = _block(mesh_device, S)
    blk.plan()
    blk.reduce(y).deallocate(True)  # every program compiled; g_sp / g_tp hold real partials
    lr, ys = blk.lreduce, blk.plan_op.y_slot
    runs = {
        "route_plan": ("/moe_ag/device/kernels/route_plan.cpp", lambda: blk.plan()),
        "reduce_phase1": ("/moe_ag/device/kernels/reduce2_reader.cpp", lambda: lr.phase(y, ys, blk.gw, 1)),
        "reduce_phase2": (
            "/moe_ag/device/kernels/reduce2_reader.cpp",
            lambda: lr.phase(y, ys, blk.gw, 2, peer=blk.g_sp),
        ),
        "tp_add_tilize": ("/moe_ag/device/kernels/addt_reader.cpp", lambda: blk.tp(blk.g_tp).deallocate(True)),
    }
    missing = []
    for name, (path, fn) in runs.items():
        label = f"S{S}_{name}"
        assert_op_duration_merged(mesh_device, fn, path, iters=3, **_check(label, missing))
    if missing:
        pytest.skip(f"no baseline for {missing}; add them to _PERF_EXPECTED_NS")


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("S", SEQS, ids=[f"S{s}" for s in SEQS])
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_moe_block_perf(mesh_device, device_params, S):
    """Device time of one MoE block call: the sum of its programs' durations (each the max over chips)."""
    require_realtime_profiler("MoE block perf checks")
    moe, h = _moe(mesh_device, S)
    for _ in range(2):
        moe(h).deallocate(True)
    ttnn.synchronize_device(mesh_device)
    totals = []
    for _ in range(3):
        _, per_program = profile_realtime_program_merged(mesh_device, lambda: moe(h).deallocate(True))
        totals.append(sum(e["duration_ns"] for e in per_program.values()))
        n_programs = len(per_program)
    median_ns = statistics.median(totals)
    label = f"S{S}_moe_block"
    expected = _PERF_EXPECTED_NS.get(label)
    logger.info(f"RT-CAL {label}: {round(median_ns):_} ns  # sum of {n_programs} programs, median of 3 calls")
    if expected is None:
        pytest.skip(f"no baseline for {label}; add it to _PERF_EXPECTED_NS")
    lower, upper = expected * (1 - _PERF_MARGIN), expected * (1 + _PERF_MARGIN)
    assert lower <= median_ns <= upper, f"{label}: {median_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}]"
