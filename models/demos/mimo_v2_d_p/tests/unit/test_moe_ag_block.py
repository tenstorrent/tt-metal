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
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_x_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.moe_ag import MoeAgBlock, flat_rows
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


def _block(mesh_device, S, seed=3, topk=None):
    """A MoeAgBlock with gathered random tokens / routing (or the given top-k [T, K]) and a random row-major y (the
    worst-case flat rows, generated on device)."""
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc = E // n_dev
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]
    T = rows * S
    buf_rows = flat_rows(T, K, epc)
    blk = MoeAgBlock.get(mesh_device, chunk_size_per_chip=S, hidden=H, k=K, n_global=E, gids=gids, buf_rows=buf_rows)
    gen = torch.Generator().manual_seed(seed)
    row_shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, None))
    dev = lambda t, dtype, layout, mapper: ttnn.from_torch(
        t, device=mesh_device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )
    x = dev(torch.randn(rows, 1, S, H, generator=gen), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_shard)
    it = torch.rand(T, E, generator=gen).argsort(-1)[:, :K] if topk is None else topk
    idx = dev(it.reshape(rows, 1, S, K).to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, row_shard)
    w = dev(torch.rand(rows, 1, S, K, generator=gen), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, row_shard)
    y = ttnn.rand(
        [1, 1, buf_rows, H], device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, low=-1, high=1
    )
    x_rm = blk.to_rm(x)
    blk.gather(x_rm, idx, w)
    return blk, y, dict(topk=it, gids=gids, epc=epc, rows=buf_rows)


def _moe(mesh_device, S, return_state=False, options=None):
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
        options=options,
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
    return (moe, h, sd) if return_state else (moe, h)


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("S", SEQS, ids=[f"S{s}" for s in SEQS])
def test_moe_ag_programs_determinism(mesh_device, device_params, S):
    blk, y, _ = _block(mesh_device, S)
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
def test_moe_ag_adversarial_routing(mesh_device, device_params):
    """Every token of each mesh column picks the same K experts, all local to chip (0, 0): its flat space holds all
    T K pairs (the worst case the flat rows are sized for). Route plan exact vs the host reference on every chip,
    the reduced output vs a host sum."""
    from models.demos.mimo_v2_d_p.tt.moe_ag import NONE, RoutePlan

    S = 640
    rows, cols = tuple(mesh_device.shape)
    n_dev, T = rows * cols, rows * S
    epc = E // n_dev
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    hot = [int(g) for g in table[0, 0]][:K]  # chip (0, 0)'s first K experts
    topk = torch.tensor(hot).expand(T, K).contiguous()
    blk, y, h = _block(mesh_device, S, topk=topk)
    counts, regions, tidx, yslot = blk.plan()
    for d in range(n_dev):
        lmap = torch.full((E,), NONE, dtype=torch.int64)
        for l, g in enumerate(h["gids"][d]):
            lmap[g] = l
        c_ref, r_ref, t_ref, y_ref, used = RoutePlan.reference(topk, lmap, epc, h["rows"])
        u32 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[d]).to(torch.int64).reshape(-1) & 0xFFFFFFFF
        assert torch.equal(u32(counts), c_ref) and torch.equal(u32(regions), r_ref), f"chip {d}: counts / regions"
        assert torch.equal(u32(tidx)[:used], t_ref[:used]), f"chip {d}: token_index"
        assert torch.equal(u32(yslot), y_ref.reshape(-1)), f"chip {d}: y_slot"
        if d == 0:
            assert used <= h["rows"], (used, h["rows"])
            logger.info(f"adversarial: chip 0 uses {used} of {h['rows']} flat rows (T K = {T * K})")
    out = blk.reduce(y)
    # host: every pair lives on chip (0, 0); the send-back and the TP all-reduce bring its partials to every chip
    wg = ttnn.to_torch(ttnn.get_device_tensors(blk.gw)[0]).float().reshape(T, K)
    y0 = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float().reshape(h["rows"], H)
    ys0 = (ttnn.to_torch(ttnn.get_device_tensors(yslot)[0]).to(torch.int64) & 0xFFFFFFFF).reshape(T, K)
    col0 = (wg[:, :, None] * y0[ys0]).sum(1)  # [T, H]: every pair of column 0 is on chip (0, 0)
    for d in range(n_dev):
        r, c = divmod(d, cols)
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[d]).float().reshape(S, H)
        ref = col0[r * S : (r + 1) * S]
        a, b = got.double().flatten(), ref.double().flatten()
        pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        assert pcc > 0.9999, (d, pcc)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((1, 4), torus_x_device_params(), id="1x4")],
    indirect=["mesh_device", "device_params"],
)
def test_moe_block_single_row(mesh_device, device_params):
    """One mesh row (no dispatch axis: the block uses x / top-k / weights ungathered, so they must outlive the expert
    and the reduce), through TtMoE with layer 1's real weights, vs the HF MoE (fp32) on the same input."""
    _check_vs_hf(mesh_device, "1x4")


def _check_vs_hf(mesh_device, label, options=None):
    from models.demos.mimo_v2_d_p.reference import hf

    S = 640
    moe, h, sd = _moe(mesh_device, S, return_state=True, options=options)
    out = moe(h)
    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(S, H)
    h_host = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).float().reshape(1, S, H)
    mlp = hf.decoder_layer(1, sd, dtype=torch.float32).mlp
    with torch.no_grad():
        ref = mlp(h_host)
    ref = (ref[0] if isinstance(ref, tuple) else ref).reshape(S, H)
    a, b = got.double().flatten(), ref.double().flatten()
    pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    logger.info(f"{label} MoE vs HF: pcc {pcc:.5f}")
    assert pcc > 0.97, pcc  # bf4 experts: ~0.98-0.99 per layer (README)


@pytest.mark.timeout(3600)
@MESH_PARAMS
def test_moe_py_expert_dispatch(mesh_device, device_params):
    """routed_expert="py" (the Python FlatExpert builder, materialized weights) on the dispatch / combine path."""
    from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions

    _check_vs_hf(mesh_device, "2x2 py expert", MiMoRuntimeOptions(routed_expert="py", moe_ag=False))


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
    blk, y, _ = _block(mesh_device, S)
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
