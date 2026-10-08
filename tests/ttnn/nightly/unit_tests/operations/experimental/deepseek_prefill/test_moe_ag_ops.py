# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.deepseek_prefill.moe_ag_* (the all-gather MoE block's programs) on one device, over shapes:
correctness vs host references, program-cache hits with moved buffers, on-device determinism, device perf.

Shape families: MiMo-V2 2x2 (H 4096, top-8 of 256, 64 experts / chip), Galaxy 8x4 (8 experts / chip, 8 mesh rows of
640 tokens), Kimi K2 (H 7168, 384 experts), a small top-6 / 160-expert case, and the deepseek_v3_d_p models' all-gather
MoE block shapes: LoudBox 2 x 4 (two mesh rows of 640 tokens) for Kimi-K2.7 (48 experts / chip), GLM-5.3 (H 6144,
32 / chip) and Kimi-K3 (its 3584 latent, top-16; 512 experts so 64 / chip fit the flat expert), and Kimi-K3 on the
Galaxy (896 experts, 28 / chip, 8 mesh rows). K3's H 3584 is not a multiple of 1024: only the ops the deepseek block
runs (route plan, local reduce) take it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import assert_op_duration_merged, require_realtime_profiler

NONE = 0xFFFFFFFF
ops = ttnn.experimental.deepseek_prefill

# (tag, H, K, NG, EPC, T tokens of the column, S tokens per chip)
CASES = [
    ("mimo", 4096, 8, 256, 64, 1280, 640),
    ("galaxy", 4096, 8, 256, 8, 5120, 640),
    ("k2", 7168, 8, 384, 48, 2048, 1024),
    ("small", 2048, 6, 160, 20, 256, 128),
    ("k27_lb", 7168, 8, 384, 48, 1280, 640),
    ("glm_lb", 6144, 8, 256, 32, 1280, 640),
    ("k3_lb", 3584, 16, 512, 64, 1280, 640),
    ("k3_glx", 3584, 16, 896, 28, 5120, 640),
]
IDS = [c[0] for c in CASES]
DET_ITERS = 20


def _up(v, a):
    return -(-v // a) * a


def _dev(device, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _u32(t):
    return ttnn.to_torch(t).to(torch.int64).reshape(-1) & 0xFFFFFFFF


def _plan_reference(idx, lmap, epc, rows):
    """counts / regions [NG], token_index [rows] (used prefix), y_slot [T K], used rows."""
    T, K = idx.shape
    NG = lmap.shape[0]
    lists = [[] for _ in range(epc)]
    for g in range(T):
        for k in range(K):
            l = int(lmap[int(idx[g, k])])
            if l < epc:
                lists[l].append((g, k))
    counts, regions = torch.zeros(NG, dtype=torch.int64), torch.zeros(NG, dtype=torch.int64)
    tidx = torch.zeros(rows, dtype=torch.int64)
    yslot = torch.full((T * K,), NONE, dtype=torch.int64)
    region = 0
    for l in range(epc):
        gid = int((lmap == l).nonzero()[0])
        counts[gid], regions[gid] = len(lists[l]), region
        for i, (g, k) in enumerate(lists[l]):
            tidx[region + i] = g
            yslot[g * K + k] = region + i
        region += _up(len(lists[l]), 32)
    return counts, regions, tidx, yslot, region


def _worst_rows(T, K, EPC):
    pairs = T * min(K, EPC)
    return _up(pairs, 32) + 32 * (min(pairs, EPC) - 1)


def _plan_inputs(device, case, seed=0, adversarial=False):
    """Random top-k, or (adversarial) every token picking min(K, EPC) of this chip's experts (the worst case)."""
    tag, H, K, NG, EPC, T, S = case
    gen = torch.Generator().manual_seed(seed)
    gids = torch.randperm(NG, generator=gen)[:EPC]
    if adversarial:
        other = torch.tensor([g for g in range(NG) if g not in set(gids.tolist())])
        n_loc = min(K, EPC)
        idx = torch.cat([gids[:n_loc].expand(T, n_loc), other[: K - n_loc].expand(T, K - n_loc)], 1)
    else:
        idx = torch.rand(T, NG, generator=gen).argsort(-1)[:, :K]
    lmap = torch.full((NG,), NONE, dtype=torch.int64)
    lmap[gids] = torch.arange(EPC)
    rows = _worst_rows(T, K, EPC)
    d = dict(
        idx=_dev(device, idx.reshape(1, 1, T, K).to(torch.int32), ttnn.uint16),
        lmap=_dev(device, lmap.reshape(1, 1, 1, NG).to(torch.int32), ttnn.uint32),
        rows=rows,
    )
    return d, idx, lmap


def _marker(reference, actual):
    tile = lambda t: t if t.layout == ttnn.TILE_LAYOUT else ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    return ttnn.max(ttnn.ne(tile(reference), tile(actual), dtype=ttnn.bfloat16))


def _deterministic(run, n=DET_ITERS, views=None):
    """run() -> list of outputs; n launches compared on device (bitwise) against the first, one read at the end.
    views[i] (optional): restricts output i to its specified part (e.g. the used rows) before the compare."""
    view = lambda i, t: views[i](t) if views and views[i] else t
    ref = [ttnn.clone(view(i, t)) for i, t in enumerate(run())]
    marker = None
    for _ in range(n):
        for i, (r, t) in enumerate(zip(ref, run())):
            mk = _marker(r, view(i, t))
            marker = mk if marker is None else ttnn.maximum(marker, mk)
    return float(ttnn.to_torch(marker).item()) == 0.0


def _skip_unless_h1024(case):
    if case[1] % 1024:
        pytest.skip(f"H {case[1]} % 1024 != 0: this op stages 1024-column blocks (the deepseek block does not run it)")


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


# ---------------------------------------------------------------- route plan
@pytest.mark.parametrize("adversarial", [False, True], ids=["random", "adversarial"])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_moe_ag_route_plan(device, case, adversarial):
    tag, H, K, NG, EPC, T, S = case
    d, idx, lmap = _plan_inputs(device, case, adversarial=adversarial)
    counts, regions, tidx, yslot, used = _plan_reference(idx, lmap, EPC, d["rows"])
    if adversarial:  # every token's pairs local: the flat space is full up to the region padding
        assert used == min(K, EPC) * _up(T, 32), (used, T, K, EPC)
    for attempt in range(2):  # the second launch is a program-cache hit on freshly allocated outputs
        out = ops.moe_ag_route_plan(d["idx"], d["lmap"], EPC, d["rows"])
        assert torch.equal(_u32(out[0]), counts), f"{tag}: counts"
        assert torch.equal(_u32(out[1]), regions), f"{tag}: regions"
        assert torch.equal(_u32(out[2])[:used], tidx[:used]), f"{tag}: token_index"
        assert torch.equal(_u32(out[3]), yslot), f"{tag}: y_slot"
    pre = [ttnn.clone(t) for t in out]
    ops.moe_ag_route_plan(d["idx"], d["lmap"], EPC, d["rows"], outputs=pre)
    assert torch.equal(_u32(pre[3]), yslot), f"{tag}: y_slot (preallocated)"
    used_rows = lambda t: ttnn.slice(t, [0, 0], [1, used])  # token_index past the used regions is unspecified
    assert _deterministic(
        lambda: ops.moe_ag_route_plan(d["idx"], d["lmap"], EPC, d["rows"]), views=[None, None, used_rows, None]
    ), f"{tag}: determinism"


# ---------------------------------------------------------------- local reduce
def _reduce_inputs(device, case, seed=1):
    tag, H, K, NG, EPC, T, S = case
    d, idx, lmap = _plan_inputs(device, case, seed)
    plan = ops.moe_ag_route_plan(d["idx"], d["lmap"], EPC, d["rows"])
    gen = torch.Generator().manual_seed(seed)
    y = torch.randn(d["rows"], H, generator=gen).bfloat16()
    w = torch.rand(T, K, generator=gen).bfloat16()
    info = torch.zeros(16, dtype=torch.int64)
    info[0], info[1], info[2] = 0, S, 0  # mesh row 0 of 2 (phases / split use T == 2 S)
    ys = _u32(plan[3]).reshape(T, K)
    m = ys != NONE
    gi, ki = m.nonzero(as_tuple=True)
    partial = torch.zeros(T, H)
    partial.index_add_(0, gi, w[gi, ki, None].float() * y[ys[gi, ki]].float())
    peer = torch.randn(2 * S, H, generator=gen).bfloat16()
    dd = dict(
        y=_dev(device, y.reshape(1, 1, -1, H), ttnn.bfloat16),
        ys=plan[3],
        w=_dev(device, w.reshape(1, 1, T, K), ttnn.bfloat16),
        info=_dev(device, info.reshape(1, 1, 1, 16).to(torch.int32), ttnn.uint32),
        peer=_dev(device, peer.reshape(1, 1, 2 * S, H), ttnn.bfloat16),
    )
    return dd, partial, peer


def _check(label, got, ref):
    got = ttnn.to_torch(got).float().reshape(ref.shape)
    p = _pcc(got, ref)
    rel = float((got - ref).norm() / ref.norm().clamp_min(1e-9))
    logger.info(f"{label}: pcc {p:.7f} rel {rel:.2e}")
    assert p > 0.9999 and rel < 1e-2, (label, p, rel)


@pytest.mark.parametrize("mode", ["plain", "tiled", "split", "phases"])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_moe_ag_local_reduce(device, case, mode):
    tag, H, K, NG, EPC, T, S = case
    if mode in ("split", "phases") and T != 2 * S:
        pytest.skip("split / phases need two mesh rows (T == 2 S)")
    d, partial, peer = _reduce_inputs(device, case)
    args = (d["y"], d["ys"], d["w"], d["info"], S)
    if mode in ("plain", "tiled"):
        run = lambda: ops.moe_ag_local_reduce(*args, tiled=mode == "tiled")
        for _ in range(2):
            (out,) = run()
            _check(f"{tag} {mode}", out, partial)
    elif mode == "split":
        run = lambda: ops.moe_ag_local_reduce(*args, split=True)
        own, other = run()
        _check(f"{tag} split own", own, partial[:S])
        _check(f"{tag} split other", other, partial[S:])
    else:
        (p1,) = ops.moe_ag_local_reduce(*args, phase=1)
        _check(f"{tag} phase 1", p1, partial[S:])
        run = lambda: ops.moe_ag_local_reduce(*args, phase=2, peer=d["peer"])
        (p2,) = run()
        # this row's tokens plus the peer's partial for them (gathered at the other block: rows S .. 2 S)
        _check(f"{tag} phase 2", p2, partial[:S] + peer[S:].float())
    assert _deterministic(run), f"{tag} {mode}: determinism"


# ---------------------------------------------------------------- row ops
@pytest.mark.parametrize("n_blocks", [2, 4, 8])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_moe_ag_sum_rows_tiled(device, case, n_blocks):
    _skip_unless_h1024(case)
    tag, H, K, NG, EPC, T, S = case
    src = torch.randn(n_blocks * S, H).bfloat16()
    s_dev = _dev(device, src.reshape(1, 1, -1, H), ttnn.bfloat16)
    ref = sum(src[i * S : (i + 1) * S].float() for i in range(n_blocks))
    run = lambda: [ops.moe_ag_sum_rows_tiled(s_dev, S, n_blocks, S)]
    _check(f"{tag} sum_rows_tiled N{n_blocks}", run()[0], ref)
    assert _deterministic(run), f"{tag}: determinism"


@pytest.mark.parametrize("info_offset", [False, True])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_moe_ag_add_rows(device, case, info_offset):
    _skip_unless_h1024(case)
    tag, H, K, NG, EPC, T, S = case
    a, b = torch.randn(2 * S, H).bfloat16(), torch.randn(2 * S, H).bfloat16()
    info = torch.zeros(16, dtype=torch.int64)
    info[1] = S // 2
    info_dev = _dev(device, info.reshape(1, 1, 1, 16).to(torch.int32), ttnn.uint32)
    a_dev, b_dev = (_dev(device, t.reshape(1, 1, -1, H), ttnn.bfloat16) for t in (a, b))
    b_off = S // 2 if info_offset else 3
    run = lambda: [
        ops.moe_ag_add_rows(
            a_dev, b_dev, info_dev, S, a_offset=S, b_offset=0 if info_offset else 3, info_offset=info_offset
        )
    ]
    _check(f"{tag} add_rows", run()[0], a[S:].float() + b[b_off : b_off + S].float())
    assert _deterministic(run), f"{tag}: determinism"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_moe_ag_untilize_active(device, case):
    tag, H, K, NG, EPC, T, S = case
    d, idx, lmap = _plan_inputs(device, case, seed=2)
    counts, regions, *_ = ops.moe_ag_route_plan(d["idx"], d["lmap"], EPC, d["rows"])
    y = _dev(device, torch.randn(1, 1, d["rows"], H), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    yh = ttnn.to_torch(y).float().reshape(d["rows"], H)
    W = 32 if H % 1024 == 0 else 16
    out = ops.moe_ag_untilize_active(y, counts, regions, d["lmap"], EPC, tiles_per_block=W)
    oh = ttnn.to_torch(out).float().reshape(d["rows"], H)
    c, r = _u32(counts), _u32(regions)
    for g in range(NG):
        if c[g]:
            r0, n = int(r[g]), _up(int(c[g]), 32)
            assert torch.equal(oh[r0 : r0 + n], yh[r0 : r0 + n]), f"{tag}: expert {g} rows"
    # rows outside the active regions are never written: a zeroed output keeps them defined (no stale NaN)
    out0 = _dev(device, torch.zeros(d["rows"], H), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    assert _deterministic(
        lambda: [ops.moe_ag_untilize_active(y, counts, regions, d["lmap"], EPC, tiles_per_block=W, output=out0)]
    ), f"{tag}: determinism"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_moe_ag_untilize_x(device, case):
    _skip_unless_h1024(case)
    tag, H, K, NG, EPC, T, S = case
    x = torch.randn(1, 1, S, H).bfloat16()
    x_dev = _dev(device, x, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out = ops.moe_ag_untilize_x(x_dev)
    assert torch.equal(ttnn.to_torch(out).reshape(S, H), x.reshape(S, H)), f"{tag}: untilize_x"
    assert _deterministic(lambda: [ops.moe_ag_untilize_x(x_dev)]), f"{tag}: determinism"


# ---------------------------------------------------------------- validation
def test_moe_ag_validation(device, expect_error):
    case = CASES[0]
    d, _, _ = _plan_inputs(device, case)
    with expect_error(RuntimeError, "experts_per_chip"):
        ops.moe_ag_route_plan(d["idx"], d["lmap"], 65, d["rows"])
    with expect_error(RuntimeError, "num_rows"):
        ops.moe_ag_route_plan(d["idx"], d["lmap"], 64, 100)
    counts, regions, *_ = ops.moe_ag_route_plan(d["idx"], d["lmap"], 64, d["rows"])
    y33 = _dev(device, torch.randn(1, 1, 33, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "must be a multiple of 32"):  # untilize writes whole tiles
        ops.moe_ag_untilize_active(y33, counts, regions, d["lmap"], 64, tiles_per_block=32)
    y2x16 = _dev(device, torch.randn(1, 2, 16, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)  # 32 rows over 2 padded slices
    with expect_error(RuntimeError, "must be a multiple of 32"):
        ops.moe_ag_untilize_active(y2x16, counts, regions, d["lmap"], 64, tiles_per_block=32)
    u32 = lambda n: _dev(device, torch.zeros(1, n, dtype=torch.int32), ttnn.uint32)
    y64 = _dev(device, torch.randn(1, 1, 64, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "multiple of 16"):  # [NG] rows at NG x 4 B strides: 64 B NoC alignment
        ops.moe_ag_untilize_active(
            y64,
            u32(8),
            u32(8),
            _dev(device, torch.zeros(1, 1, 1, 8, dtype=torch.int32), ttnn.uint32),
            4,
            tiles_per_block=32,
        )
    with expect_error(RuntimeError, "worst-case flat rows"):  # the dispatch capacity-factor sizing (1280 x 4 + 32 x 63)
        ops.moe_ag_route_plan(d["idx"], d["lmap"], 64, 7136)
    x = _dev(device, torch.randn(1, 1, 64, 1000), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "multiple of 1024"):
        ops.moe_ag_untilize_x(x)


# ---------------------------------------------------------------- perf
# Device ns, median of 3, one BH p150 (QuietBox chip), 2026-09-29 (3 calibration runs, spread <= 2%); recalibrate from the "RT-CAL" lines.
_PERF_EXPECTED_NS = {
    "mimo_route_plan": 29_758,
    "mimo_local_reduce_tiled": 160_110,
    "mimo_sum_rows_tiled": 61_755,
    "mimo_untilize_x": 46_583,
    "mimo_local_reduce_phase2": 75_089,
    "galaxy_route_plan": 47_881,
    "galaxy_local_reduce_tiled": 278_859,
    "galaxy_sum_rows_tiled": 61_219,
    "galaxy_untilize_x": 46_323,
    "k2_route_plan": 34_776,
    "k2_local_reduce_tiled": 288_256,
    "k2_sum_rows_tiled": 129_889,
    "k2_untilize_x": 97_264,
    "k2_local_reduce_phase2": 141_866,
    "small_route_plan": 15_256,
    "small_local_reduce_tiled": 35_477,
    "small_sum_rows_tiled": 18_865,
    "small_untilize_x": 13_154,
    "small_local_reduce_phase2": 15_734,
    # deepseek_v3_d_p shapes: one BH p150b (LoudBox chip), 2026-10-08, median of 3 runs (spread <= 4.3%)
    "k27_lb_route_plan": 28_946,
    "k27_lb_local_reduce_tiled": 198_810,
    "k27_lb_local_reduce_phase2": 93_348,
    "k27_lb_sum_rows_tiled": 87_607,
    "k27_lb_untilize_x": 65_313,
    "glm_lb_route_plan": 24_907,
    "glm_lb_local_reduce_tiled": 171_534,
    "glm_lb_local_reduce_phase2": 78_355,
    "glm_lb_sum_rows_tiled": 75_478,
    "glm_lb_untilize_x": 61_410,
    "k3_lb_route_plan": 40_061,
    "k3_lb_local_reduce_tiled": 142_764,
    "k3_lb_local_reduce_phase2": 69_121,
    "k3_glx_route_plan": 87_231,
    "k3_glx_local_reduce_tiled": 262_882,
}
_PERF_MARGIN = 0.05
_KDIR = "/deepseek_prefill/moe_ag/device/kernels/"


@pytest.mark.parametrize("case", CASES, ids=IDS)
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_moe_ag_ops_perf(device, case):
    require_realtime_profiler("moe_ag op perf checks")
    tag, H, K, NG, EPC, T, S = case
    d, partial, peer = _reduce_inputs(device, case)
    plan_in, _, _ = _plan_inputs(device, case)
    src = _dev(device, torch.randn(1, 1, 2 * S, H), ttnn.bfloat16)
    x = _dev(device, torch.randn(1, 1, S, H), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    runs = {
        "route_plan": (
            "route_plan.cpp",
            lambda: ops.moe_ag_route_plan(plan_in["idx"], plan_in["lmap"], EPC, plan_in["rows"]),
        ),
        "local_reduce_tiled": (
            "reduce_compute_t.cpp",
            lambda: ops.moe_ag_local_reduce(d["y"], d["ys"], d["w"], d["info"], S, tiled=True),
        ),
    }
    if H % 1024 == 0:  # the row ops stage 1024-column blocks
        runs["sum_rows_tiled"] = ("addt_reader.cpp", lambda: ops.moe_ag_sum_rows_tiled(src, S, 2, S))
        runs["untilize_x"] = ("untilize_x_reader.cpp", lambda: ops.moe_ag_untilize_x(x))
    if T == 2 * S:
        runs["local_reduce_phase2"] = (
            "reduce2_reader.cpp",
            lambda: ops.moe_ag_local_reduce(d["y"], d["ys"], d["w"], d["info"], S, phase=2, peer=d["peer"]),
        )
    missing = []
    for name, (kernel, fn) in runs.items():
        label = f"{tag}_{name}"
        expected = _PERF_EXPECTED_NS.get(label)
        if expected is None:
            missing.append(label)
        assert_op_duration_merged(
            device,
            fn,
            _KDIR + kernel,
            expected_ns=expected or 1,
            margin=_PERF_MARGIN if expected else float("inf"),
            label=f'"{label}"',
            iters=3,
        )
    if missing:
        pytest.skip(f"no baseline for {missing}; add them to _PERF_EXPECTED_NS")
