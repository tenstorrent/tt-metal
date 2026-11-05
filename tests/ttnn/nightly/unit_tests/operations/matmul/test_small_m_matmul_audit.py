# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Coverage audit for ttnn.experimental.small_m_matmul: paths the main suite does not reach.

WHY THIS FILE EXISTS. Running the main 111-test suite with the factory's own config log
(TT_LOGGER_LEVEL=Debug, grep small_m_cfg) showed which production paths it actually reaches:

    99 programs  reduction=chain  placement=bank-local
    10 programs  reduction=chain  placement=in1-near
     1 program   reduction=chain  placement=mesh
     0 programs  reduction=reduce-scatter        <-- never exercised

Ring reduce-scatter ships on 14 of the 66 corpus shapes, so "111/111 pass" was saying nothing about it.
The 2D mesh placement was covered by a single program, and Ns>1 by a single config. This file closes those
gaps and adds the program-cache DISCRIMINATION cases the main suite lacks (it only ever replays the SAME
program twice, which cannot catch cross-serving between distinct entries).

Shapes/configs below are the ones the ground-truth sweep showed select each path when this file was written.

DELIBERATELY NOT ASSERTED: which reduction or placement a given shape ends up on. That is picker/gate POLICY --
a tuning decision we expect to keep changing -- and pinning it in a test would make every future policy change
look like a test failure. These tests therefore assert only CORRECTNESS, which must hold on whatever path the
gates choose.

The cost of that choice is that coverage here can drift silently: if a policy change moves these shapes onto
another path, the tests still pass but stop covering the path they were written for. Coverage is therefore
re-checked deliberately rather than continuously -- run the suite with the factory's config log and read off
which paths were actually taken:

    TT_LOGGER_LEVEL=Debug pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_small_m_matmul_audit.py \
        | grep -o 'reduction=\S* placement=\S*' | sort | uniq -c

When this file was written that reported 19 reduce-scatter programs (11 in1-near, 5 bank-local, 3 mesh) and 17
chain, i.e. all six reduction x placement combinations. Re-run it after any gate change and pick new shapes if a
combination has dropped out.
"""
import torch

import pytest

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.999


def _mm(device, M, K, N, cfg=None, seed=0):
    """Run the op and return (result, fp32 reference)."""
    torch.manual_seed(seed)
    t0 = torch.randn(1, 1, M, K)
    t1 = torch.randn(1, 1, K, N)
    ref = (t0.bfloat16().float() @ t1.bfloat16().float())[0, 0]
    a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_small_m_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    conf = None
    if cfg is not None:
        Pk, Ns, Sm, kb, nsb = cfg
        conf = ttnn.SmallMMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    out = ttnn.experimental.small_m_matmul(a0, a1, config=conf)
    return ttnn.to_torch(ttnn.from_device(out))[0, 0].float(), ref


# ---------------------------------------------------------------------------------------------------
# 1. RING REDUCE-SCATTER. Zero coverage in the main suite. These shapes/configs were chosen because they
#    reached the reduce-scatter path under the gate policy in force when this file was written (Pk>=4,
#    N_sub>=2, rs_T>=Pk, unfused, single-chunk, shallow K or Pk<=6 with max_chunk>=2) and because they span its
#    distinct code paths: 1-tile and multi-tile chunks, Sm>1, deep K, and -- importantly -- UNEVEN chunk
#    partitions (rs_T % Pk != 0), which is a separate branch in both the writer and compute. Only correctness
#    is asserted, so a policy change that reroutes them cannot fail these tests.
# ---------------------------------------------------------------------------------------------------
_RSCATTER = [
    ("shallowK_sm1", 64, 2048, 1024, (4, 2, 1, 2, 2)),  # rs_T=4,  Pk=4 -> even, 1 tile/chunk
    ("shallowK_sm2", 128, 2048, 1024, (4, 1, 2, 2, 4)),  # rs_T=8,  Pk=4 -> even, 2 tiles/chunk
    ("uneven_9over4", 256, 2048, 1536, (4, 1, 3, 2, 3)),  # rs_T=9,  Pk=4 -> 3,2,2,2  UNEVEN
    ("uneven_6over4", 256, 2048, 512, (4, 1, 3, 2, 2)),  # rs_T=6,  Pk=4 -> 2,2,1,1  UNEVEN
    ("uneven_12over4_sm3", 256, 2048, 2048, (4, 1, 3, 2, 4)),  # rs_T=12, Pk=4 -> even
    ("deepK_pk6_sm2", 256, 6144, 1536, (6, 1, 2, 4, 2)),  # deep K admitted by Pk<=6 & max_chunk>=2
]


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N,cfg", _RSCATTER, ids=[c[0] for c in _RSCATTER])
def test_audit_reduce_scatter(device, label, M, K, N, cfg):
    got, ref = _mm(device, M, K, N, cfg)
    assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N,cfg", _RSCATTER, ids=[c[0] for c in _RSCATTER])
def test_audit_reduce_scatter_auto(device, label, M, K, N, cfg):
    """Same shapes at config=None: what actually ships must be correct, not just the explicit config."""
    got, ref = _mm(device, M, K, N, None)
    assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
def test_audit_reduce_scatter_shape_with_fusion(device):
    """A reduce-scatter-shaped problem WITH a fusion must still be correct, on whatever path is chosen.

    Fusion needs the epilogue applied exactly once at a single reduction root. Whether the reduction strategy
    yields to that, or one day supports it directly, is policy; either way the numbers must come out right, so
    only correctness is asserted here.
    """
    torch.manual_seed(7)
    M, K, N = 256, 2048, 1024
    t0 = torch.randn(1, 1, M, K)
    t1 = torch.randn(1, 1, K, N)
    bias = torch.randn(1, 1, 1, N)
    ref = (t0.bfloat16().float() @ t1.bfloat16().float())[0, 0] + bias.bfloat16().float().reshape(1, -1)
    a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_small_m_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    bt = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    out = ttnn.experimental.small_m_matmul(a0, a1, bias_tensor=bt)
    assert_with_pcc(ref, ttnn.to_torch(ttnn.from_device(out))[0, 0].float(), PCC)


# ---------------------------------------------------------------------------------------------------
# 2. 2D MESH placement: one program in the main suite. These shapes reached the mesh at config=None under the
#    placement policy in force when this file was written (Mt>=8 and either Pk*Ns>=10 with Sm==1, or
#    ring >= 2x in1). Again only correctness is asserted.
# ---------------------------------------------------------------------------------------------------
_MESH = [
    ("mesh_pk12", 256, 6144, 768),
    ("mesh_mt16", 512, 6144, 2304),
    ("mesh_rscatter", 256, 15360, 768),  # mesh AND reduce-scatter together (when written)
    ("mesh_tails", 256, 6080, 4640),  # mesh with non-divisible Kt and Nt
]


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N", _MESH, ids=[c[0] for c in _MESH])
def test_audit_mesh_placement(device, label, M, K, N):
    got, ref = _mm(device, M, K, N, None)
    assert_with_pcc(ref, got, PCC)


# ---------------------------------------------------------------------------------------------------
# 3. Pk / Ns / Sm grid, UNFUSED, on one shape so the only variable is the parallel decomposition.
#    Includes Mt not divisible by Sm (M-split tail) and Ns>2, neither of which the main suite reaches.
# ---------------------------------------------------------------------------------------------------
_GRID = [
    ("pk1", (1, 1, 1, 2, 4)),
    ("pk2_ns2", (2, 2, 1, 2, 2)),
    ("pk4_ns1", (4, 1, 1, 2, 4)),
    ("ns4", (1, 4, 1, 2, 1)),  # Ns>2: the main suite only reaches Ns=2
    ("sm2", (2, 1, 2, 2, 4)),
    ("sm3_mt_nondiv", (2, 1, 3, 2, 4)),  # Mt=8 not divisible by Sm=3 -> M-split tail
    ("pk4_sm2", (4, 1, 2, 2, 4)),
]


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize("label,cfg", _GRID, ids=[c[0] for c in _GRID])
def test_audit_pk_ns_sm_grid(device, label, cfg):
    got, ref = _mm(device, 256, 2048, 1024, cfg)
    assert_with_pcc(ref, got, PCC)


# ---------------------------------------------------------------------------------------------------
# 4. PROGRAM-CACHE DISCRIMINATION. The main suite only replays the SAME program twice, which proves
#    address refresh but cannot catch the cache serving one entry's program for another's inputs -- the
#    exact failure class that invalidated four experiments in this op's history (a stale attribute made two
#    distinct modes alias onto one cached program). These interleave DISTINCT programs in one process.
# ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
def test_audit_cache_distinct_configs_interleaved(device):
    """Two different configs for the SAME shape, interleaved A,B,A,B: each must keep its own program."""
    M, K, N = 256, 2048, 1024
    ca, cb = (4, 1, 1, 2, 4), (2, 1, 2, 2, 4)
    for rep in range(2):
        for cfg in (ca, cb):
            got, ref = _mm(device, M, K, N, cfg, seed=10 + rep)
            assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
def test_audit_cache_distinct_shapes_interleaved(device):
    """Different shapes interleaved: a shape-keyed cache miss must not reuse another shape's program."""
    shapes = [(64, 2048, 1024), (256, 2048, 1024), (32, 6144, 768)]
    for rep in range(2):
        for M, K, N in shapes:
            got, ref = _mm(device, M, K, N, None, seed=20 + rep)
            assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
def test_audit_cache_reduction_strategies_interleaved(device):
    """A reduce-scatter shape and a chain shape interleaved: the two reduction programs must not cross-serve."""
    rs, ch = (256, 2048, 1024), (32, 6144, 4608)  # rscatter, chain (per the ground-truth config log)
    for rep in range(2):
        for M, K, N in (rs, ch):
            got, ref = _mm(device, M, K, N, None, seed=30 + rep)
            assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
def test_audit_cache_explicit_vs_auto_same_shape(device):
    """config=None and an explicit config are different attribute values -> different cache entries.

    Interleaved on one shape, both must stay correct and config=None must replay deterministically. The
    explicit config is deliberately NOT asserted equal to the picker's choice: that is tuning policy, and
    pinning it here would turn every table update into a test failure.
    """
    M, K, N = 256, 6144, 768
    auto1, ref = _mm(device, M, K, N, None, seed=40)
    assert_with_pcc(ref, auto1, PCC)
    expl, ref2 = _mm(device, M, K, N, (12, 1, 1, 2, 1), seed=40)
    assert_with_pcc(ref2, expl, PCC)
    auto2, _ = _mm(device, M, K, N, None, seed=40)
    assert torch.equal(auto1, auto2), "config=None became non-deterministic across cache hits"


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
def test_audit_cache_fused_unfused_chunked_interleaved(device):
    """Unfused / bias-fused / chunked on the same shape, interleaved: three distinct programs, no bleed."""
    M, K, N = 64, 6144, 3072
    for rep in range(2):
        torch.manual_seed(50 + rep)
        t0 = torch.randn(1, 1, M, K)
        t1 = torch.randn(1, 1, K, N)
        bias = torch.randn(1, 1, 1, N)
        base = (t0.bfloat16().float() @ t1.bfloat16().float())[0, 0]
        a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_small_m_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
        a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
        bt = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        plain = ttnn.experimental.small_m_matmul(a0, a1)
        assert_with_pcc(base, ttnn.to_torch(ttnn.from_device(plain))[0, 0].float(), PCC)

        fused = ttnn.experimental.small_m_matmul(a0, a1, bias_tensor=bt)
        assert_with_pcc(
            base + bias.bfloat16().float().reshape(1, -1),
            ttnn.to_torch(ttnn.from_device(fused))[0, 0].float(),
            PCC,
        )

        outs = ttnn.experimental.small_m_matmul_split(a0, a1, 2, -1)
        cat = torch.cat([ttnn.to_torch(ttnn.from_device(o))[0, 0] for o in outs], dim=-1)
        assert_with_pcc(base, cat.float(), PCC)


# ---------------------------------------------------------------------------------------------------
# 7. L1 FEASIBILITY ACCOUNTING. The planner's L1 check must charge for EVERY circular buffer the program
#    factory allocates, not just the always-present ones. It previously omitted the fused-epilogue operands
#    (c_4 bias / c_5 residual / c_6 gate) and the reduce-scatter ring (c_8/c_9 and the c_10 epilogue
#    scratch), so a config could pass feasibility and then be built with buffers nobody had budgeted for.
# ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
def test_audit_l1_accounting_counts_fusion_cbs(device, expect_error):
    """A config that fits UNFUSED but not FUSED must be rejected by the planner, not silently built.

    256x2048x1024 at (Pk1,Ns1,Sm1,kb1,nsb4) needs 1,343,488 B of the 1,474,560 B budget unfused. Adding
    bias (c_4, 4 tiles) + an addcmul residual (c_5, 32 tiles) + a FULL [M,N] gate (c_6, 32 tiles) adds
    139,264 B -> 1,482,752 B, i.e. 8,192 B over. A broadcast [1,N] gate would still fit, so the full-gate
    form is the point of the case. If the accounting regresses to ignoring c_4..c_6 this call succeeds and
    the test fails.
    """
    M, K, N = 256, 2048, 1024
    cfg = ttnn.SmallMMatmulConfig(k_slices=1, n_slices=1, m_slices=1, k_block_tiles=1, n_subblock_tiles=4)
    torch.manual_seed(0)
    t0, t1 = torch.randn(1, 1, M, K), torch.randn(1, 1, K, N)
    a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_small_m_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    dev_t = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    # Control: the same config unfused must still be accepted, so the case is proving that FUSION is what
    # pushed it over -- not that the config was infeasible all along.
    ttnn.experimental.small_m_matmul(a0, a1, config=cfg)

    with expect_error(RuntimeError, "L1 over budget"):
        ttnn.experimental.small_m_matmul(
            a0,
            a1,
            config=cfg,
            bias_tensor=dev_t(torch.randn(1, 1, 1, N)),
            fused_ternary_scalar=1.0,
            fused_ternary_input_a=dev_t(torch.randn(1, 1, M, N)),
            fused_ternary_input_b=dev_t(torch.randn(1, 1, M, N)),
        )


# ---------------------------------------------------------------------------------------------------
# 8. PICKER: large-Mt deep-K rescue. The cost-model fallback searches Sm=1 for its anchor, but on large-Mt
#    deep-K shapes the in0 k-slice-resident CB (M_block * K_slice tiles) exceeds L1 for EVERY Sm=1 candidate,
#    so auto-select used to raise "found no feasible config" even though M-split configs fit. The Sm>1 search
#    must therefore be reachable when no Sm=1 anchor exists -- independently of the narrow-N preference gate,
#    which only decides whether M-split is PREFERABLE, not whether anything fits.
# ---------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize("M,K,N", [(512, 15360, 768), (512, 15360, 1536)], ids=["512x15360x768", "512x15360x1536"])
def test_audit_picker_large_mt_deep_k_rescue(device, M, K, N):
    """config=None must resolve these to a feasible M-split config instead of raising."""
    got, ref = _mm(device, M, K, N)
    assert_with_pcc(ref, got, PCC)


# ---------------------------------------------------------------------------------------------------
# 9. REGRESSION cases for the reduce-scatter c_2 WRAP-ALIGNMENT bug.
#
#    ROOT CAUSE: c_2 (out_cb) was sized 2*out_blk_tiles, but under reduce-scatter both producer and consumer
#    move max_chunk = ceil(out_blk_tiles/Pk) tiles per sub-block. When 2*rs_T is not a multiple of max_chunk a
#    push eventually STRADDLES the circular-buffer wrap, corrupting a partial block: e.g. rs_T=16, Pk=6 ->
#    max_chunk=3 into a 32-tile CB first misaligns on the 11th sub-block. Hence the bug needed Nbpc >= 11 and
#    got worse with Nbpc (244 non-finite at Nbpc=12, 1188 at Nbpc=16), was independent of Kt, and vanished
#    whenever max_chunk divided 2*rs_T (Pk=4, or Pk=5 with rem=1, or nsb=4 lowering Nbpc below the wrap).
#    Fix: size c_2 in max_chunk units under reduce-scatter.
#
#    These cases exist because the suite's only uneven-partition coverage was uneven_9over4 / uneven_6over4,
#    BOTH at Pk=4 -- a value where max_chunk happens to divide the CB. That gap is why it shipped undetected.
#
#    They assert FINITENESS as well as PCC: the corruption was ~0.015% of elements, so finite garbage at that
#    density would leave PCC at ~0.9999 and pass every threshold we use. NaN only tripped PCC because it
#    poisons the whole correlation.
# ---------------------------------------------------------------------------------------------------
_RS_WRAP = [
    # (rs_T, Pk, max_chunk, 2*rs_T, first misaligned push, Nbpc)
    ("wrap_16over6_nbpc12", 256, 6144, 6144, (6, 1, 1, 2, 2)),  # 16,6,3,32 -> push 11, Nbpc 12
    ("wrap_32over6_nbpc12", 512, 3072, 6144, (6, 1, 1, 2, 2)),  # 32,6,6,64 -> push 11, Nbpc 12
    ("wrap_32over5_nbpc12", 512, 2304, 6144, (5, 1, 1, 2, 2)),  # 32,5,7,64 -> push 10, Nbpc 12
    ("wrap_16over6_nbpc16", 256, 1536, 8192, (6, 1, 1, 2, 2)),  # worst observed: 1188 non-finite pre-fix
    ("wrap_14over6_nbpc12", 224, 1536, 6144, (6, 1, 1, 2, 2)),  # 14,6,3,28 -> push 10, Nbpc 12
]


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N,cfg", _RS_WRAP, ids=[c[0] for c in _RS_WRAP])
def test_audit_reduce_scatter_cb_wrap_alignment(device, label, M, K, N, cfg):
    got, ref = _mm(device, M, K, N, cfg)
    assert torch.isfinite(
        got
    ).all(), f"{label}: {int((~torch.isfinite(got)).sum())} non-finite elements of {got.numel()}"
    assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize(
    "label,M,K,N,cfg",
    [
        ("pk4_rem2_clean", 256, 6144, 6144, (4, 1, 3, 2, 2)),  # rs_T=6,  Pk=4 -> rem 2 : CLEAN
        ("pk5_rem1_clean", 256, 6144, 6144, (5, 1, 1, 1, 2)),  # rs_T=16, Pk=5 -> rem 1 : CLEAN
        ("pk5_rem2_nbpc1_clean", 256, 15360, 768, (5, 1, 2, 4, 3)),  # max_chunk 3 divides 24, Nbpc 1: shipped cfg
    ],
    ids=["pk4_rem2_clean", "pk5_rem1_clean", "pk5_rem2_nbpc1_clean"],
)
def test_audit_reduce_scatter_uneven_controls(device, label, M, K, N, cfg):
    """Configs where max_chunk DIVIDES 2*rs_T (or Nbpc is below the wrap), so they were clean even before the
    fix. Kept as controls: they must stay clean, and they guard against a regression that would only show up
    on the aligned cases."""
    got, ref = _mm(device, M, K, N, cfg)
    assert torch.isfinite(got).all(), f"{label}: non-finite output in a control case"
    assert_with_pcc(ref, got, PCC)
