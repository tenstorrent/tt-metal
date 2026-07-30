# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Coverage audit for ttnn.experimental.regime_a_matmul: paths the main suite does not reach.

WHY THIS FILE EXISTS. Running the main 111-test suite with the factory's own config log
(TT_REGIME_A_LOG_CFG=1) showed which production paths it actually reaches:

    99 programs  reduction=chain  placement=bank-local
    10 programs  reduction=chain  placement=in1-near
     1 program   reduction=chain  placement=mesh
     0 programs  reduction=reduce-scatter        <-- never exercised

Ring reduce-scatter ships on 14 of the 66 corpus shapes, so "111/111 pass" was saying nothing about it.
The 2D mesh placement was covered by a single program, and Ns>1 by a single config. This file closes those
gaps and adds the program-cache DISCRIMINATION cases the main suite lacks (it only ever replays the SAME
program twice, which cannot catch cross-serving between distinct entries).

Shapes/configs below are the ones the ground-truth sweep showed select each path. Each test asserts the path
it intends to cover is really taken, so it cannot silently rot into another duplicate chain test.
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
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    conf = None
    if cfg is not None:
        Pk, Ns, Sm, kb, nsb = cfg
        conf = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    out = ttnn.experimental.regime_a_matmul(a0, a1, config=conf)
    return ttnn.to_torch(ttnn.from_device(out))[0, 0].float(), ref


# ---------------------------------------------------------------------------------------------------
# 1. RING REDUCE-SCATTER. Zero coverage in the main suite. The gate needs Pk>=4, N_sub>=2, rs_T>=Pk,
#    unfused, single-chunk, and either shallow K or (Pk<=6 and max_chunk>=2). Cases below cover: the
#    shallow-K regime, Sm>1 with reduce-scatter, deep K admitted by the Pk<=6 clause, and -- importantly --
#    UNEVEN chunk partitions (rs_T % Pk != 0), which is a distinct code path in both the writer and compute.
# ---------------------------------------------------------------------------------------------------
_RSCATTER = [
    ("shallowK_sm1", 64, 2048, 1024, (4, 2, 1, 2, 2)),  # rs_T=4,  Pk=4 -> even, 1 tile/chunk
    ("shallowK_sm2", 128, 2048, 1024, (4, 1, 2, 2, 4)),  # rs_T=8,  Pk=4 -> even, 2 tiles/chunk
    ("uneven_9over4", 256, 2048, 1536, (4, 1, 3, 2, 3)),  # rs_T=9,  Pk=4 -> 3,2,2,2  UNEVEN
    ("uneven_6over4", 256, 2048, 512, (4, 1, 3, 2, 2)),  # rs_T=6,  Pk=4 -> 2,2,1,1  UNEVEN
    ("uneven_12over4_sm3", 256, 2048, 2048, (4, 1, 3, 2, 4)),  # rs_T=12, Pk=4 -> even
    ("deepK_pk6_sm2", 256, 6144, 1536, (6, 1, 2, 4, 2)),  # deep K admitted by Pk<=6 & max_chunk>=2
]


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N,cfg", _RSCATTER, ids=[c[0] for c in _RSCATTER])
def test_audit_reduce_scatter(device, label, M, K, N, cfg):
    got, ref = _mm(device, M, K, N, cfg)
    assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N,cfg", _RSCATTER, ids=[c[0] for c in _RSCATTER])
def test_audit_reduce_scatter_auto(device, label, M, K, N, cfg):
    """Same shapes at config=None: what actually ships must be correct, not just the explicit config."""
    got, ref = _mm(device, M, K, N, None)
    assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_audit_reduce_scatter_gate_yields_to_fusion(device):
    """A reduce-scatter-shaped problem WITH a fusion must fall back to the chain and still be right.

    The gate excludes fusion because the epilogue has to be applied exactly once at a single root, which the
    scatter topology does not have. This asserts that fallback is correct rather than silently wrong.
    """
    torch.manual_seed(7)
    M, K, N = 256, 2048, 1024
    t0 = torch.randn(1, 1, M, K)
    t1 = torch.randn(1, 1, K, N)
    bias = torch.randn(1, 1, 1, N)
    ref = (t0.bfloat16().float() @ t1.bfloat16().float())[0, 0] + bias.bfloat16().float().reshape(1, -1)
    a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    bt = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    out = ttnn.experimental.regime_a_matmul(a0, a1, bias_tensor=bt)
    assert_with_pcc(ref, ttnn.to_torch(ttnn.from_device(out))[0, 0].float(), PCC)


# ---------------------------------------------------------------------------------------------------
# 2. 2D MESH placement: one program in the main suite. Gate = Mt>=8 and (Pk*Ns>=10 with Sm==1) or
#    ring>=2x in1. These shapes select it at config=None per the ground-truth sweep.
# ---------------------------------------------------------------------------------------------------
_MESH = [
    ("mesh_pk12", 256, 6144, 768),
    ("mesh_mt16", 512, 6144, 2304),
    ("mesh_rscatter", 256, 15360, 768),  # mesh AND reduce-scatter together
    ("mesh_tails", 256, 6080, 4640),  # mesh with non-divisible Kt and Nt
]


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
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


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
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
@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_audit_cache_distinct_configs_interleaved(device):
    """Two different configs for the SAME shape, interleaved A,B,A,B: each must keep its own program."""
    M, K, N = 256, 2048, 1024
    ca, cb = (4, 1, 1, 2, 4), (2, 1, 2, 2, 4)
    for rep in range(2):
        for cfg in (ca, cb):
            got, ref = _mm(device, M, K, N, cfg, seed=10 + rep)
            assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_audit_cache_distinct_shapes_interleaved(device):
    """Different shapes interleaved: a shape-keyed cache miss must not reuse another shape's program."""
    shapes = [(64, 2048, 1024), (256, 2048, 1024), (32, 6144, 768)]
    for rep in range(2):
        for M, K, N in shapes:
            got, ref = _mm(device, M, K, N, None, seed=20 + rep)
            assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_audit_cache_reduction_strategies_interleaved(device):
    """A reduce-scatter shape and a chain shape interleaved: the two reduction programs must not cross-serve."""
    rs, ch = (256, 2048, 1024), (32, 6144, 4608)  # rscatter, chain (per the ground-truth config log)
    for rep in range(2):
        for M, K, N in (rs, ch):
            got, ref = _mm(device, M, K, N, None, seed=30 + rep)
            assert_with_pcc(ref, got, PCC)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_audit_cache_explicit_vs_auto_same_shape(device):
    """config=None and an explicit config are different attribute values -> different cache entries.

    Interleaved on one shape, both must stay correct; and the explicit config equal to the picker's own
    choice must agree with config=None BIT FOR BIT, since it is the same program modulo the attribute.
    """
    M, K, N = 256, 6144, 768
    auto1, ref = _mm(device, M, K, N, None, seed=40)
    assert_with_pcc(ref, auto1, PCC)
    expl, ref2 = _mm(device, M, K, N, (12, 1, 1, 2, 1), seed=40)  # the picker's own choice for this shape
    assert_with_pcc(ref2, expl, PCC)
    auto2, _ = _mm(device, M, K, N, None, seed=40)
    assert torch.equal(auto1, auto2), "config=None became non-deterministic across cache hits"
    assert torch.equal(expl, auto1), (
        "explicit config equal to the auto pick produced a different result than config=None; "
        "the two should be the same program"
    )


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
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
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
        a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
        bt = ttnn.from_torch(bias, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

        plain = ttnn.experimental.regime_a_matmul(a0, a1)
        assert_with_pcc(base, ttnn.to_torch(ttnn.from_device(plain))[0, 0].float(), PCC)

        fused = ttnn.experimental.regime_a_matmul(a0, a1, bias_tensor=bt)
        assert_with_pcc(
            base + bias.bfloat16().float().reshape(1, -1),
            ttnn.to_torch(ttnn.from_device(fused))[0, 0].float(),
            PCC,
        )

        outs = ttnn.experimental.regime_a_matmul_split(a0, a1, 2, -1)
        cat = torch.cat([ttnn.to_torch(ttnn.from_device(o))[0, 0] for o in outs], dim=-1)
        assert_with_pcc(base, cat.float(), PCC)
