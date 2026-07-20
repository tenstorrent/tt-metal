# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Correctness tests for ttnn.experimental.regime_a_matmul (Blackhole DRAM-BW-optimal Regime-A matmul).
# Random BF16 inputs vs a Torch reference. in1 is DRAM width-sharded via the op's canonical helper.
# Cases follow the plan's bring-up order: Pk=1 (no reduction) -> Pk>1 (reduction) -> Ns>1 -> Sm>1 ->
# golden parity configs -> non-divisible.

import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc


def _run_regime_a(device, M, K, N, Ns, Pk, Sm, kb, nsb, pcc=0.999):
    torch.manual_seed(0)
    torch_in0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    torch_in1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    torch_ref = (torch_in0.float() @ torch_in1.float())[0, 0]

    in0 = ttnn.from_torch(torch_in0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    weight_mem_cfg = ttnn.create_regime_a_weight_memory_config(list(torch_in1.shape), ttnn.bfloat16, device)
    in1 = ttnn.from_torch(
        torch_in1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=weight_mem_cfg
    )

    config = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    out = ttnn.experimental.regime_a_matmul(in0, in1, config=config)
    out_torch = ttnn.to_torch(ttnn.from_device(out))[0, 0]

    assert out_torch.shape == torch_ref.shape, f"shape {out_torch.shape} != {torch_ref.shape}"
    assert_with_pcc(torch_ref, out_torch.float(), pcc)


# (label, M, K, N, Ns, Pk, Sm, kb, nsb)
CASES = [
    ("pk1_noreduce", 32, 6144, 3072, 1, 1, 1, 4, 6),  # 8 cores, no reduction
    ("pk3_reduce", 32, 6144, 3072, 1, 3, 1, 4, 6),  # split-K reduction chain
    ("ns2", 32, 2048, 2048, 2, 2, 1, 4, 4),  # N-slice > 1
    ("sm2", 128, 6144, 4608, 1, 6, 2, 2, 1),  # M-split > 1 (in1 forward)
    ("golden_mt1", 32, 6144, 4608, 1, 12, 1, 2, 1),
    ("golden_mt2", 64, 6144, 4608, 1, 6, 1, 4, 2),
    ("golden_mt4", 128, 6144, 4608, 1, 12, 1, 2, 1),
    ("golden_mt8", 256, 6144, 4608, 1, 12, 1, 2, 1),
]


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N,Ns,Pk,Sm,kb,nsb", CASES, ids=[c[0] for c in CASES])
def test_regime_a_matmul_correctness(device, label, M, K, N, Ns, Pk, Sm, kb, nsb):
    _run_regime_a(device, M, K, N, Ns, Pk, Sm, kb, nsb)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize(
    "M,K,N",
    [(32, 6144, 4608), (64, 6144, 4608), (128, 6144, 4608), (32, 6144, 6144)],
    ids=["mt1", "mt2", "mt4", "mt1-big"],
)
def test_regime_a_matmul_auto_config(device, M, K, N):
    # config=None -> the op auto-selects (Pk,Ns,Sm,kb,nsb) via the ported FLUX/LTX picker.
    torch.manual_seed(0)
    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = (t0.float() @ t1.float())[0, 0]
    a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    out = ttnn.experimental.regime_a_matmul(a0, a1)  # config omitted -> auto
    got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
    assert_with_pcc(ref, got.float(), 0.999)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_matmul_cache_replay(device):
    # Run the same config twice on FRESH tensors (different buffer addresses). The second call hits the
    # program cache and must pick up the new addresses via override_runtime_arguments.
    M, K, N, cfg = 64, 6144, 4608, (1, 6, 1, 4, 2)
    Ns, Pk, Sm, kb, nsb = cfg
    for trial in range(2):
        torch.manual_seed(trial)  # different contents + fresh allocations each trial
        t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        ref = (t0.float() @ t1.float())[0, 0]
        a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
        a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
        conf = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
        out = ttnn.experimental.regime_a_matmul(a0, a1, config=conf)
        got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
        assert_with_pcc(ref, got.float(), 0.999)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_matmul_column_dependent_layout(device):
    # Structured in1 (each N-column has a distinct signature, constant over K) pins down N-column /
    # bank ownership: a bank-permutation or wrong N-offset scrambles columns and drops PCC even where
    # a random weight might average out. in0=ones so out[m,n] = sum_k in1[k,n].
    M, K, N, cfg = 32, 6144, 4608, (1, 12, 1, 2, 1)
    Ns, Pk, Sm, kb, nsb = cfg
    torch.manual_seed(0)
    t0 = torch.ones(1, 1, M, K, dtype=torch.bfloat16)
    # small per-column signature (keep bf16-representable): col n -> (n % 251) * 0.01
    col = ((torch.arange(N) % 251).float() * 0.01).to(torch.bfloat16)
    t1 = col.view(1, 1, 1, N).expand(1, 1, K, N).contiguous()
    ref = (t0.float() @ t1.float())[0, 0]
    a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    conf = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    out = ttnn.experimental.regime_a_matmul(a0, a1, config=conf)
    got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
    assert_with_pcc(ref, got.float(), 0.999)


def _run_regime_a_auto(device, M, K, N, Ns=None, Pk=None, Sm=None, kb=None, nsb=None, pcc=0.999):
    # Balanced-tail path: LOGICAL M×K and K×N tensors (no manual padding). config=None auto-selects
    # unless a manual config is given. Output is logical M×N.
    torch.manual_seed(0)
    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = (t0.float() @ t1.float())[0, 0]
    in0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    in1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    cfg = None
    if Pk is not None:
        cfg = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    out = ttnn.experimental.regime_a_matmul(in0, in1, config=cfg)
    got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
    assert tuple(got.shape) == tuple(ref.shape), f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
    # No NaN/Inf: guards the K-tail path (pad tiles must be exactly 0.0, never 0*garbage).
    assert torch.isfinite(got.float()).all(), "regime_a_matmul produced non-finite output (K-tail poisoning?)"
    assert_with_pcc(ref, got.float(), pcc)


# Balanced-tail corner cases (LOGICAL inputs, config=None auto-select). Covers the BT-4 matrix:
# Kt%Pk!=0 / valid_k%kb!=0 (6080=190t), Nt%8!=0 (4640=145t), Mt%Sm!=0 (M=96 Mt=3),
# non-tile-aligned element dims (48/6100/4600), and combinations.
NONDIV = [
    ("kt_not_div", 32, 6080, 4608),  # Kt=190 not divisible by Pk (balanced K tails, valid_k%kb!=0)
    ("nt_not_8", 32, 6144, 4640),  # Nt=145 not multiple of 8 (bank-7 N tail)
    ("kt_and_nt", 32, 6080, 4640),  # both
    ("mt_not_div_sm", 96, 6144, 4608),  # Mt=3 (M-split tail when Sm>1 is chosen)
    ("subtile_dims", 48, 6100, 4600),  # none of M/K/N tile-aligned (sub-32 tail elements)
    ("mt2_nondiv", 80, 6080, 4640),  # M=80 (Mt=3, sub-tile), Kt=190, Nt=145
]


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N", NONDIV, ids=[c[0] for c in NONDIV])
def test_regime_a_matmul_balanced_tails(device, label, M, K, N):
    _run_regime_a_auto(device, M, K, N)


# =====================================================================================================
# Fused epilogue: bias / unary activation / addcmul, output-split (chunks), and their compositions.
# Split-K (Pk>1) must apply the fusion EXACTLY ONCE at the reduction root (never per-partial); the
# config=None cases below drive the auto-picker (which selects Pk>1 for these shapes) with each fusion
# to guard against a silent split-K + fusion correctness failure.
# =====================================================================================================


def _fused_reference(a, b, bias=None, act=None, scalar=None, resid=None, gate=None):
    ref = (a.float() @ b.float())[0, 0]  # [M, N]
    if bias is not None:
        ref = ref + bias.float().reshape(1, -1)
    if act == "relu":
        ref = torch.relu(ref)
    elif act == "gelu":
        ref = torch.nn.functional.gelu(ref)
    if scalar is not None:
        g = gate.float()[0, 0]  # [1,N] or [M,N]
        ref = resid.float()[0, 0] + scalar * ref * g
    return ref


def _run_fused(
    device,
    M,
    K,
    N,
    Ns,
    Pk,
    Sm,
    kb,
    nsb,
    bias=False,
    act=None,
    scalar=None,
    gate_full=False,
    gate_fp32=False,
    chunks=1,
    pcc=0.999,
):
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    in0 = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, device)
    in1 = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    cfg = None
    if Pk is not None:
        cfg = ttnn.RegimeAMatmulConfig(k_slices=Pk, n_slices=Ns, m_slices=Sm, k_block_tiles=kb, n_subblock_tiles=nsb)
    kw = dict(config=cfg)
    bias_t = torch.randn(1, 1, 1, N, dtype=torch.bfloat16) if bias else None
    if bias:
        kw["bias_tensor"] = ttnn.from_torch(bias_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    if act is not None:
        kw["fused_activation"] = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU if act == "relu" else ttnn.UnaryOpType.GELU)
    resid_t = gate_t = None
    if scalar is not None:
        resid_t = torch.randn(1, 1, M, N, dtype=torch.bfloat16)
        gm = M if gate_full else 1
        gate_t = torch.randn(1, 1, gm, N, dtype=(torch.float32 if gate_fp32 else torch.bfloat16))
        kw["fused_ternary_scalar"] = scalar
        kw["fused_ternary_input_a"] = ttnn.from_torch(
            resid_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
        )
        kw["fused_ternary_input_b"] = ttnn.from_torch(
            gate_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=(ttnn.float32 if gate_fp32 else ttnn.bfloat16)
        )
    ref = _fused_reference(a, b, bias_t, act, scalar, resid_t, gate_t)
    if chunks > 1:
        outs = ttnn.experimental.regime_a_matmul_split(in0, in1, chunks, -1, **kw)
        assert len(outs) == chunks
        got = torch.cat([ttnn.to_torch(ttnn.from_device(o))[0, 0] for o in outs], dim=-1)
    else:
        out = ttnn.experimental.regime_a_matmul(in0, in1, **kw)
        got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
    assert tuple(got.shape) == tuple(ref.shape), f"{tuple(got.shape)} != {tuple(ref.shape)}"
    assert torch.isfinite(got.float()).all(), "non-finite fused output"
    assert_with_pcc(ref, got.float(), pcc)


# (label, Pk) on 32x6144x3072 (Mt1,Kt192,Nt96, W=1). Covers Pk=1 (no reduction) and Pk=3 (split-K root).
_FUSE_PK = [("pk1", 1), ("pk3", 3)]


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("pk_label,Pk", _FUSE_PK, ids=[c[0] for c in _FUSE_PK])
def test_regime_a_fused_bias(device, pk_label, Pk):
    _run_fused(device, 32, 6144, 3072, 1, Pk, 1, 4, 6, bias=True)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("pk_label,Pk", _FUSE_PK, ids=[c[0] for c in _FUSE_PK])
@pytest.mark.parametrize("act", ["relu", "gelu"])
def test_regime_a_fused_activation(device, pk_label, Pk, act):
    _run_fused(device, 32, 6144, 3072, 1, Pk, 1, 4, 6, act=act)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("pk_label,Pk", _FUSE_PK, ids=[c[0] for c in _FUSE_PK])
def test_regime_a_fused_bias_activation(device, pk_label, Pk):
    _run_fused(device, 32, 6144, 3072, 1, Pk, 1, 4, 6, bias=True, act="relu")


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("pk_label,Pk", _FUSE_PK, ids=[c[0] for c in _FUSE_PK])
@pytest.mark.parametrize("scalar", [1.0, 0.5, 2.5])
def test_regime_a_fused_addcmul_broadcast(device, pk_label, Pk, scalar):
    _run_fused(device, 32, 6144, 3072, 1, Pk, 1, 4, 6, scalar=scalar)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("pk_label,Pk", _FUSE_PK, ids=[c[0] for c in _FUSE_PK])
def test_regime_a_fused_addcmul_bias(device, pk_label, Pk):
    _run_fused(device, 32, 6144, 3072, 1, Pk, 1, 4, 6, bias=True, scalar=1.0)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("pk_label,Pk", _FUSE_PK, ids=[c[0] for c in _FUSE_PK])
@pytest.mark.parametrize("M", [32, 64])
def test_regime_a_fused_addcmul_full_gate(device, pk_label, Pk, M):
    # Regression: M=32 (Mt=1) makes the full [M,N] gate occupy exactly ONE tile row, so it is
    # indistinguishable from a [1,N] broadcast gate by padded shape. The broadcast-vs-full decision must
    # key off LOGICAL M (==1 => broadcast), NOT padded rows, or a genuine per-row gate is silently
    # broadcast (row 0 across all 32 rows). M=64 (Mt=2) covers the multi-tile-row full-gate case.
    _run_fused(device, M, 6144, 3072, 1, Pk, 1, 4, 6, scalar=0.7, gate_full=True)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("pk_label,Pk", _FUSE_PK, ids=[c[0] for c in _FUSE_PK])
def test_regime_a_fused_addcmul_fp32_gate(device, pk_label, Pk):
    _run_fused(device, 32, 6144, 3072, 1, Pk, 1, 4, 6, scalar=1.0, gate_fp32=True)


# Ns>1, Sm>1, W>1 planner modes each with a fusion (Sm2 uses M=128; W>1 via K=15360 Pk6 -> W5).
@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_fused_ns_gt1_bias(device):
    _run_fused(device, 32, 2048, 2048, 2, 2, 1, 4, 4, bias=True)  # Ns=2


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_fused_sm_gt1_addcmul(device):
    _run_fused(device, 128, 6144, 4608, 1, 6, 2, 2, 1, scalar=1.0)  # Sm=2, split-K root fusion under M-split


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_fused_deepK_W_gt1_bias_act(device):
    _run_fused(device, 32, 15360, 3072, 1, 6, 1, 2, 3, bias=True, act="relu")  # W>1 (deep-K)


# Chunking (output-split) composed with fusions.
@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("chunks", [2, 3])
def test_regime_a_split_no_fusion(device, chunks):
    _run_fused(device, 32, 6144, 3072, 1, 3, 1, 4, 6, chunks=chunks)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_split_bias(device):
    _run_fused(device, 32, 6144, 3072, 1, 3, 1, 4, 6, bias=True, chunks=2)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_split_bias_activation(device):
    _run_fused(device, 32, 6144, 3072, 1, 3, 1, 4, 6, bias=True, act="relu", chunks=3)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_split_addcmul(device):
    _run_fused(device, 32, 6144, 3072, 1, 3, 1, 4, 6, scalar=1.0, chunks=2)


# Balanced tails (non-divisible logical dims) with a fusion, config=None auto-select.
@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
@pytest.mark.parametrize("label,M,K,N", NONDIV, ids=[c[0] for c in NONDIV])
def test_regime_a_fused_bias_tails(device, label, M, K, N):
    _run_fused(device, M, K, N, None, None, None, None, None, bias=True)


# ---- config=None (auto-picker) + each fusion. The picker selects Pk>1 for 32x6144x3072 (Pk3), so this
# is the code-review regression: split-K auto-selection MUST be fusion-aware (fuse once at the root). ----
@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_autoconfig_pk_gt1_bias(device):
    _run_fused(device, 32, 6144, 3072, None, None, None, None, None, bias=True)


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_autoconfig_pk_gt1_activation(device):
    _run_fused(device, 32, 6144, 3072, None, None, None, None, None, act="relu")


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_autoconfig_pk_gt1_addcmul(device):
    _run_fused(device, 32, 6144, 3072, None, None, None, None, None, scalar=1.0)


# ---- Program-cache replay with fused inputs on FRESH buffers (must pick up new addresses). ----
@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_fused_cache_replay(device):
    M, K, N = 32, 6144, 3072
    for trial in range(2):
        torch.manual_seed(100 + trial)
        a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
        bias_t = torch.randn(1, 1, 1, N, dtype=torch.bfloat16)
        resid_t = torch.randn(1, 1, M, N, dtype=torch.bfloat16)
        gate_t = torch.randn(1, 1, 1, N, dtype=torch.bfloat16)
        in0 = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        wcfg = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, device)
        in1 = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
        bt = ttnn.from_torch(bias_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        rt = ttnn.from_torch(resid_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        gt = ttnn.from_torch(gate_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        cfg = ttnn.RegimeAMatmulConfig(k_slices=3, n_slices=1, m_slices=1, k_block_tiles=4, n_subblock_tiles=6)
        out = ttnn.experimental.regime_a_matmul(
            in0,
            in1,
            config=cfg,
            bias_tensor=bt,
            fused_ternary_scalar=1.0,
            fused_ternary_input_a=rt,
            fused_ternary_input_b=gt,
        )
        got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
        ref = (
            resid_t.float()[0, 0]
            + 1.0 * ((a.float() @ b.float())[0, 0] + bias_t.float().reshape(1, -1)) * gate_t.float()[0, 0]
        )
        assert_with_pcc(ref, got.float(), 0.999)


# ---- Validation: invalid shapes / dim / chunks / dtypes / activation+addcmul must fail clearly. ----
def _mk(device, shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=dtype,
    )


@pytest.mark.skipif(not is_blackhole(), reason="Regime-A matmul is Blackhole-only")
def test_regime_a_fused_validation(device):
    M, K, N = 32, 6144, 3072
    a = _mk(device, (1, 1, M, K))
    b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    wcfg = ttnn.create_regime_a_weight_memory_config(list(b.shape), ttnn.bfloat16, device)
    in1 = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)

    # activation + addcmul together -> reject
    with pytest.raises(RuntimeError):
        ttnn.experimental.regime_a_matmul(
            a,
            in1,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            fused_ternary_scalar=1.0,
            fused_ternary_input_a=_mk(device, (1, 1, M, N)),
            fused_ternary_input_b=_mk(device, (1, 1, 1, N)),
        )
    # bias wrong N -> reject
    with pytest.raises(RuntimeError):
        ttnn.experimental.regime_a_matmul(a, in1, bias_tensor=_mk(device, (1, 1, 1, N + 32)))
    # addcmul scalar without residual/gate -> reject
    with pytest.raises(RuntimeError):
        ttnn.experimental.regime_a_matmul(a, in1, fused_ternary_scalar=1.0)
    # residual wrong shape -> reject
    with pytest.raises(RuntimeError):
        ttnn.experimental.regime_a_matmul(
            a,
            in1,
            fused_ternary_scalar=1.0,
            fused_ternary_input_a=_mk(device, (1, 1, M + 32, N)),
            fused_ternary_input_b=_mk(device, (1, 1, 1, N)),
        )
    # split: dim != -1 -> reject
    with pytest.raises(RuntimeError):
        ttnn.experimental.regime_a_matmul_split(a, in1, 2, 0)
    # split: N not divisible by chunks -> reject (3072/5 not integer)
    with pytest.raises(RuntimeError):
        ttnn.experimental.regime_a_matmul_split(a, in1, 5, -1)
