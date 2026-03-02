# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Debug tests for sdpa_standard_v2 with MLA-like dimensions (DHt != vDHt).
Ported from test_scaled_dot_product_attention_sprint.py (pjosipovic/sdpa_wan_integration).
"""

import torch
import ttnn
from loguru import logger
import pytest
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def run_sdpa_noncausal(
    device,
    b,
    nh,
    nkv,
    sq,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    sk=None,
    pcc_threshold=0.9997,
    v_head_dim=None,
    kv_dtype=None,
):
    """
    Run non-causal SDPA and compare against PyTorch reference.
    If v_head_dim is set, uses flash_mla_prefill (MLA mode) with separate V dimension.
    """
    torch.manual_seed(1234)
    if sk is None:
        sk = sq
    if kv_dtype is None:
        kv_dtype = dtype

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    Q = fa_rand(b, nh, sq, d)
    K = fa_rand(b, nkv, sk, d)

    if v_head_dim is not None:
        # MLA mode: V has different head dimension
        V = fa_rand(b, nh, sk, v_head_dim)
    else:
        V = fa_rand(b, nkv, sk, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    if v_head_dim is not None:
        # MLA: flash_mla_prefill with explicit V tensor
        tt_out = ttnn.transformer.flash_mla_prefill(
            tt_Q,
            tt_K,
            tt_V,
            scale=(d**-0.5),
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            is_causal=False,
        )
    else:
        tt_out = ttnn.transformer.scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    tt_back = ttnn.to_torch(tt_out)[:, :, :sq, :]

    # PyTorch reference
    if nkv > 1 and nkv != nh:
        assert nh % nkv == 0
        K_ref = K.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)
        V_ref = V if v_head_dim is not None else K_ref  # V already has nh heads in MLA
        if v_head_dim is None:
            V_ref = V.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)
    else:
        K_ref = K.expand(b, nh, sk, d) if nkv == 1 else K
        V_ref = V

    scale = d**-0.5
    scores = torch.matmul(Q, K_ref.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    gt = torch.matmul(attn, V_ref)

    out_pass, out_pcc = comp_pcc(gt, tt_back, pcc_threshold)
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.info(f"PCC: {out_pcc}, RMSE: {rmse}")

    assert out_pass, f"PCC check failed: {out_pcc} < {pcc_threshold}"


# ============================================================================
# Test 1: Known-good v2 config (d=128, sbh=2, from sbh=2 validation doc)
# Sq_chunk_t=8, Sk_chunk_t=4, DHt=4, vDHt=4, sbh=2
# ============================================================================
@pytest.mark.parametrize(
    "b, nh, sq, d, q_chunk, k_chunk",
    [
        (1, 10, 2368, 128, 256, 128),  # wan_4xGLX_analog, sbh=2
    ],
    ids=["wan_4xGLX_d128_sbh2"],
)
def test_sdpa_v2_known_good(device, b, nh, sq, d, q_chunk, k_chunk):
    """Verify the known-good sbh=2 v2 config still works on this branch."""
    run_sdpa_noncausal(device, b, nh, nh, sq, d, q_chunk, k_chunk, ttnn.bfloat16)


# ============================================================================
# Test 2: MLA config (DHt=18, vDHt=4)
# Q: [b, nh, sq, 576], K: [b, 1, sk, 576], V: [b, nh, sk, 128]
# q_chunk=128 -> Sq_chunk_t=4, k_chunk=512 -> Sk_chunk_t=16
# sbh = determine_largest_subblock_size(4, 16, 8) = {2, 4} -> sbh=2
# sbh*vDHt = 2*4 = 8 <= 8 -> passes v2 gate (with sbh<=2 fix)
# ============================================================================
@pytest.mark.parametrize(
    "b, nh, sq, sk, d, v_head_dim, q_chunk, k_chunk",
    [
        # DHt=4, vDHt=4: same as known-good but via MLA path (nkv=1)
        (1, 4, 1024, 4096, 128, 128, 128, 256),
    ],
    ids=["mla_DHt4_vDHt4"],
)
def test_sdpa_v2_mla(device, b, nh, sq, sk, d, v_head_dim, q_chunk, k_chunk):
    """Test v2 path via MLA (flash_mla_prefill) with nkv=1."""
    run_sdpa_noncausal(
        device,
        b,
        nh,
        1,
        sq,
        d,
        q_chunk,
        k_chunk,
        ttnn.bfloat16,
        sk=sk,
        v_head_dim=v_head_dim,
        pcc_threshold=0.999,
        kv_dtype=ttnn.bfloat8_b,
    )


# ============================================================================
# Test 2b: MLA with nkv=nh (no K broadcasting) to isolate NKH=1 issue
# ============================================================================
@pytest.mark.parametrize(
    "b, nh, nkv, sq, sk, d, v_head_dim, q_chunk, k_chunk",
    [
        (1, 4, 4, 1024, 4096, 128, 128, 128, 256),
        (1, 4, 1, 1024, 4096, 128, 128, 128, 256),
    ],
    ids=["mla_nkv4", "mla_nkv1"],
)
def test_sdpa_v2_mla_nkv(device, b, nh, nkv, sq, sk, d, v_head_dim, q_chunk, k_chunk):
    """Test MLA path with nkv=nh (no K broadcasting) vs nkv=1."""
    run_sdpa_noncausal(
        device,
        b,
        nh,
        nkv,
        sq,
        d,
        q_chunk,
        k_chunk,
        ttnn.bfloat16,
        sk=sk,
        v_head_dim=v_head_dim,
        pcc_threshold=0.999,
        kv_dtype=ttnn.bfloat8_b,
    )


# ============================================================================
# Test 4: MLA with all bfloat16 (isolate data format vs MLA path)
# ============================================================================
def test_sdpa_v2_mla_bf16(device):
    """MLA path with all bfloat16 — isolate bfloat8_b vs use_mla flag."""
    run_sdpa_noncausal(
        device,
        1,
        4,
        4,
        1024,
        128,
        128,
        256,
        ttnn.bfloat16,
        sk=4096,
        v_head_dim=128,
        pcc_threshold=0.999,
    )


# ============================================================================
# Test 5: Non-MLA with bfloat8_b K/V — isolate compute vs reader
# ============================================================================
def test_sdpa_v2_nonmla_bf8b(device):
    """Non-MLA (scaled_dot_product_attention) with bfloat8_b K/V."""
    run_sdpa_noncausal(
        device,
        1,
        4,
        4,
        1024,
        128,
        128,
        256,
        ttnn.bfloat16,
        sk=4096,
        pcc_threshold=0.999,
        kv_dtype=ttnn.bfloat8_b,
    )


# ============================================================================
# Test 3: Non-MLA GQA (nkv=1) via scaled_dot_product_attention
# Isolates whether nkv=1 K-broadcasting is the issue
# ============================================================================
@pytest.mark.parametrize(
    "b, nh, nkv, sq, sk, d, q_chunk, k_chunk",
    [
        (1, 4, 1, 1024, 4096, 128, 128, 256),
        (1, 4, 4, 1024, 4096, 128, 128, 256),
    ],
    ids=["gqa_nkv1", "gqa_nkv4"],
)
def test_sdpa_v2_gqa(device, b, nh, nkv, sq, sk, d, q_chunk, k_chunk):
    """Test v2 with GQA (nkv=1) via non-MLA path to isolate K broadcasting."""
    run_sdpa_noncausal(
        device,
        b,
        nh,
        nkv,
        sq,
        d,
        q_chunk,
        k_chunk,
        ttnn.bfloat16,
        sk=sk,
        pcc_threshold=0.999,
    )


# ============================================================================
# Test 6: MLA with vDHt=16 (v_head_dim=512, requires DST batching)
# Q: [b, nh, sq, 576], K: [b, 1, sk, 576], V: [b, nh, sk, 512]
# q_chunk=128 -> Sq_chunk_t=4, k_chunk=256 -> Sk_chunk_t=8
# DHt=18, vDHt=16
# sbh = determine_largest_subblock_size(4, 8, 8) = {2, 4} -> sbh=2
# ============================================================================
@pytest.mark.parametrize(
    "b, nh, sq, sk, d, v_head_dim, q_chunk, k_chunk",
    [
        (1, 4, 1024, 4096, 576, 512, 128, 256),
    ],
    ids=["mla_DHt18_vDHt16"],
)
def test_sdpa_v2_mla_vDHt16(device, b, nh, sq, sk, d, v_head_dim, q_chunk, k_chunk):
    """Test v2 path with vDHt=16, requiring DST batching in normalize and mul_block."""
    run_sdpa_noncausal(
        device,
        b,
        nh,
        1,
        sq,
        d,
        q_chunk,
        k_chunk,
        ttnn.bfloat16,
        sk=sk,
        v_head_dim=v_head_dim,
        pcc_threshold=0.999,
        kv_dtype=ttnn.bfloat8_b,
    )
