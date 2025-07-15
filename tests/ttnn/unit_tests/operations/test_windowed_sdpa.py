# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_wormhole_b0, skip_for_blackhole


def fa_rand(*shape):
    """Create random tensor following FlashAttention test pattern"""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def create_windowed_attention_mask(cu_window_seqlens, seq_len, dtype=torch.bfloat16):
    """Create windowed attention mask from cumulative window sequence lengths.

    Args:
        cu_window_seqlens: List of cumulative window sequence lengths [0, win1_end, win2_end, ...]
        seq_len: Total sequence length (padded)
        dtype: Data type for the mask

    Returns:
        Attention mask tensor of shape [1, 1, seq_len, seq_len]
    """
    attention_mask = torch.full([1, 1, seq_len, seq_len], -1e9, dtype=dtype)

    # Create windows where tokens can only attend within their window
    for i in range(1, len(cu_window_seqlens)):
        start = cu_window_seqlens[i - 1]
        end = cu_window_seqlens[i]
        attention_mask[..., start:end, start:end] = 0

    return attention_mask


def run_test_windowed_sdpa(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    cu_window_seqlens,
    q_chunk_size,
    k_chunk_size,
    dtype,
    use_high_precision_compute=False,
    rmse_threshold=None,
):
    """Run windowed SDPA test comparing against regular SDPA with attention mask."""
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    if use_high_precision_compute:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    # Create input tensors
    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    # Convert to TT tensors
    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Convert cu_window_seqlens to tensor
    tt_cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)

    # Run windowed SDPA
    tt_windowed_out = ttnn.transformer.windowed_scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        cu_window_seqlens=tt_cu_window_seqlens,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_windowed_out = ttnn.to_torch(tt_windowed_out)

    # Create attention mask for regular SDPA
    attention_mask = create_windowed_attention_mask(cu_window_seqlens, s)
    tt_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Run regular SDPA with attention mask
    tt_regular_out = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=False,
        attn_mask=tt_mask,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_regular_out = ttnn.to_torch(tt_regular_out)

    # Slice out any tile-padding
    tt_windowed_out = tt_windowed_out[:, :, :s, :]
    tt_regular_out = tt_regular_out[:, :, :s, :]

    # Compare outputs
    out_pass, out_pcc = comp_pcc(tt_regular_out, tt_windowed_out, 0.999)
    logger.debug(f"windowed vs regular SDPA: PCC = {out_pcc}")
    rmse = torch.sqrt(((tt_regular_out - tt_windowed_out) ** 2).mean()).item()
    logger.debug(f"RMSE: {rmse}")

    # Also check against PyTorch reference
    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)

    # Apply attention mask for PyTorch reference
    mask_expanded = attention_mask.expand(b, nh, s, s)
    torch_out = torch.nn.functional.scaled_dot_product_attention(
        Q, K_repeated, V_repeated, is_causal=False, attn_mask=mask_expanded
    )

    ref_pass, ref_pcc = comp_pcc(torch_out, tt_windowed_out[:, :, :s, :], 0.994)
    logger.debug(f"windowed SDPA vs PyTorch reference: PCC = {ref_pcc}")

    if rmse_threshold is not None:
        assert rmse < rmse_threshold
    else:
        assert out_pass, f"PCC {out_pcc} failed between windowed and regular SDPA"
        assert ref_pass, f"PCC {ref_pcc} failed between windowed SDPA and PyTorch reference"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize("b", [1, 2], ids=["b1", "b2"])
@pytest.mark.parametrize("nh", [8, 16], ids=["nh8", "nh16"])
@pytest.mark.parametrize("nkv", [1], ids=["nkv1"])
@pytest.mark.parametrize(
    "s, cu_window_seqlens",
    [
        (512, [0, 128, 256, 384, 512]),  # 4 windows of 128 each
        (1024, [0, 256, 512, 768, 1024]),  # 4 windows of 256 each
        (2048, [0, 512, 1024, 1536, 2048]),  # 4 windows of 512 each
        (2048, [0, 1024, 2048]),  # 2 windows of 1024 each
        (2048, [0, 256, 768, 1280, 2048]),  # Variable window sizes
    ],
    ids=["s512_4win", "s1024_4win", "s2048_4win", "s2048_2win", "s2048_varwin"],
)
@pytest.mark.parametrize("d", [128], ids=["d128"])
def test_windowed_sdpa(device, b, nh, nkv, s, d, cu_window_seqlens, q_chunk_size, k_chunk_size, dtype):
    """Test windowed SDPA against regular SDPA with attention mask."""
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh % nkv != 0:
        pytest.skip("nkv must divide nh")

    ttnn.device.DisablePersistentKernelCache()
    run_test_windowed_sdpa(
        device, b, nh, nkv, s, d, cu_window_seqlens, q_chunk_size, k_chunk_size, dtype, rmse_threshold=0.01
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, cu_window_seqlens",
    [
        [1, 8, 1, 2048, 128, [0, 512, 1024, 1536, 2048]],  # Llama2-style config
        [1, 16, 1, 2048, 64, [0, 256, 512, 1024, 1536, 2048]],  # Falcon-style config
        [8, 8, 1, 2048, 128, [0, 1024, 2048]],  # Large batch
    ],
    ids=["llama2", "falcon", "large_batch"],
)
def test_windowed_sdpa_models(device, b, nh, nkv, s, d, cu_window_seqlens, q_chunk_size, k_chunk_size, dtype):
    """Test windowed SDPA with model-specific configurations."""
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")

    ttnn.device.DisablePersistentKernelCache()
    rmse_threshold = 0.0092 if dtype == ttnn.bfloat8_b else 0.0093
    run_test_windowed_sdpa(
        device, b, nh, nkv, s, d, cu_window_seqlens, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [256], ids=["q256"])
@pytest.mark.parametrize("k_chunk_size", [256], ids=["k256"])
def test_windowed_sdpa_large_sequence(device):
    """Test windowed SDPA with large sequence lengths."""
    b, nh, nkv, s, d = 1, 8, 1, 8192, 128
    # Create 8 windows of 1024 tokens each
    cu_window_seqlens = [i * 1024 for i in range(9)]  # [0, 1024, 2048, ..., 8192]

    ttnn.device.DisablePersistentKernelCache()
    run_test_windowed_sdpa(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        cu_window_seqlens,
        q_chunk_size,
        k_chunk_size,
        dtype,
        use_high_precision_compute=True,
        rmse_threshold=0.01,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
def test_windowed_sdpa_edge_cases(device):
    """Test windowed SDPA edge cases."""
    # Test 1: Single window (should behave like regular attention)
    b, nh, nkv, s, d = 1, 8, 1, 512, 128
    cu_window_seqlens = [0, 512]  # Single window
    run_test_windowed_sdpa(device, b, nh, nkv, s, d, cu_window_seqlens, 128, 128, dtype, rmse_threshold=0.01)

    # Test 2: Many small windows
    cu_window_seqlens = list(range(0, 513, 32))  # 16 windows of 32 tokens each
    run_test_windowed_sdpa(device, b, nh, nkv, s, d, cu_window_seqlens, 128, 128, dtype, rmse_threshold=0.01)

    # Test 3: Uneven window sizes
    cu_window_seqlens = [0, 100, 300, 400, 512]  # Windows of sizes: 100, 200, 100, 112
    run_test_windowed_sdpa(device, b, nh, nkv, s, d, cu_window_seqlens, 128, 128, dtype, rmse_threshold=0.01)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
def test_windowed_sdpa_program_cache(device):
    """Test that windowed SDPA uses program cache correctly."""
    b, nh, nkv, s, d = 1, 8, 1, 2048, 128
    cu_window_seqlens = [0, 512, 1024, 1536, 2048]
    q_chunk_size, k_chunk_size = 256, 256

    # Run twice
    for _ in range(2):
        run_test_windowed_sdpa(device, b, nh, nkv, s, d, cu_window_seqlens, q_chunk_size, k_chunk_size, dtype)

    # Check program cache
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 program cache entry, got {device.num_program_cache_entries()}"


def test_windowed_sdpa_performance(device):
    """Compare performance of windowed SDPA vs regular SDPA with mask."""
    import time

    b, nh, nkv, s, d = 1, 8, 1, 4096, 128
    cu_window_seqlens = [i * 512 for i in range(9)]  # 8 windows of 512 each
    q_chunk_size, k_chunk_size = 256, 256
    dtype = ttnn.bfloat16

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create inputs
    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Time windowed SDPA
    tt_cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
    start_time = time.time()
    _ = ttnn.transformer.windowed_scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        cu_window_seqlens=tt_cu_window_seqlens,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    windowed_time = time.time() - start_time

    # Time regular SDPA with mask
    attention_mask = create_windowed_attention_mask(cu_window_seqlens, s)
    tt_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = time.time()
    _ = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=False,
        attn_mask=tt_mask,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    regular_time = time.time() - start_time

    logger.info(f"Windowed SDPA time: {windowed_time:.4f}s")
    logger.info(f"Regular SDPA with mask time: {regular_time:.4f}s")
    logger.info(f"Speedup: {regular_time/windowed_time:.2f}x")

    # Windowed should be at least as fast (usually faster due to no mask transfer)
    assert windowed_time <= regular_time * 1.1, "Windowed SDPA should not be significantly slower than regular SDPA"
