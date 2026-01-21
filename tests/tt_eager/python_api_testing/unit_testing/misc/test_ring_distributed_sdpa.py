# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import math
import time
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
import ttnn
from loguru import logger
import pytest

torch.set_printoptions(threshold=torch.inf, linewidth=200, edgeitems=20)


def fa_rand(*shape):
    """Generate test tensors with mixture of normal and outlier values"""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    """Check if TT_METAL_WATCHER is enabled"""
    return os.environ.get("TT_METAL_WATCHER") is not None


def gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, global_seq_len):
    """Gather outputs from all devices and reshuffle to restore global sequence order."""
    assert ring_size > 0, f"Ring size must be positive for reshuffling, got {ring_size}"
    assert (
        global_seq_len % (2 * ring_size) == 0
    ), f"Sequence length {global_seq_len} must be divisible by 2 * ring_size ({2 * ring_size})"

    chunk_size = global_seq_len // (2 * ring_size)
    batch_size, num_heads, _, head_dim = ring_outputs[0].shape
    final_output = torch.zeros(batch_size, num_heads, global_seq_len, head_dim)

    for device_id, device_output in enumerate(ring_outputs):
        first_chunk_id = device_id
        second_chunk_id = (2 * ring_size - 1) - device_id

        local_chunk_size = 2 * chunk_size
        first_chunk_output = device_output[:, :, :chunk_size, :]
        second_chunk_output = device_output[:, :, chunk_size:local_chunk_size, :]

        first_start = first_chunk_id * chunk_size
        first_end = first_start + chunk_size
        second_start = second_chunk_id * chunk_size
        second_end = second_start + chunk_size

        final_output[:, :, first_start:first_end, :] = first_chunk_output
        final_output[:, :, second_start:second_end, :] = second_chunk_output

    return final_output


def run_test_ring_distributed_sdpa(device, b, s, ring_size, q_chunk_size, k_chunk_size):
    """Test ring-distributed SDPA implementation with fixed parameters."""
    torch.manual_seed(1234)

    # Fixed parameters as requested
    nh, nkv, d = 8, 1, 128
    dtype = ttnn.bfloat8_b

    # Validate ring distribution constraints
    assert ring_size > 0, f"Ring size must be positive, got {ring_size}"
    assert ring_size % 2 == 0, f"Ring size must be even, got {ring_size}"
    assert s % (2 * ring_size) == 0, f"Sequence length {s} must be divisible by 2 * ring_size ({2 * ring_size})"

    # Create program config
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    # Create compute kernel config
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Generate test tensors
    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    # Convert to TT tensors
    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    # Run ring-distributed SDPA for each device
    ring_outputs = []
    ttnn.synchronize_device(device)
    ring_start_time = time.time()
    for ring_id in range(ring_size):
        tt_ring_out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            ring_size=ring_size,
            ring_id=ring_id,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

        ring_out_torch = ttnn.to_torch(tt_ring_out)
        local_seq_len = s // ring_size
        ring_out_torch = ring_out_torch[:, :, :local_seq_len, :]
        ring_outputs.append(ring_out_torch)
    ttnn.synchronize_device(device)
    ring_end_time = time.time()
    ring_time = ring_end_time - ring_start_time

    # Gather and reshuffle ring outputs
    final_ring_output = gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, s)

    # Run baseline regular SDPA for comparison
    tt_baseline = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    baseline_output = ttnn.to_torch(tt_baseline)

    # Validation: Ring-distributed vs Regular SDPA
    out_pass, out_pcc = comp_pcc(baseline_output, final_ring_output, 0.99)
    logger.debug(f"Ring-distributed vs Regular SDPA PCC: {out_pcc}")

    # Log timing information
    logger.info(
        f"Timing (seq_len={s}, q_chunk={q_chunk_size}, k_chunk={k_chunk_size}): "
        f"ring_distributed={ring_time*1000:.2f}ms"
    )

    assert out_pass, f"Ring vs Regular PCC {out_pcc} < 0.99"
    logger.info("Ring-distributed SDPA correctness test passed!")


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize(
    "s",
    [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    ids=["s1k", "s2k", "s4k", "s8k", "s16k", "s32k", "s64k", "s128k"],
)
def test_ring_distributed_sdpa_main(device, q_chunk_size, k_chunk_size, s):
    """Main test with powers of 2 sequence lengths, fixed nh=8, nkv=1, d=128, dtype=bfloat8_b"""
    b, ring_size = 1, 4

    # Skip if sequence length not compatible with ring size
    if s % (2 * ring_size) != 0:
        pytest.skip(f"Sequence length {s} not divisible by 2 * ring_size ({2 * ring_size})")

    if s / (2 * ring_size) < q_chunk_size:
        pytest.skip(f"Sequence length {s} not compatible with ring size {ring_size} and q_chunk_size {q_chunk_size}")
    run_test_ring_distributed_sdpa(device, b, s, ring_size, q_chunk_size, k_chunk_size)


def run_test_ring_distributed_sdpa_with_prefix_and_paged_kv(
    device, b, s, ring_size, q_chunk_size, k_chunk_size, prefix_len, page_block_size
):
    """Test ring-distributed SDPA with both prefix caching and paged KV cache."""
    torch.manual_seed(1234)

    # Fixed parameters
    nh, nkv, d = 8, 1, 128
    dtype = ttnn.bfloat8_b

    # Create program config
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

    # Generate full sequence tensors
    Q_full = fa_rand(b, nh, s, d)
    K_full = fa_rand(b, nkv, s, d)
    V_full = fa_rand(b, nkv, s, d)

    # Create shorter Q (prefix caching)
    Q_short = Q_full[:, :, prefix_len:, :]
    q_seq_len = s - prefix_len

    # Prepare paged KV cache for full sequence
    max_num_blocks_per_seq = s // page_block_size
    max_num_blocks = b * max_num_blocks_per_seq

    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq)

    def page_cache(cache):
        paged_cache = (
            cache.reshape(b, nkv, max_num_blocks_per_seq, page_block_size, d)
            .transpose(1, 2)
            .reshape(max_num_blocks, nkv, page_block_size, d)
        )
        shuffled_page_cache = paged_cache[permutation]
        return shuffled_page_cache

    paged_K = page_cache(K_full)
    paged_V = page_cache(V_full)

    # Convert to TT tensors
    tt_Q = ttnn.from_torch(Q_short, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_paged_K = ttnn.Tensor(paged_K, dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_paged_V = ttnn.Tensor(paged_V, dtype).to(ttnn.TILE_LAYOUT).to(device)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # Run ring-distributed SDPA with prefix caching and paged KV
    ring_outputs = []
    ttnn.synchronize_device(device)
    ring_start_time = time.time()
    for ring_id in range(ring_size):
        tt_ring_out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            tt_Q,
            tt_paged_K,
            tt_paged_V,
            ring_size=ring_size,
            ring_id=ring_id,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            page_table=page_table_tt,
            chunk_start_idx=prefix_len,  # Prefix caching offset
        )

        ring_out_torch = ttnn.to_torch(tt_ring_out)
        local_seq_len = q_seq_len // ring_size
        ring_out_torch = ring_out_torch[:, :, :local_seq_len, :]
        ring_outputs.append(ring_out_torch)
    ttnn.synchronize_device(device)
    ring_end_time = time.time()
    ring_time = ring_end_time - ring_start_time

    # Gather and reshuffle ring outputs
    final_ring_output = gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, q_seq_len)

    # Run baseline: chunked SDPA for comparison
    tt_chunked_out = ttnn.transformer.chunked_scaled_dot_product_attention(
        input_tensor_q=tt_Q,
        input_tensor_k=tt_paged_K,
        input_tensor_v=tt_paged_V,
        page_table_tensor=page_table_tt,
        chunk_start_idx=prefix_len,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
    )
    baseline_output = ttnn.to_torch(tt_chunked_out)

    # Validation
    out_pass, out_pcc = comp_pcc(baseline_output, final_ring_output, 0.99)
    logger.debug(f"Ring-distributed with prefix+paged KV vs Chunked SDPA PCC: {out_pcc}")
    assert out_pass, f"Ring with prefix+paged KV vs Chunked PCC {out_pcc} < 0.99"
    logger.info(
        f"Ring-distributed SDPA with prefix caching (prefix_len={prefix_len}) and paged KV (page_block_size={page_block_size}) test passed!"
    )

    # Log timing information
    logger.info(
        f"Timing (New tokens seq_len={s-prefix_len}, prefix_len={prefix_len}, page_block_size={page_block_size}, q_chunk={q_chunk_size}, k_chunk={k_chunk_size}): "
        f"ring_distributed={ring_time*1000:.2f}ms"
    )


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize(
    "s",
    [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    ids=["s1k", "s2k", "s4k", "s8k", "s16k", "s32k", "s64k", "s128k"],
)
@pytest.mark.parametrize(
    "prefix_len",
    [0, 64, 128, 256, 512],
    ids=["p0", "p64", "p128", "p256", "p512"],
)
@pytest.mark.parametrize("page_block_size", [64], ids=["b64"])
@pytest.mark.parametrize("q_chunk_size", [64, 256], ids=["q64", "q256"])
@pytest.mark.parametrize("k_chunk_size", [64, 512], ids=["k64", "k512"])
def test_ring_distributed_sdpa_prefix_and_paged_kv(device, s, prefix_len, page_block_size, q_chunk_size, k_chunk_size):
    """Test ring-distributed SDPA with both prefix caching and paged KV cache."""
    b, ring_size = 1, 4

    # Skip if constraints not met

    # ring_distributed_sdpa_device_operation.cpp:218
    if s % (2 * ring_size) != 0:
        pytest.skip(f"Sequence length {s} not divisible by 2 * ring_size ({2 * ring_size})")

    # ring_distributed_sdpa_device_operation.cpp:137
    if prefix_len % q_chunk_size != 0:
        pytest.skip(f"prefix_len {prefix_len} not divisible by q_chunk_size {q_chunk_size}")

    # ring_distributed_sdpa_device_operation.cpp:154
    if s % page_block_size != 0:
        pytest.skip(f"Sequence length {s} not divisible by page_block_size {page_block_size}")

    # ring_distributed_sdpa_device_operation.cpp: 160
    if prefix_len % page_block_size != 0:
        pytest.skip(f"prefix_len {prefix_len} not divisible by page_block_size {page_block_size}")

    # ring_distributed_sdpa_device_operation.cpp:240
    if q_chunk_size > s / (2 * ring_size):
        pytest.skip(
            f"q_chunk_size {q_chunk_size} must be less or equal to per-device sequence length {s / (2 * ring_size)} for sequence length {s} and ring size {ring_size}."
        )

    # ring_distributed_sdpa_device_operation.cpp:249
    if (s / (2 * ring_size)) % q_chunk_size != 0:
        pytest.skip(
            f"per-device sequence length {s / (2 * ring_size)} not divisible by q_chunk_size {q_chunk_size} for sequence length {s} and ring size {ring_size}."
        )

    # ring_distributed_sdpa_device_operation.cpp:166
    if (s + prefix_len) % k_chunk_size != 0:
        pytest.skip(f"(s+prefix_len) {s+prefix_len} not divisible by k_chunk_size {k_chunk_size}")

    run_test_ring_distributed_sdpa_with_prefix_and_paged_kv(
        device, b, prefix_len + s, ring_size, q_chunk_size, k_chunk_size, prefix_len, page_block_size
    )


if __name__ == "__main__":
    print("Minimal ring-distributed SDPA test file ready!")
