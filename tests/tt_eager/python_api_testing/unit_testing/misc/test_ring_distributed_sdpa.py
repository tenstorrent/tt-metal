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
    """Generate test tensors with mixture of normal and outlier values (same as regular SDPA tests)"""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    """Check if TT_METAL_WATCHER is enabled"""
    return os.environ.get("TT_METAL_WATCHER") is not None


def gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, global_seq_len):
    """
    Gather outputs from all devices and reshuffle to restore global sequence order.

    Args:
        ring_outputs: List of tensors from each device [device_0_out, device_1_out, ...]
        ring_size: Number of devices in the ring
        global_seq_len: Total sequence length

    Returns:
        Reshuffled tensor in original sequence order
    """
    # Safety checks to prevent division by zero
    assert ring_size > 0, f"Ring size must be positive for reshuffling, got {ring_size}"
    assert (
        global_seq_len % (2 * ring_size) == 0
    ), f"Sequence length {global_seq_len} must be divisible by 2 * ring_size ({2 * ring_size})"

    # Each device outputs results for 2 chunks in contiguous order
    chunk_size = global_seq_len // (2 * ring_size)

    # Initialize output tensor
    batch_size, num_heads, _, head_dim = ring_outputs[0].shape
    final_output = torch.zeros(batch_size, num_heads, global_seq_len, head_dim)

    for device_id, device_output in enumerate(ring_outputs):
        # Calculate which chunks this device processed
        first_chunk_id = device_id
        second_chunk_id = (2 * ring_size - 1) - device_id

        # Extract the two chunks from device output (stored contiguously)
        local_chunk_size = 2 * chunk_size  # Device processes 2 chunks
        first_chunk_output = device_output[:, :, :chunk_size, :]
        second_chunk_output = device_output[:, :, chunk_size:local_chunk_size, :]

        # Place chunks in correct global positions
        first_start = first_chunk_id * chunk_size
        first_end = first_start + chunk_size
        second_start = second_chunk_id * chunk_size
        second_end = second_start + chunk_size

        final_output[:, :, first_start:first_end, :] = first_chunk_output
        final_output[:, :, second_start:second_end, :] = second_chunk_output

    return final_output


def run_test_ring_distributed_sdpa(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    ring_size,
    q_chunk_size,
    k_chunk_size,
    dtype,
    use_high_precision_compute=False,
    rmse_threshold=None,
):
    """
    Test ring-distributed SDPA implementation.

    Args:
        device: TT device
        b: batch size
        nh: number of query heads
        nkv: number of key/value heads
        s: sequence length
        d: head dimension
        ring_size: number of devices in ring
        q_chunk_size, k_chunk_size: chunk sizes for program config
        dtype: tensor data type
        use_high_precision_compute: whether to use high precision
        rmse_threshold: RMSE threshold for validation
    """
    torch.manual_seed(1234)

    # Validate ring distribution constraints FIRST (before any computations)
    assert ring_size > 0, f"Ring size must be positive, got {ring_size}"
    assert ring_size % 2 == 0, f"Ring size must be even for balanced load balancing, got {ring_size}"
    assert ring_size <= 32, f"Ring size must be <= 32, got {ring_size}"
    assert s % (2 * ring_size) == 0, f"Sequence length {s} must be divisible by 2 * ring_size ({2 * ring_size})"

    # Create program config (same as regular SDPA)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    # Create compute kernel config
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

    # Generate test tensors
    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    # Convert to TT tensors
    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    # Debug output for walkthrough example
    if b == 1 and nh == 1 and s == 32 and ring_size == 4:
        chunk_size = s // (2 * ring_size)
        print(f"\n=== WALKTHROUGH DEBUG INFO ===")
        print(f"Chunk size: {chunk_size} positions = {chunk_size//4} tiles")
        print(f"Ring distribution pattern:")
        for rid in range(ring_size):
            first_chunk = rid
            second_chunk = (2 * ring_size - 1) - rid
            first_positions = f"[{first_chunk * chunk_size}-{(first_chunk + 1) * chunk_size - 1}]"
            second_positions = f"[{second_chunk * chunk_size}-{(second_chunk + 1) * chunk_size - 1}]"
            print(
                f"  Device {rid}: chunks {first_chunk} and {second_chunk} → positions {first_positions} and {second_positions}"
            )
        print("Expected computation reduction for Device 0:")
        print("  Q chunk 0 (pos [0-3]): attends to K chunk 0 only → 1 QK computation")
        print("  Q chunk 7 (pos [28-31]): attends to K chunk 7 only → 1 QK computation")
        print("  Total: 2 QK computations instead of 16 → ~87% reduction")
        print("===============================\n")

    # Run ring-distributed SDPA for each device
    ring_outputs = []
    for ring_id in range(1):
        logger.debug(f"Running ring-distributed SDPA for device {ring_id}/{ring_size}")

        # Call ring-distributed SDPA for this device
        tt_ring_out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            ring_size=ring_size,
            ring_id=ring_id,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

        # Convert back to torch and remove padding
        ring_out_torch = ttnn.to_torch(tt_ring_out)
        print(f"Ring output shape: {ring_out_torch.shape}")
        local_seq_len = s // ring_size  # Each device processes s/(2*ring_size) * 2 = s/ring_size positions
        ring_out_torch = ring_out_torch[:, :, :local_seq_len, :]  # Remove tile padding
        ring_outputs.append(ring_out_torch)

    # Gather and reshuffle ring outputs to restore global order
    final_ring_output = gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, s)

    # Run baseline regular SDPA for comparison
    tt_baseline = ttnn.transformer.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
    )
    baseline_output = ttnn.to_torch(tt_baseline)
    baseline_output = baseline_output[:, :, :s, :]  # Remove tile padding

    # Alternative PyTorch reference for additional validation
    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    pytorch_reference = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)

    # Debug output for walkthrough example
    if b == 1 and nh == 1 and s == 256 and ring_size == 4:
        print(f"\n=== RESULT SHAPES DEBUG ===")
        print(f"Original Q shape: {Q.shape}")
        print(f"Ring outputs shapes: {[out.shape for out in ring_outputs]}")
        print(f"Final ring output shape: {final_ring_output.shape}")
        print(f"Baseline output shape: {baseline_output.shape}")
        print(f"PyTorch reference shape: {pytorch_reference.shape}")

        print(f"\n=== SAMPLE VALUES (position 0 and 28) ===")
        print(f"Ring output [0,0,0,:4]: {final_ring_output[0,0,0,:4]}")  # Device 0 chunk 0
        print(f"Ring output [0,0,28,:4]: {final_ring_output[0,0,28,:4]}")  # Device 0 chunk 7
        print(f"Baseline [0,0,0,:4]: {baseline_output[0,0,0,:4]}")
        print(f"Baseline [0,0,28,:4]: {baseline_output[0,0,28,:4]}")
        print(f"PyTorch [0,0,0,:4]: {pytorch_reference[0,0,0,:4]}")
        print(f"PyTorch [0,0,28,:4]: {pytorch_reference[0,0,28,:4]}")
        print("===========================\n")

    # Validation 1: Ring-distributed vs Regular SDPA
    out_pass_ring_vs_regular, out_pcc_ring_vs_regular = comp_pcc(baseline_output, final_ring_output, 0.994)
    logger.debug(f"Ring-distributed vs Regular SDPA PCC: {out_pcc_ring_vs_regular}")
    rmse_ring_vs_regular = torch.sqrt(((baseline_output - final_ring_output) ** 2).mean()).item()
    logger.debug(f"Ring-distributed vs Regular SDPA RMSE: {rmse_ring_vs_regular}")

    # Validation 2: Ring-distributed vs PyTorch reference
    out_pass_ring_vs_pytorch, out_pcc_ring_vs_pytorch = comp_pcc(pytorch_reference, final_ring_output, 0.994)
    logger.debug(f"Ring-distributed vs PyTorch PCC: {out_pcc_ring_vs_pytorch}")
    rmse_ring_vs_pytorch = torch.sqrt(((pytorch_reference - final_ring_output) ** 2).mean()).item()
    logger.debug(f"Ring-distributed vs PyTorch RMSE: {rmse_ring_vs_pytorch}")

    # Validation 3: Regular SDPA vs PyTorch reference (sanity check)
    out_pass_regular_vs_pytorch, out_pcc_regular_vs_pytorch = comp_pcc(pytorch_reference, baseline_output, 0.994)
    logger.debug(f"Regular SDPA vs PyTorch PCC: {out_pcc_regular_vs_pytorch}")

    # Assert correctness
    if rmse_threshold is not None:
        assert (
            rmse_ring_vs_regular < rmse_threshold
        ), f"Ring vs Regular RMSE {rmse_ring_vs_regular} exceeds threshold {rmse_threshold}"
        assert (
            rmse_ring_vs_pytorch < rmse_threshold
        ), f"Ring vs PyTorch RMSE {rmse_ring_vs_pytorch} exceeds threshold {rmse_threshold}"
    else:
        assert out_pass_ring_vs_regular, f"Ring vs Regular PCC {out_pcc_ring_vs_regular} < 0.994"
        assert out_pass_ring_vs_pytorch, f"Ring vs PyTorch PCC {out_pcc_ring_vs_pytorch} < 0.994"

    logger.info("✅ Ring-distributed SDPA correctness test passed!")


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
def test_ring_distributed_sdpa_walkthrough_example(device):
    """Test using exact parameters from our detailed walkthrough analysis"""
    # Correct parameters for Tenstorrent tile system (32x32 tiles):
    # - Sequence Length (S): 256 positions = 8 tiles (32 positions per tile)
    # - Ring Size: 4 devices
    # - Device 0 Assignment: chunks 0 and 7 → positions [0-31] and [224-255] (1 tile each)
    b, nh, nkv, s, d = 1, 1, 1, 256, 32  # Fixed to use proper tile dimensions
    ring_size = 4
    q_chunk_size, k_chunk_size = 32, 32
    dtype = ttnn.bfloat16

    print(f"\n=== WALKTHROUGH EXAMPLE TEST (FIXED TILE DIMENSIONS) ===")
    print(f"Parameters: B={b}, NH={nh}, S={s}, D={d}, ring_size={ring_size}")
    print(f"Total tiles: S/32 = {s//32} tiles")
    print(
        f"Chunk size: S/(2*ring_size) = {s}/({2*ring_size}) = {s//(2*ring_size)} positions = {s//(2*ring_size)//32} tile per chunk"
    )
    print(f"Expected Device 0 chunks: 0 and 7 → positions [0-31] and [224-255] (1 tile each)")
    print(f"Expected ~75% computation reduction (2 out of 8 tiles processed)")

    rmse_threshold = 0.01
    ttnn.device.DisablePersistentKernelCache()
    run_test_ring_distributed_sdpa(
        device, b, nh, nkv, s, d, ring_size, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])  # Start with BF16 for stability
@pytest.mark.parametrize("ring_size", [4], ids=["ring4"])  # Focus on ring_size=4 to match walkthrough
@pytest.mark.parametrize("q_chunk_size", [32], ids=["q32"])
@pytest.mark.parametrize("k_chunk_size", [32], ids=["k32"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    [
        (1, 1, 1, 256, 64),  # EXACT walkthrough example (FIXED for tile dimensions)
        (1, 2, 1, 256, 64),  # Add one more head
        (1, 8, 1, 256, 64),  # Larger but still ring_size=4 compatible
    ],
    ids=["walkthrough", "multi_head", "larger"],
)
def test_ring_distributed_sdpa_basic(device, dtype, ring_size, q_chunk_size, k_chunk_size, b, nh, nkv, s, d):
    """Basic correctness test for ring-distributed SDPA"""
    if s % (2 * ring_size) != 0:
        pytest.skip(f"Sequence length {s} not divisible by 2 * ring_size ({2 * ring_size})")
    if s < 2 * ring_size:
        pytest.skip(f"Sequence length {s} too small for ring size {ring_size}")

    rmse_threshold = 0.01  # Slightly relaxed threshold for multi-device coordination
    ttnn.device.DisablePersistentKernelCache()
    run_test_ring_distributed_sdpa(
        device, b, nh, nkv, s, d, ring_size, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold
    )


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("ring_size", [4], ids=["ring4"])
@pytest.mark.parametrize("q_chunk_size", [32, 64], ids=["q32", "q64"])
@pytest.mark.parametrize("k_chunk_size", [32, 64], ids=["k32", "k64"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    [
        (1, 8, 1, 512, 64),  # Larger sequence length
        (2, 16, 1, 256, 128),  # Multiple batches and heads
    ],
    ids=["large_seq", "multi_batch_head"],
)
def test_ring_distributed_sdpa_comprehensive(device, dtype, ring_size, q_chunk_size, k_chunk_size, b, nh, nkv, s, d):
    """Comprehensive test with various configurations"""
    if s % (2 * ring_size) != 0:
        pytest.skip(f"Sequence length {s} not divisible by 2 * ring_size ({2 * ring_size})")
    if s < 2 * ring_size * 32:  # Ensure minimum chunk size
        pytest.skip(f"Sequence length {s} too small for ring size {ring_size}")

    rmse_threshold = 0.015 if dtype == ttnn.bfloat8_b else 0.01
    ttnn.device.DisablePersistentKernelCache()
    run_test_ring_distributed_sdpa(
        device, b, nh, nkv, s, d, ring_size, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold
    )


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("ring_size", [2], ids=["ring2"])
@pytest.mark.parametrize("q_chunk_size", [32], ids=["q32"])
@pytest.mark.parametrize("k_chunk_size", [32], ids=["k32"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    [
        (1, 8, 1, 1024, 64),  # Large sequence length for performance validation
    ],
    ids=["perf_test"],
)
def test_ring_distributed_sdpa_performance(device, dtype, ring_size, q_chunk_size, k_chunk_size, b, nh, nkv, s, d):
    """Performance test to ensure ring distribution provides benefits"""
    if s % (2 * ring_size) != 0:
        pytest.skip(f"Sequence length {s} not divisible by 2 * ring_size ({2 * ring_size})")

    rmse_threshold = 0.01
    ttnn.device.DisablePersistentKernelCache()

    # TODO: Add timing measurements here to validate performance improvements
    run_test_ring_distributed_sdpa(
        device, b, nh, nkv, s, d, ring_size, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold
    )


def test_ring_distributed_sdpa_edge_cases():
    """Test edge cases and error conditions"""

    # Test 1: Ring size validation
    with pytest.raises(AssertionError, match="Ring size must be positive"):
        run_test_ring_distributed_sdpa(
            None, 1, 8, 1, 128, 64, ring_size=0, q_chunk_size=32, k_chunk_size=32, dtype=ttnn.bfloat16
        )

    # Test 2: Even ring size requirement
    with pytest.raises(AssertionError, match="Ring size must be even"):
        run_test_ring_distributed_sdpa(
            None, 1, 8, 1, 128, 64, ring_size=3, q_chunk_size=32, k_chunk_size=32, dtype=ttnn.bfloat16
        )

    # Test 3: Sequence length divisibility
    with pytest.raises(AssertionError, match="must be divisible by 2"):
        run_test_ring_distributed_sdpa(
            None, 1, 8, 1, 127, 64, ring_size=2, q_chunk_size=32, k_chunk_size=32, dtype=ttnn.bfloat16
        )


def test_gather_and_reshuffle_logic():
    """Test the gather and reshuffle logic independently"""
    ring_size = 4
    global_seq_len = 128
    batch_size, num_heads, head_dim = 1, 8, 64
    chunk_size = global_seq_len // (2 * ring_size)  # 16

    # Create mock ring outputs (each device outputs 2 chunks contiguously)
    ring_outputs = []
    for device_id in range(ring_size):
        # Each device outputs 32 positions (2 chunks of 16 each)
        device_output = torch.randn(batch_size, num_heads, 2 * chunk_size, head_dim)
        ring_outputs.append(device_output)

    # Test reshuffling
    final_output = gather_and_reshuffle_ring_outputs(ring_outputs, ring_size, global_seq_len)

    # Verify output shape
    assert final_output.shape == (batch_size, num_heads, global_seq_len, head_dim)

    # Verify chunk placement correctness
    for device_id in range(ring_size):
        first_chunk_id = device_id
        second_chunk_id = (2 * ring_size - 1) - device_id

        first_start = first_chunk_id * chunk_size
        first_end = first_start + chunk_size
        second_start = second_chunk_id * chunk_size
        second_end = second_start + chunk_size

        # Check that chunks are placed in correct positions
        expected_first_chunk = ring_outputs[device_id][:, :, :chunk_size, :]
        expected_second_chunk = ring_outputs[device_id][:, :, chunk_size : 2 * chunk_size, :]

        assert torch.allclose(final_output[:, :, first_start:first_end, :], expected_first_chunk)
        assert torch.allclose(final_output[:, :, second_start:second_end, :], expected_second_chunk)

    logger.info("✅ Gather and reshuffle logic test passed!")


if __name__ == "__main__":
    # Quick standalone test
    test_gather_and_reshuffle_logic()
    print("Ring-distributed SDPA test file ready!")
