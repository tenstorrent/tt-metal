# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for testing distributed layer norm and RMS norm operations.

This module provides reusable test functions that can be called from test files,
making it easy to add new test cases with minimal overhead.

MAIN FUNCTION:
==============
run_distributed_norm_test() - The primary test function that:
  1. Generates test data with specified distribution
  2. Computes PyTorch reference output
  3. Computes TTNN distributed normalization output
  4. Compares results using torch.allclose
  5. Returns pass/fail status and detailed error metrics

HELPER FUNCTIONS: =================
- setup_ccl_semaphores(): Setup CCL semaphores for distributed operations
- generate_test_data(): Generate input tensors with specified distribution
- compute_torch_reference(): Compute PyTorch reference normalization
- compute_ttnn_distributed_norm(): Compute TTNN distributed normalization
- check_average_relative_diff(): Check if average relative difference is under 5%

USAGE:
======
Import and call run_distributed_norm_test() from your test file:

    from tests.ttnn.unit_tests.operations.fused.distributed_norm_test_utils import run_distributed_norm_test

    passes, max_abs_diff, max_rel_diff, mean_rel_diff = run_distributed_norm_test(
        mesh_device=mesh_device,
        batch_size=1,
        seq_len=1024,
        hidden_dim=4096,
        eps=1e-6,
        norm_type="layer_norm",
    )

    assert passes, (
        f"TEST FAILED: Average relative difference {mean_rel_diff*100:.2f}% exceeds 5% threshold | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e}"
    )

NOTE: Test passes only if average relative difference is under 5%
"""

import pytest
import torch
import ttnn
from loguru import logger


def setup_ccl_semaphores(mesh_device):
    """Setup CCL semaphores for distributed operations."""
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]
    return ccl_semaphore_handles


def generate_test_data(batch_size, seq_len, hidden_dim, mean=0, var=1, outlier_pct=0, outlier_var=0, seed=42):
    """
    Generate test input data with specified distribution parameters.

    Args:
        batch_size: Batch size for input tensor
        seq_len: Sequence length
        hidden_dim: Hidden dimension size
        mean: Mean of the base distribution
        var: Variance of the base distribution
        outlier_pct: Percentage of outliers (0-1)
        outlier_var: Variance of outlier distribution
        seed: Random seed for reproducibility

    Returns:
        Tuple of (input, weight, bias) tensors in bfloat16
    """
    torch.manual_seed(seed)

    input_shape = (batch_size, 1, seq_len, hidden_dim)
    # torch_input = torch.ones(input_shape)
    torch_input = torch.randn(input_shape) * var + mean

    if outlier_pct > 0:
        # fa_rand - style input distribution: addition of two gaussian distributions
        torch_input = torch_input + torch.randn(input_shape) * outlier_var * torch.bernoulli(
            torch.full(input_shape, outlier_pct)
        )

    torch_weight = torch.randn(hidden_dim)
    torch_bias = torch.randn(hidden_dim)

    # Quantize to bfloat16
    torch_input = torch_input.to(torch.bfloat16)
    torch_weight = torch_weight.to(torch.bfloat16)
    torch_bias = torch_bias.to(torch.bfloat16)

    return torch_input, torch_weight, torch_bias


def compute_torch_reference(torch_input, torch_weight, torch_bias, norm_type, eps):
    """
    Compute PyTorch reference output for normalization.

    Args:
        torch_input: Input tensor
        torch_weight: Weight tensor
        torch_bias: Bias tensor
        norm_type: "layer_norm" or "rms_norm"
        eps: Epsilon value for numerical stability

    Returns:
        PyTorch reference output tensor
    """
    hidden_dim = torch_weight.shape[0]

    if norm_type == "layer_norm":
        torch_norm = torch.nn.LayerNorm(normalized_shape=hidden_dim, eps=eps)
        torch_norm.bias.data = torch_bias.clone().float()
    elif norm_type == "rms_norm":
        torch_norm = torch.nn.RMSNorm(normalized_shape=hidden_dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    torch_norm.weight.data = torch_weight.clone().float()

    with torch.no_grad():
        # Torch reference operates on float32 for high precision
        torch_output = torch_norm(torch_input.float()).type_as(torch_input)

    return torch_output


def compute_ttnn_distributed_norm(
    torch_input,
    torch_weight,
    torch_bias,
    mesh_device,
    norm_type,
    eps,
    use_legacy=False,
    use_high_precision=True,
    weight_layout=ttnn.TILE_LAYOUT,
    bias_layout=ttnn.TILE_LAYOUT,
    use_welford=True,
):
    """
    Compute TTNN distributed normalization output.

    Args:
        torch_input: Input tensor (torch)
        torch_weight: Weight tensor (torch)
        torch_bias: Bias tensor (torch)
        mesh_device: TTNN mesh device
        norm_type: "layer_norm" or "rms_norm"
        eps: Epsilon value
        use_legacy: Whether to use legacy reduction/rsqrt
        use_high_precision: Whether to use high precision compute config
        weight_layout: Memory layout for weight tensor
        bias_layout: Memory layout for bias tensor
        use_welford: Use Welford algorithm for variance computation

    Returns:
        TTNN output converted to torch tensor
    """
    hidden_dim = torch_weight.shape[0]
    num_mesh_devices = mesh_device.get_num_devices()

    # Convert to TTNN tensors
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    # Reshape and shard weight based on layout
    if weight_layout == ttnn.ROW_MAJOR_LAYOUT:
        # ROW_MAJOR: Reshape to (num_mesh_devices, 1, -1, 32) and shard over dim 0
        weight_shape = (num_mesh_devices, 1, -1, 32)
        weight_shard_dim = 0
    else:
        # TILE: Reshape to (1, 1, 1, hidden_dim) and shard over dim -1
        weight_shape = (1, 1, 1, hidden_dim)
        weight_shard_dim = -1

    ttnn_weight = ttnn.from_torch(
        torch_weight.reshape(weight_shape),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=weight_layout,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=weight_shard_dim),
    )

    ttnn_bias = None
    if norm_type == "layer_norm":
        # Reshape and shard bias based on layout
        if bias_layout == ttnn.ROW_MAJOR_LAYOUT:
            # ROW_MAJOR: Reshape to (num_mesh_devices, 1, -1, 32) and shard over dim 0
            bias_shape = (num_mesh_devices, 1, -1, 32)
            bias_shard_dim = 0
        else:
            # TILE: Reshape to (1, 1, 1, hidden_dim) and shard over dim -1
            bias_shape = (1, 1, 1, hidden_dim)
            bias_shard_dim = -1

        ttnn_bias = ttnn.from_torch(
            torch_bias.reshape(bias_shape),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=bias_layout,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=bias_shard_dim),
        )

    # Configure compute kernel
    if use_high_precision:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    # Configure program
    if use_legacy:
        program_config = ttnn.LayerNormDefaultProgramConfig(
            legacy_reduction=True,
            legacy_rsqrt=True,
            use_welford=False,
        )
    else:
        program_config = ttnn.LayerNormDefaultProgramConfig(
            legacy_reduction=False,
            legacy_rsqrt=False,
            use_welford=use_welford,
        )

    # Step 1: Compute local statistics
    if norm_type == "layer_norm":
        ttnn_stats = ttnn.layer_norm_pre_all_gather(
            ttnn_input,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            program_config=program_config,
        )
    elif norm_type == "rms_norm":
        ttnn_stats = ttnn.rms_norm_pre_all_gather(
            ttnn_input,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            program_config=program_config,
        )
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

    # Step 2: Gather statistics across devices
    ccl_semaphore_handles = setup_ccl_semaphores(mesh_device)
    ttnn.synchronize_device(mesh_device)

    ttnn_stats_gathered = ttnn.experimental.all_gather_async(
        ttnn_stats,
        dim=3,
        multi_device_global_semaphore=ccl_semaphore_handles,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        cluster_axis=1,
    )

    # Step 3: Compute normalization
    if norm_type == "layer_norm":
        ttnn_output = ttnn.layer_norm_post_all_gather(
            ttnn_input,
            ttnn_stats_gathered,
            epsilon=eps,
            weight=ttnn_weight,
            bias=ttnn_bias,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
        )
    elif norm_type == "rms_norm":
        ttnn_output = ttnn.rms_norm_post_all_gather(
            ttnn_input,
            ttnn_stats_gathered,
            epsilon=eps,
            weight=ttnn_weight,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
        )

    # Convert back to torch
    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    return ttnn_output_torch


def check_average_relative_diff(torch_output, ttnn_output_torch):
    """
    Calculate differences and check that average relative difference is under 5%.

    Args:
        torch_output: PyTorch reference output
        ttnn_output_torch: TTNN output converted to torch

    Returns:
        Tuple of (passes_avg_check, max_abs_diff, max_rel_diff, mean_rel_diff)
    """
    # Calculate differences for error reporting
    abs_diff = torch.abs(torch_output - ttnn_output_torch)
    max_abs_diff = torch.max(abs_diff).item()

    # Calculate relative error where torch_output is not near zero
    mask = torch.abs(torch_output) > 1e-5
    if mask.any():
        rel_diff = abs_diff[mask] / torch.abs(torch_output[mask])
        max_rel_diff = torch.max(rel_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
    else:
        max_rel_diff = 0.0
        mean_rel_diff = 0.0

    # Check if average relative difference is under 5%
    passes_avg_check = mean_rel_diff < 0.05

    return passes_avg_check, max_abs_diff, max_rel_diff, mean_rel_diff


def run_distributed_norm_test(
    mesh_device,
    batch_size,
    seq_len,
    hidden_dim,
    eps,
    norm_type,
    mean=0,
    var=1,
    outlier_pct=0,
    outlier_var=0,
    use_legacy=False,
    use_high_precision=True,
    verbose=False,
    weight_layout=ttnn.TILE_LAYOUT,
    bias_layout=ttnn.TILE_LAYOUT,
    use_welford=True,
):
    """
    Main test function for distributed normalization.

    Args:
        mesh_device: TTNN mesh device
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        eps: Epsilon for numerical stability
        norm_type: "layer_norm" or "rms_norm"
        mean: Mean of input distribution
        var: Variance of input distribution
        outlier_pct: Percentage of outliers
        outlier_var: Variance of outliers
        use_legacy: Use legacy reduction/rsqrt
        use_high_precision: Use high precision compute config
        verbose: Print results (default: False for minimal output)
        weight_layout: Memory layout for weight tensor (default: TILE_LAYOUT)
        bias_layout: Memory layout for bias tensor (default: TILE_LAYOUT)
        use_welford: Use Welford algorithm for variance computation (default: True)

    Returns:
        Tuple of (passes, max_abs_diff, max_rel_diff, mean_rel_diff)
        where passes is True only if average relative difference < 5%
    """
    # Validate that RMS norm is not called with Welford
    if norm_type == "rms_norm" and use_welford:
        pytest.skip(
            "INVALID TEST CONFIGURATION: RMS norm cannot be used with use_welford=True. "
            "This test should be removed or fixed. RMS norm only supports use_welford=False."
        )

    # Generate test data
    torch_input, torch_weight, torch_bias = generate_test_data(
        batch_size, seq_len, hidden_dim, mean, var, outlier_pct, outlier_var
    )

    # Compute PyTorch reference
    torch_output = compute_torch_reference(torch_input, torch_weight, torch_bias, norm_type, eps)

    # Compute TTNN output
    ttnn_output_torch = compute_ttnn_distributed_norm(
        torch_input,
        torch_weight,
        torch_bias,
        mesh_device,
        norm_type,
        eps,
        use_legacy,
        use_high_precision,
        weight_layout,
        bias_layout,
        use_welford,
    )

    # Check average relative difference
    passes, max_abs_diff, max_rel_diff, mean_rel_diff = check_average_relative_diff(torch_output, ttnn_output_torch)

    # Always log the results
    status = "PASS" if passes else "FAIL"
    avg_status = "✓" if passes else "✗"

    logger.info(
        f"{status} | "
        f"avg<5%:{avg_status} | "
        f"max_abs_diff={max_abs_diff:.6e} | "
        f"max_rel_diff={max_rel_diff:.6e} | "
        f"mean_rel_diff={mean_rel_diff:.6e} ({mean_rel_diff*100:.2f}%)"
    )

    return passes, max_abs_diff, max_rel_diff, mean_rel_diff
