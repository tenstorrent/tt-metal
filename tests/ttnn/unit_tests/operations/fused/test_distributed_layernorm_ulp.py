# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose_and_pcc, comp_ulp, calculate_detailed_ulp_stats


# Maps from tuple of (mean, var, outlier_pct, outlier_var) to expected max delta ULP threshold with new reduction and rsqrt
RMS_EXPECTED_MAX_ULP_THRESHOLD_WITH_NEW_REDUCTION = {
    (0, 1, 0, 0): 4.0,
    (0, 10, 0, 0): 3.0,
    (-10, 10, 0, 0): 4.0,
    (0, 1, 0.01, 10): 4.0,
}


def calculate_mse(expected, actual):
    """Calculate Mean Squared Error between two tensors"""
    if isinstance(actual, ttnn.Tensor):
        actual = ttnn.to_torch(actual)
    if isinstance(expected, ttnn.Tensor):
        expected = ttnn.to_torch(expected)

    mse = torch.nn.functional.mse_loss(expected.float(), actual.float())
    return mse.item()


def calculate_max_error(expected, actual):
    """
    Calculate the maximum error between two tensors and return the max error value
    along with the corresponding elements from expected and actual tensors
    """
    if isinstance(actual, ttnn.Tensor):
        actual = ttnn.to_torch(actual)
    if isinstance(expected, ttnn.Tensor):
        expected = ttnn.to_torch(expected)

    abs_diff = torch.abs(expected - actual)
    max_error = torch.max(abs_diff)
    max_idx = torch.argmax(abs_diff)

    return {
        "max_error": max_error.item(),
        "expected_value": expected.flatten()[max_idx].item(),
        "actual_value": actual.flatten()[max_idx].item(),
    }


def setup_ccl_semaphores(mesh_device):
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


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("hidden_dim", [4096])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(
    "mean, var, outlier_pct, outlier_var",
    [(0, 1, 0, 0), (0, 10, 0, 0), (-10, 10, 0, 0), (0, 1, 0.01, 10)],
    ids=["case1", "case2", "case3", "case4"],
)
@pytest.mark.parametrize("norm_type", ["layer_norm", "rms_norm"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_distributed_norm_comparison(
    mesh_device, batch_size, seq_len, hidden_dim, eps, mean, var, outlier_pct, outlier_var, norm_type
):
    torch.manual_seed(42)

    # Generate random input data
    input_shape = (batch_size, 1, seq_len, hidden_dim)
    torch_input = torch.randn(input_shape) * var + mean
    if outlier_pct > 0:
        # fa_rand - style input distribution. addition of two gaussian distributions, one with higher variance
        torch_input = torch_input + torch.randn(input_shape) * outlier_var * torch.bernoulli(
            torch.full(input_shape, outlier_pct)
        )
    logger.info(
        f"Input shape: {input_shape}, Mean: {mean}, Var: {var}, Outlier Pct: {outlier_pct}, Outlier Var: {outlier_var}"
    )
    torch_weight = torch.randn(hidden_dim)
    torch_bias = torch.randn(hidden_dim)

    # Quantize to bfloat16, as that's the standard dataformat for activations in neural networks
    torch_input = torch_input.to(torch.bfloat16)
    torch_weight = torch_weight.to(torch.bfloat16)
    torch_bias = torch_bias.to(torch.bfloat16)

    # PyTorch reference implementation
    if norm_type == "layer_norm":
        torch_norm = torch.nn.LayerNorm(normalized_shape=hidden_dim, eps=eps)
    elif norm_type == "rms_norm":
        torch_norm = torch.nn.RMSNorm(normalized_shape=hidden_dim, eps=eps)
    torch_norm.weight.data = torch_weight.clone().float()
    if norm_type == "layer_norm":
        torch_norm.bias.data = torch_bias.clone().float()

    with torch.no_grad():
        # Torch reference operates on float32 tensors to compute in high precision, commonly seen in LLM and DiT models.
        torch_output = torch_norm(torch_input.float()).type_as(torch_input)

    # TTNN implementation
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    N_DEV = 8
    ttnn_weight = ttnn.from_torch(
        torch_weight.reshape(N_DEV, 1, -1, 32),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    if norm_type == "layer_norm":
        ttnn_bias = ttnn.from_torch(
            torch_bias.reshape(N_DEV, 1, -1, 32),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

    # Use highest precision compute kernel config for comparison
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    """
    Distributed Layernorm operates in 3 parts.
    1. Compute local sum(x) and sum(x**2). Write into stats tensor.
    2. Gather stats tensor across devices.
    3. Reduce stats tensor. Compute variance and mean. Do normalization.
    """
    use_new = ttnn.LayerNormDistributedDefaultProgramConfig(
        legacy_reduction=False,
        legacy_rsqrt=False,
    )
    use_old = ttnn.LayerNormDistributedDefaultProgramConfig(
        legacy_reduction=True,
        legacy_rsqrt=True,
    )
    if norm_type == "layer_norm":
        ttnn_stats = ttnn.layer_norm_pre_all_gather(
            ttnn_input,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            distributed_program_config=use_new,
        )
    elif norm_type == "rms_norm":
        ttnn_stats = ttnn.rms_norm_pre_all_gather(
            ttnn_input,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            distributed_program_config=use_new,
        )
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

    if norm_type == "layer_norm":
        ttnn_output = ttnn.layer_norm_post_all_gather(
            ttnn_input,
            ttnn_stats_gathered,
            epsilon=eps,
            weight=ttnn_weight,
            bias=ttnn_bias,
            compute_kernel_config=compute_kernel_config,
        )
    elif norm_type == "rms_norm":
        ttnn_output = ttnn.rms_norm_post_all_gather(
            ttnn_input,
            ttnn_stats_gathered,
            epsilon=eps,
            weight=ttnn_weight,
            compute_kernel_config=compute_kernel_config,
            distributed_program_config=use_new,
        )

    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    # Calculate metrics
    mse = calculate_mse(torch_output, ttnn_output_torch)
    ulp_stats = calculate_detailed_ulp_stats(torch_output, ttnn_output_torch)
    if norm_type == "rms_norm":
        ulp_passed = comp_ulp(
            torch_output,
            ttnn_output_torch,
            ulp_threshold=RMS_EXPECTED_MAX_ULP_THRESHOLD_WITH_NEW_REDUCTION[(mean, var, outlier_pct, outlier_var)],
        )
    pcc_passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=0.99)
    max_error = calculate_max_error(torch_output, ttnn_output_torch)

    passes_allclose = torch.allclose(torch_output, ttnn_output_torch)

    # Print results
    logger.info(f"\n=== RMSNorm Comparison Results ===")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Hidden dim: {hidden_dim}, eps: {eps}")
    logger.info(f"")
    logger.info(f"Accuracy Metrics:")
    logger.info(f"  PCC: {pcc_value:.6f}")
    logger.info(f"  MSE: {mse:.6e}")
    logger.info(f"  Max Error: {max_error}")
    logger.info(f"")
    logger.info(f"ULP Error Analysis (bfloat16):")
    logger.info(f"  Max ULP: {ulp_stats['max_ulp']:.1f}")
    logger.info(f"  Mean ULP: {ulp_stats['mean_ulp']:.1f}")
    logger.info(f"  Median ULP: {ulp_stats['median_ulp']:.1f}")
    logger.info(f"  Std ULP: {ulp_stats['std_ulp']:.1f}")
    logger.info(f"  95th percentile ULP: {ulp_stats['p95_ulp']:.1f}")
    logger.info(f"  99th percentile ULP: {ulp_stats['p99_ulp']:.1f}")
    logger.info(f"  Perfect matches: {ulp_stats['perfect_matches']*100:.1f}%")
    logger.info(f"")
    logger.info(f"Test Results:")
    logger.info(f"  PCC Passed (>0.99): {pcc_passed}")
    if norm_type == "rms_norm":
        logger.info(f"  ULP Passed: {ulp_passed}")
    logger.info(f"  Allclose Passed: {passes_allclose}")
