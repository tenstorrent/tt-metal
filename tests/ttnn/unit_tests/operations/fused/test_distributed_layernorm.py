# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from models.common.utility_functions import tt2torch_tensor
from tests.ttnn.utils_for_testing import assert_with_pcc

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from tests.tests_common.skip_reasons import LEGACY_CCL_SKIP
from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def reference_layernorm(x, gamma, beta, epsilon, is_rmsnorm):
    if gamma is None:
        gamma = torch.ones(x.shape[-1])
    if beta is None:
        beta = torch.zeros(x.shape[-1])
    if is_rmsnorm:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) * gamma
    else:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, epsilon)


def tt_distributed_layernorm(inp, gamma, beta, epsilon, is_rmsnorm, compute_kernel_config, stats_dtype):
    pytest.skip(LEGACY_CCL_SKIP)
    # Run layernorm part 1
    if is_rmsnorm:
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=stats_dtype)
    else:
        tt_stats = ttnn.layer_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=stats_dtype)

    # Legacy ccl call removed until new implementation is done - see https://github.com/tenstorrent/tt-metal/issues/26649
    assert False, "Legacy ccl call removed until new implementation is done"
    # AllGather stats
    # tt_stats = ttnn.all_gather(tt_stats, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Run layernorm part 2
    if is_rmsnorm:
        tt_out = ttnn.rms_norm_post_all_gather(
            inp, tt_stats, epsilon=epsilon, weight=gamma, compute_kernel_config=compute_kernel_config
        )
    else:
        tt_out = ttnn.layer_norm_post_all_gather(
            inp, tt_stats, epsilon=epsilon, weight=gamma, bias=beta, compute_kernel_config=compute_kernel_config
        )

    tt_stats.deallocate(True)
    return tt_out


def run_distributed_layernorm(
    inp_shape,
    n_devices,
    is_rmsnorm,
    dtype,
    stats_dtype,
    mesh_device,
    has_weights=True,
    fp32_enabled=False,
    iterations=1,
):
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_enabled,
        packer_l1_acc=False,
    )

    torch.manual_seed(1234)

    canon_inp = torch.randn(inp_shape) * 4 - 1
    gamma = torch.rand(inp_shape[-1]) * 2 - 1
    beta = torch.rand(inp_shape[-1]) * 2 - 1
    gamma_chunked = gamma.chunk(n_devices, dim=-1)
    beta_chunked = beta.chunk(n_devices, dim=-1)
    inp_chunked = canon_inp.chunk(n_devices, dim=-1)
    epsilon = 1e-5

    tt_inp = ttnn.as_tensor(
        canon_inp,
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=-1),
    )
    tt_gamma = ttnn.as_tensor(
        gamma.reshape(n_devices, 1, -1, 32),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )
    tt_beta = ttnn.as_tensor(
        beta.reshape(n_devices, 1, -1, 32),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )

    if not has_weights:
        gamma = None
        beta = None
        tt_gamma = None
        tt_beta = None

    for i in range(iterations):
        tt_out = tt_distributed_layernorm(
            tt_inp, tt_gamma, tt_beta, epsilon, is_rmsnorm, compute_kernel_config, stats_dtype
        )
        tt_output_host = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    # reference impl
    out_torch = reference_layernorm(canon_inp, gamma, beta, epsilon, is_rmsnorm)

    passing, output_str = comp_allclose(tt_output_host, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"torch vs tt distributed layernorm = {output_str}")

    assert passing


inp_shapes = [
    (1, 1, 2048, 8192),
    (1, 1, 128, 8192),
    (2, 1, 128, 8192),
]
inp_shape_ids = ["inp_shape0", "inp_shape1", "inp_shape2"]

stats_dtypes = [ttnn.bfloat16, ttnn.bfloat8_b]
stats_dtypes_ids = ["BFLOAT16_stats", "BFLOAT8_B_stats"]

dtypes = [ttnn.bfloat16, ttnn.bfloat8_b]
dtype_ids = ["BFLOAT16_in", "BFLOAT8_B_in"]

rms_norm_parametrizations = [True, False]
rms_norm_parametrization_ids = ["rmsnorm", "layernorm"]


def run_test_distributed_layernorm_with_program_cache_and_checks(
    inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, mesh_device, iterations, has_weights=True
):
    if mesh_device.get_num_devices() < n_devices:
        pytest.skip("Not T3000!")

    run_distributed_layernorm(
        inp_shape,
        n_devices,
        is_rmsnorm,
        dtype,
        stats_dtype,
        mesh_device,
        has_weights=has_weights,
        iterations=iterations,
    )

    assert mesh_device.num_program_cache_entries() == 3, "Program cache should have only 3 entries, but has " + str(
        mesh_device.num_program_cache_entries()
    )


@pytest.mark.parametrize("iterations", [2], ids=["loops2"])
@pytest.mark.parametrize("dtype", dtypes, ids=dtype_ids)
@pytest.mark.parametrize("stats_dtype", stats_dtypes, ids=stats_dtypes_ids)
@pytest.mark.parametrize("inp_shape", inp_shapes, ids=inp_shape_ids)
@pytest.mark.parametrize("n_devices", [8])
@pytest.mark.parametrize("is_rmsnorm", rms_norm_parametrizations, ids=rms_norm_parametrization_ids)
def test_distributed_layernorm_with_program_cache(
    inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, iterations, t3k_mesh_device
):
    run_test_distributed_layernorm_with_program_cache_and_checks(
        inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, t3k_mesh_device, iterations=iterations
    )


@pytest.mark.parametrize("iterations", [2], ids=["loops2"])
@pytest.mark.parametrize("dtype", dtypes, ids=dtype_ids)
@pytest.mark.parametrize("stats_dtype", stats_dtypes, ids=stats_dtypes_ids)
@pytest.mark.parametrize("inp_shape", inp_shapes, ids=inp_shape_ids)
@pytest.mark.parametrize("n_devices", [4])
@pytest.mark.parametrize("is_rmsnorm", rms_norm_parametrizations, ids=rms_norm_parametrization_ids)
@pytest.mark.parametrize("has_weights", [True, False], ids=["has_weights", "no_weights"])
def test_distributed_layernorm_with_program_cache_4chip(
    inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, iterations, pcie_mesh_device, has_weights
):
    if not has_weights and is_rmsnorm:
        pytest.skip("RMSNorm does not support no weights")
    run_test_distributed_layernorm_with_program_cache_and_checks(
        inp_shape,
        n_devices,
        is_rmsnorm,
        dtype,
        stats_dtype,
        pcie_mesh_device,
        iterations=iterations,
        has_weights=has_weights,
    )


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


# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose_and_pcc


def calculate_mse(expected, actual):
    """Calculate Mean Squared Error between two tensors"""
    if isinstance(actual, ttnn.Tensor):
        actual = ttnn.to_torch(actual)
    if isinstance(expected, ttnn.Tensor):
        expected = ttnn.to_torch(expected)

    mse = torch.nn.functional.mse_loss(expected.float(), actual.float())
    return mse.item()


def calculate_ulp_error_bf16(expected, actual):
    """
    Calculate ULP (Units in the Last Place) error for bfloat16 tensors.

    ULP error measures how many representable floating-point numbers lie between
    the expected and actual values. For bfloat16, this gives a precise measure
    of numerical accuracy.

    Args:
        expected: Expected tensor values
        actual: Actual tensor values

    Returns:
        float: Maximum ULP error across all elements
    """
    if isinstance(actual, ttnn.Tensor):
        actual = ttnn.to_torch(actual)
    if isinstance(expected, ttnn.Tensor):
        expected = ttnn.to_torch(expected)

    # Convert to bfloat16 if not already
    expected = expected.to(torch.bfloat16)
    actual = actual.to(torch.bfloat16)

    # Handle special cases
    if torch.allclose(expected, actual, rtol=0, atol=0, equal_nan=True):
        return 0.0

    # Convert bfloat16 to uint16 representation for bit manipulation
    expected_bits = expected.view(torch.int16).to(torch.int32)
    actual_bits = actual.view(torch.int16).to(torch.int32)

    # Handle sign differences - if signs differ, we need to handle differently
    expected_sign = expected_bits < 0
    actual_sign = actual_bits < 0

    # For same signs, ULP is just the absolute difference in bit representation
    same_sign = expected_sign == actual_sign

    # For different signs, we need to calculate distance through zero
    # Distance = |expected| + |actual| in ULP terms
    expected_abs_bits = torch.where(expected_sign, -expected_bits, expected_bits)
    actual_abs_bits = torch.where(actual_sign, -actual_bits, actual_bits)

    ulp_diff = torch.where(same_sign, torch.abs(expected_bits - actual_bits), expected_abs_bits + actual_abs_bits)

    # Handle NaN and infinity cases
    expected_finite = torch.isfinite(expected)
    actual_finite = torch.isfinite(actual)
    both_finite = expected_finite & actual_finite

    # If both are finite, use calculated ULP difference
    # If one is not finite, set ULP error to a large value
    ulp_diff = torch.where(both_finite, ulp_diff, torch.tensor(float("inf")))

    # Handle the case where both are the same non-finite value
    both_nan = torch.isnan(expected) & torch.isnan(actual)
    both_posinf = torch.isposinf(expected) & torch.isposinf(actual)
    both_neginf = torch.isneginf(expected) & torch.isneginf(actual)
    same_nonfinite = both_nan | both_posinf | both_neginf

    ulp_diff = torch.where(same_nonfinite, torch.tensor(0.0), ulp_diff)

    # Return maximum ULP error
    max_ulp = torch.max(ulp_diff[torch.isfinite(ulp_diff)])
    return max_ulp.item() if torch.isfinite(max_ulp) else float("inf")


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


def calculate_detailed_ulp_stats(expected, actual):
    """
    Calculate detailed ULP statistics for analysis.

    Returns:
        dict: Dictionary with ULP statistics including max, mean, std, and percentiles
    """
    if isinstance(actual, ttnn.Tensor):
        actual = ttnn.to_torch(actual)
    if isinstance(expected, ttnn.Tensor):
        expected = ttnn.to_torch(expected)

    # Convert to bfloat16 if not already
    expected = expected.to(torch.bfloat16)
    actual = actual.to(torch.bfloat16)

    # Handle special cases
    if torch.allclose(expected, actual, rtol=0, atol=0, equal_nan=True):
        return {
            "max_ulp": 0.0,
            "mean_ulp": 0.0,
            "median_ulp": 0.0,
            "std_ulp": 0.0,
            "p95_ulp": 0.0,
            "p99_ulp": 0.0,
            "perfect_matches": 1.0,
        }

    # Convert bfloat16 to uint16 representation for bit manipulation
    expected_bits = expected.view(torch.int16).to(torch.int32)
    actual_bits = actual.view(torch.int16).to(torch.int32)

    # Handle sign differences
    expected_sign = expected_bits < 0
    actual_sign = actual_bits < 0
    same_sign = expected_sign == actual_sign

    # Calculate ULP differences
    expected_abs_bits = torch.where(expected_sign, -expected_bits, expected_bits)
    actual_abs_bits = torch.where(actual_sign, -actual_bits, actual_bits)

    ulp_diff = torch.where(same_sign, torch.abs(expected_bits - actual_bits), expected_abs_bits + actual_abs_bits)

    # Handle non-finite values
    expected_finite = torch.isfinite(expected)
    actual_finite = torch.isfinite(actual)
    both_finite = expected_finite & actual_finite

    ulp_diff = torch.where(both_finite, ulp_diff, torch.tensor(float("inf")))

    # Handle same non-finite values
    both_nan = torch.isnan(expected) & torch.isnan(actual)
    both_posinf = torch.isposinf(expected) & torch.isposinf(actual)
    both_neginf = torch.isneginf(expected) & torch.isneginf(actual)
    same_nonfinite = both_nan | both_posinf | both_neginf

    ulp_diff = torch.where(same_nonfinite, torch.tensor(0.0), ulp_diff)

    # Calculate statistics only on finite ULP differences
    finite_ulp = ulp_diff[torch.isfinite(ulp_diff)]

    if len(finite_ulp) == 0:
        return {
            "max_ulp": float("inf"),
            "mean_ulp": float("inf"),
            "median_ulp": float("inf"),
            "std_ulp": float("inf"),
            "p95_ulp": float("inf"),
            "p99_ulp": float("inf"),
            "perfect_matches": 0.0,
        }

    finite_ulp_float = finite_ulp.float()
    perfect_matches = (finite_ulp == 0).float().mean().item()

    return {
        "max_ulp": torch.max(finite_ulp).item(),
        "mean_ulp": torch.mean(finite_ulp_float).item(),
        "median_ulp": torch.median(finite_ulp_float).item(),
        "std_ulp": torch.std(finite_ulp_float).item(),
        "p95_ulp": torch.quantile(finite_ulp_float, 0.95).item(),
        "p99_ulp": torch.quantile(finite_ulp_float, 0.99).item(),
        "perfect_matches": perfect_matches,
    }


def setup_ccl_semahpores(mesh_device):
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
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
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

    mesh_tensor = ttnn.to_device(ttnn_input, mesh_device)
    ttnn.visualize_tensor(mesh_tensor)
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
    if norm_type == "layer_norm":
        ttnn_stats = ttnn.layer_norm_pre_all_gather(
            ttnn_input, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16
        )
    elif norm_type == "rms_norm":
        ttnn_stats = ttnn.rms_norm_pre_all_gather(
            ttnn_input, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16
        )
    ccl_semaphore_handles = setup_ccl_semahpores(mesh_device)
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
        )

    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    # Calculate metrics
    mse = calculate_mse(torch_output, ttnn_output_torch)
    ulp_error = calculate_ulp_error_bf16(torch_output, ttnn_output_torch)
    ulp_stats = calculate_detailed_ulp_stats(torch_output, ttnn_output_torch)
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
    logger.info(f"  Allclose Passed: {passes_allclose}")


@pytest.mark.parametrize("h,w", [(1024, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_distributed_1x2(mesh_device, h, w, dtype):
    torch.manual_seed(1234)

    torch_input = torch.randn((1, 1, h, w), dtype=dtype)
    torch_weight = torch.ones((w,), dtype=dtype)

    # TTNN implementation
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # num_devices = mesh_device.get_num_devices()
    # ttnn_weight = ttnn.from_torch(
    #     torch_weight.reshape(num_devices, 1, -1, 32),
    #     dtype=ttnn.bfloat16,
    #     device=mesh_device,
    #     layout=ttnn.ROW_MAJOR_LAYOUT,
    #     mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    # )

    print("Doing pre all gather")
    ttnn_stats = ttnn.layer_norm_pre_all_gather(
        ttnn_input, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16
    )

    ccl_semaphore_handles = setup_ccl_semaphores(mesh_device)
    print("Doing synchronize")
    ttnn.synchronize_device(mesh_device)

    print("Doing all gather")
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

    print("Stats gathered: ", ttnn_stats_gathered)

    eps = 1e-12

    print("Doing post all gather")
    ttnn_output = ttnn.layer_norm_post_all_gather(
        ttnn_input,
        ttnn_stats_gathered,
        epsilon=eps,
        compute_kernel_config=compute_kernel_config,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))

    torch_output = reference_layernorm(torch_input, None, None, eps, False)

    # Calculate metrics
    mse = calculate_mse(torch_output, ttnn_output_torch)
    ulp_error = calculate_ulp_error_bf16(torch_output, ttnn_output_torch)
    ulp_stats = calculate_detailed_ulp_stats(torch_output, ttnn_output_torch)
    pcc_passed, pcc_value = comp_pcc(torch_output, ttnn_output_torch, pcc=0.99)
    max_error = calculate_max_error(torch_output, ttnn_output_torch)

    passes_allclose = torch.allclose(torch_output, ttnn_output_torch)

    # Print results
    logger.info(f"\n=== RMSNorm Comparison Results ===")
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
    logger.info(f"  Allclose Passed: {passes_allclose}")
