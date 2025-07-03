# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from models.utility_functions import tt2torch_tensor, get_devices_for_t3000, skip_for_grayskull

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
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
    # Run layernorm part 1
    if is_rmsnorm:
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=stats_dtype)
    else:
        tt_stats = ttnn.layer_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=stats_dtype)

    # AllGather stats
    tt_stats = ttnn.all_gather(tt_stats, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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


@skip_for_grayskull("Requires eth connected devices to run")
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


@skip_for_grayskull("Requires eth connected devices to run")
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
