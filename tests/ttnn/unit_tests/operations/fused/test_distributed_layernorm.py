# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from models.common.utility_functions import tt2torch_tensor

from loguru import logger
from tests.ttnn.utils_for_testing import assert_numeric_metrics, check_with_pcc
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
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
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

    assert_numeric_metrics(
        out_torch,
        tt_output_host,
        rtol=10.313,
        atol=0.207,
        frobenius_threshold=0.028,
        pcc_threshold=0.983,
    )


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
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_distributed_layernorm_with_program_cache(
    inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, iterations, mesh_device
):
    run_test_distributed_layernorm_with_program_cache_and_checks(
        inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, mesh_device, iterations=iterations
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


# Regression for #54697: the pre-all-gather x^2 buffer is sized per row, so a wide row with
# fp32 dest accumulation is what pushes the program past L1. The parametrizations above cannot
# catch it -- they shard 8192 across 4-8 devices, leaving only 1024 per chip. This runs the
# widest per-chip row the models actually use (Llama-3.3-70B on a 2-device mesh: 8192/2) on a
# SINGLE device, with fp32_dest_acc_en on, so it fails at allocation time if the buffer sizing
# regresses again.
@pytest.mark.parametrize("width", [4096], ids=["w4096"])
@pytest.mark.parametrize("is_rmsnorm", rms_norm_parametrizations, ids=rms_norm_parametrization_ids)
def test_pre_all_gather_wide_row_fp32_dest_acc_fits_l1(device, width, is_rmsnorm):
    torch.manual_seed(1234)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    inp = torch.randn((1, 1, 32, width), dtype=torch.bfloat16)
    tt_inp = ttnn.from_torch(
        inp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if is_rmsnorm:
        tt_stats = ttnn.rms_norm_pre_all_gather(
            tt_inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16
        )
    else:
        tt_stats = ttnn.layer_norm_pre_all_gather(
            tt_inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16
        )

    # Column 0 of the stats tile is sum(x^2) for both variants.
    got = ttnn.to_torch(tt_stats).to(torch.float32)[..., 0:1]
    expected = (inp.to(torch.float32) ** 2).sum(dim=-1, keepdim=True)
    passing, pcc = check_with_pcc(expected, got, 0.99)
    assert passing, f"sum(x^2) mismatch at width {width}: {pcc}"
