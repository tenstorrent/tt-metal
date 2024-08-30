# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from models.utility_functions import tt2torch_tensor, get_devices_for_t3000, skip_for_grayskull

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def reference_layernorm(x, gamma, beta, epsilon, is_rmsnorm):
    if is_rmsnorm:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) * gamma
    else:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, epsilon)


def tt_distributed_layernorm(inp, gamma, beta, epsilon, is_rmsnorm, compute_kernel_config, stats_dtype):
    n_devices = len(inp)

    # Run layernorm part 1
    tt_stats = []
    for d in range(n_devices):
        if is_rmsnorm:
            tt_stats.append(
                ttnn.rms_norm_pre_all_gather(inp[d], compute_kernel_config=compute_kernel_config, dtype=stats_dtype)
            )
        else:
            tt_stats.append(
                ttnn.layer_norm_pre_all_gather(inp[d], compute_kernel_config=compute_kernel_config, dtype=stats_dtype)
            )

    tt_stats = ttnn.aggregate_as_tensor(tt_stats)
    # AllGather stats
    tt_stats = ttnn.all_gather(tt_stats, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Run layernorm part 2
    tt_stats = ttnn.get_device_tensors(tt_stats)
    tt_out = []
    for d in range(n_devices):
        if is_rmsnorm:
            tt_out.append(
                ttnn.rms_norm_post_all_gather(
                    inp[d], tt_stats[d], epsilon=epsilon, weight=gamma[d], compute_kernel_config=compute_kernel_config
                )
            )
        else:
            tt_out.append(
                ttnn.layer_norm_post_all_gather(
                    inp[d],
                    tt_stats[d],
                    epsilon=epsilon,
                    weight=gamma[d],
                    bias=beta[d],
                    compute_kernel_config=compute_kernel_config,
                )
            )
        tt_stats[d].deallocate(True)
    return tt_out


def run_distributed_layernorm(
    inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, devices, fp32_enabled=False, iterations=1
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

    # reference impl
    out_torch = reference_layernorm(canon_inp, gamma, beta, epsilon, is_rmsnorm)

    tt_inp = []
    for d in range(n_devices):
        tt_inp.append(
            ttnn.as_tensor(
                inp_chunked[d],
                dtype=dtype,
                device=devices[d],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    tt_gamma = []
    for d in range(n_devices):
        tt_gamma.append(
            ttnn.as_tensor(
                gamma_chunked[d].reshape(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                device=devices[d],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    tt_beta = []
    for d in range(n_devices):
        tt_beta.append(
            ttnn.as_tensor(
                beta_chunked[d].reshape(1, 1, -1, 32),
                dtype=ttnn.bfloat16,
                device=devices[d],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    for i in range(iterations):
        tt_out = tt_distributed_layernorm(
            tt_inp, tt_gamma, tt_beta, epsilon, is_rmsnorm, compute_kernel_config, stats_dtype
        )
        tt_output_host = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_out], -1)

    passing, output_str = comp_allclose(tt_output_host, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"torch vs tt distributed layernorm = {output_str}")

    assert passing


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "iterations",
    [2],
    ids=["loops2"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16_in", "BFLOAT8_B_in"],
)
@pytest.mark.parametrize(
    "stats_dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16_stats", "BFLOAT8_B_stats"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 2048, 8192),
        (1, 1, 128, 8192),
        (2, 1, 128, 8192),
    ],
    ids=["inp_shape0", "inp_shape1", "inp_shape2"],
)
@pytest.mark.parametrize(
    "n_devices",
    [4, 8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
def test_distributed_layernorm_with_program_cache(
    inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, iterations, all_devices, use_program_cache
):
    if len(all_devices) != 8:
        pytest.skip("Not T3000!")

    devices = get_devices_for_t3000(all_devices, n_devices)

    run_distributed_layernorm(inp_shape, n_devices, is_rmsnorm, dtype, stats_dtype, devices, iterations=iterations)

    for d in range(len(devices)):
        assert devices[d].num_program_cache_entries() == 3, "Program cache should have only 3 entries, but has " + str(
            devices[d].num_program_cache_entries()
        )
