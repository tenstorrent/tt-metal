# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from models.utility_functions import tt2torch_tensor, torch2tt_tensor, skip_for_grayskull, skip_for_blackhole

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc


def reference_layernorm(x, gamma, beta, epsilon, is_rmsnorm):
    if is_rmsnorm:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) * gamma
    else:
        return torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, epsilon)


def run_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device, fp32_enabled=False):
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_enabled,
        packer_l1_acc=False,
    )

    torch.manual_seed(1234)
    tile_cols_per_device = 1 if is_rmsnorm else 2  # layernorm has 2 stats to distribute

    canon_inp = torch.randn(inp_shape) * 4 - 1
    gamma = torch.rand(inp_shape[-1]) * 2 - 1
    beta = torch.rand(inp_shape[-1]) * 2 - 1
    gamma_chunked = gamma.chunk(n_devices, dim=-1)
    beta_chunked = beta.chunk(n_devices, dim=-1)
    # Get per-chunk mean and mean(x^2)
    inp_chunked = canon_inp.chunk(n_devices, dim=-1)
    mean = [x.sum(dim=-1, keepdim=True) for x in inp_chunked]
    meanx2 = [x.pow(2).sum(dim=-1, keepdim=True) for x in inp_chunked]

    stats_tiles = torch.zeros(inp_shape[:-1] + (32 * n_devices * tile_cols_per_device,))
    for idx, (m, mm) in enumerate(zip(mean, meanx2)):
        mm_idx = idx * tile_cols_per_device * 32
        stats_tiles[..., mm_idx : mm_idx + 1] = mm

        if not is_rmsnorm:
            m_idx = mm_idx + 32  # next tile is m
            stats_tiles[..., m_idx : m_idx + 1] = m

    epsilon = 1e-5
    # reference layernorm
    ref_out = reference_layernorm(canon_inp, gamma, beta, epsilon, is_rmsnorm)
    ref_chunks = ref_out.chunk(n_devices, dim=-1)

    dram_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    all_pass = True
    # layernorm post all gather
    for d in range(n_devices):
        tt_inp = torch2tt_tensor(
            inp_chunked[d],
            tt_dtype=dtype,
            tt_device=device,
            tt_layout=ttnn.TILE_LAYOUT,
            tt_memory_config=dram_memcfg,
        )
        tt_gamma = torch2tt_tensor(
            gamma_chunked[d].reshape(1, 1, -1, 32),
            tt_dtype=ttnn.bfloat16,
            tt_device=device,
            tt_layout=ttnn.ROW_MAJOR_LAYOUT,
            tt_memory_config=dram_memcfg,
        )
        tt_beta = torch2tt_tensor(
            beta_chunked[d].reshape(1, 1, -1, 32),
            tt_dtype=ttnn.bfloat16,
            tt_device=device,
            tt_layout=ttnn.ROW_MAJOR_LAYOUT,
            tt_memory_config=dram_memcfg,
        )
        tt_stats = torch2tt_tensor(
            stats_tiles,
            tt_dtype=ttnn.bfloat16,
            tt_device=device,
            tt_layout=ttnn.TILE_LAYOUT,
            tt_memory_config=dram_memcfg,
        )

        if is_rmsnorm:
            tt_lnp2_out = ttnn.rms_norm_post_all_gather(
                tt_inp, tt_stats, epsilon=epsilon, weight=tt_gamma, compute_kernel_config=kernel_config
            )
        else:
            tt_lnp2_out = ttnn.layer_norm_post_all_gather(
                tt_inp, tt_stats, epsilon=epsilon, weight=tt_gamma, bias=tt_beta, compute_kernel_config=kernel_config
            )

        tt_lnp2_out_cpu = tt2torch_tensor(tt_lnp2_out)
        passing, output_str = comp_allclose(ref_chunks[d], tt_lnp2_out_cpu, rtol=1e-1, atol=1e-01)
        logger.debug(f"layernorm vs tt={output_str}")
        all_pass = all_pass and passing

    assert all_pass


@skip_for_blackhole("Mismatching on BH, see #12349")
@skip_for_grayskull("Requires wormhole")
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 2048, 8192),
        (1, 1, 128, 8192),
        (2, 1, 128, 8192),
    ],
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
@pytest.mark.parametrize(
    "fp32_enabled",
    [True, False],
    ids=["fp32_enabled", "fp32_disabled"],
)
def test_layernorm_part_2_with_program_cache(
    inp_shape, n_devices, is_rmsnorm, dtype, fp32_enabled, device, use_program_cache
):
    run_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device, fp32_enabled)


@skip_for_blackhole("Mismatching on BH, see #12349")
@skip_for_grayskull("Requires wormhole")
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16],
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 2048, 8192),
    ],
)
@pytest.mark.parametrize(
    "n_devices",
    [8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
def test_layernorm_part_2_with_program_cache2(inp_shape, n_devices, is_rmsnorm, dtype, device, use_program_cache):
    dummy_tensors = []

    dram_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    for i in range(2):
        if i > 0:
            dummy_tensors.append(
                torch2tt_tensor(
                    torch.randn(inp_shape),
                    tt_dtype=dtype,
                    tt_device=device,
                    tt_layout=ttnn.TILE_LAYOUT,
                    tt_memory_config=dram_memcfg,
                )
            )
        run_layernorm_part_2(inp_shape, n_devices, is_rmsnorm, dtype, device)

    assert device.num_program_cache_entries() == 1, "Program cache should have only one entry" + str(
        device.num_program_cache_entries()
    )
