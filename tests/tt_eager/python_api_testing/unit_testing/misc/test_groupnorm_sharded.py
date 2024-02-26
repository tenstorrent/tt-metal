# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from loguru import logger
import pytest

import torch
import tt_lib as ttl
from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)

from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, skip_for_wormhole_b0


def find_ceil_divisible_by_32(n):
    return ((n + 31) // 32) * 32


def manual_group_norm(input_tensor, num_groups, eps=1e-2):
    N, C, H, W = input_tensor.shape
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"

    # Reshape into groups
    group_channels = C // num_groups
    input_tensor = input_tensor.view(N, num_groups, group_channels, H, W)

    # Calculate mean and variance
    mean = input_tensor.mean(dim=(2, 3, 4), keepdim=True)
    var = input_tensor.var(dim=(2, 3, 4), keepdim=True)

    # Normalize
    input_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    # Reshape back to original dimensions
    input_tensor = input_tensor.view(N, C, H, W)

    return input_tensor


def ref_groupnorm(x, group_size, eps, **kwargs):
    n_channels = x.shape[1]
    lnorm = torch.nn.GroupNorm(group_size, n_channels, eps, **kwargs)
    return lnorm(x)


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "num_batches",
    [
        2,
    ],
    ids=[
        "b-1",
    ],
)
@pytest.mark.parametrize(
    "C, H, W, num_groups, grid_size, shard_orientation, shard_layout",
    [
        # (320, 64, 64, 32, (12, 6), ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED),
        (320, 32, 32, 32, (1, 8), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED),
        (320, 16, 16, 32, (1, 8), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED),
        (640, 16, 16, 32, (8, 2), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (640, 8, 8, 32, (4, 2), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (1280, 8, 8, 32, (4, 4), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (1280, 4, 4, 32, (1, 4), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (2560, 4, 4, 32, (1, 8), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (2560, 8, 8, 32, (4, 8), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (1920, 8, 8, 32, (2, 8), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (1920, 16, 16, 32, (2, 8), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (1280, 16, 16, 32, (8, 4), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (960, 16, 16, 32, (1, 4), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (960, 32, 32, 32, (4, 4), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (640, 32, 32, 32, (8, 2), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
        (640, 8, 8, 32, (2, 2), ttl.tensor.ShardOrientation.COL_MAJOR, ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        (ttl.tensor.Layout.ROW_MAJOR),
    ],
    ids=[
        "rm",
    ],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2),
    ids=["GN", "GN_G", "GN_GB"],
)
def test_groupnorm_sharded_narrow_channel_per_group(
    test_id, device, layout, num_batches, C, H, W, num_groups, grid_size, shard_orientation, shard_layout
):
    torch.manual_seed(1234)

    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1)
    gamma_beta_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    epsf = 1e-2

    in0_shape = (num_batches, 1, W * H, C)
    pyt_in0_shape = (num_batches, C, H, W)
    pyt_in0 = torch.rand(pyt_in0_shape)

    if shard_layout == ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED:
        if shard_orientation == ttl.tensor.ShardOrientation.COL_MAJOR:
            shard_shape = [int(num_batches * W * H / grid_size[0]), int(C / grid_size[1])]
        else:
            shard_shape = [int(num_batches * W * H / grid_size[1]), int(C / grid_size[0])]
    elif shard_layout == ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_shape = [int(num_batches * W * H / (grid_size[1] * grid_size[0])), int(C)]
        # shard_shape = [128, 320]

    logger.info("shard_shape " + str(shard_shape))

    in0 = pyt_in0.permute(0, 2, 3, 1).contiguous().view(num_batches, 1, W * H, C)

    values_per_chunk = in0_shape[3] // num_groups
    zeros_to_insert = find_ceil_divisible_by_32(in0_shape[3] // num_groups) - in0_shape[3] // num_groups

    logger.info("values_per_chunk " + str(values_per_chunk))
    logger.info("zeros_to_insert " + str(zeros_to_insert))

    # pyt_gamma = torch.ones(in0_shape[3])
    pyt_gamma = torch.rand(in0_shape[3]) * 2 - 1
    pyt_gamma_padded = pyt_gamma.view(-1, values_per_chunk)
    pyt_gamma_padded = torch.nn.functional.pad(pyt_gamma_padded, (0, zeros_to_insert))
    pyt_gamma_padded = pyt_gamma_padded.flatten()
    pyt_gamma_padded = pyt_gamma_padded[: in0_shape[3] + zeros_to_insert * (in0_shape[3] // values_per_chunk)]
    # beta
    # pyt_beta = torch.zeros(in0_shape[3])
    pyt_beta = torch.rand(in0_shape[3]) * 2.0 - 1.1
    pyt_beta_padded = pyt_beta.view(-1, values_per_chunk)
    pyt_beta_padded = torch.nn.functional.pad(pyt_beta_padded, (0, zeros_to_insert))
    pyt_beta_padded = pyt_beta_padded.flatten()
    pyt_beta_padded = pyt_beta_padded[: in0_shape[3] + zeros_to_insert * (in0_shape[3] // values_per_chunk)]
    gamma = pyt_gamma_padded.reshape(1, 1, -1, 32)
    gamma_t = ttl.tensor.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    beta = pyt_beta_padded.reshape(1, 1, -1, 32)
    beta_t = ttl.tensor.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    in0_t = torch2tt_tensor(
        in0, device, tt_memory_config=in0_mem_config, tt_dtype=ttl.tensor.DataType.BFLOAT16, tt_layout=layout
    )
    in0_t_sharded = ttl.tensor.interleaved_to_sharded(in0_t, grid_size, shard_shape, shard_layout, shard_orientation)

    logger.info("start GN")

    program_config = ttl.operations.primary.GroupNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        out_data_format=ttl.tensor.DataType.BFLOAT16,
        inplace=False if layout == ttl.tensor.Layout.ROW_MAJOR else True,
    )

    if test_id == 0:
        logger.info("Running GN")
        out_t = ttl.operations.primary.groupnorm(
            in0_t_sharded,
            num_groups,
            epsf,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 1:
        logger.info("Running GN_G")
        out_t = ttl.operations.primary.groupnorm(
            in0_t_sharded,
            num_groups,
            epsf,
            gamma_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 2:
        logger.info("Running LN_GB")
        out_t = ttl.operations.primary.groupnorm(
            in0_t_sharded,
            num_groups,
            epsf,
            gamma_t,
            beta_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )

    out_t = ttl.tensor.sharded_to_interleaved(out_t, in0_mem_config)
    out = tt2torch_tensor(out_t)

    pyt_groupnorm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=C, eps=epsf)
    manual_out = manual_group_norm(pyt_in0, num_groups, epsf)
    manual_out = manual_out.permute(0, 2, 3, 1).contiguous().view(num_batches, 1, W * H, C)

    if test_id == 1 or test_id == 2:
        pyt_groupnorm.weight.data = pyt_gamma
    if test_id == 2:
        pyt_groupnorm.bias.data = pyt_beta
    pyt_out = pyt_groupnorm(pyt_in0)
    pyt_out = pyt_out.permute(0, 2, 3, 1).contiguous().view(num_batches, 1, W * H, C)

    passing, output = comp_pcc(pyt_out, out)
    logger.info(output)
    assert passing
