# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import pytest
import torch

import tt_lib as ttl

from tt_lib.utils import (
    pad_weight,
    tilize_to_list,
    untilize,
    is_close,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
from models.utility_functions import is_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1),),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "gamma_beta_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=[
        "gb_DRAM",
        "gb_L1",
    ],
)
@pytest.mark.parametrize(
    "out_dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "cb_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2),
    ids=[
        "LN",
        "LN_G",
        "LN_GB",
    ],
)
def test_layernorm_sharded_rm(test_id, in_dtype, out_dtype, cb_dtype, gamma_beta_mem_config, out_mem_config, device):
    torch.manual_seed(1234)
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    if grid_size[1] > 8:
        grid_size[1] = 8
    fidelity = ttl.tensor.MathFidelity.HiFi4

    epsf = 1e-2
    batch = 12

    in0_shape = (batch, 1, 32 * grid_size[0], 1024)
    M = in0_shape[2] * batch
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in0_t_shard = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    in1 = torch.zeros(in0_shape)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    if test_id == 0:
        gamma = torch.ones(in0_shape[3])
        beta = torch.zeros(in0_shape[3])
    if test_id == 1:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.zeros(in0_shape[3])
    if test_id == 2:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.rand(in0_shape[3]) * 2.0 - 1.1

    gamma = gamma.reshape(1, 1, -1, 32)
    gamma_t = ttl.tensor.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        cb_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    beta = beta.reshape(1, 1, -1, 32)
    beta_t = ttl.tensor.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        cb_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    program_config = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        math_fidelity=fidelity,
        im_data_format=cb_dtype,
        out_data_format=out_dtype,
        inplace=in_dtype == out_dtype,
    )

    if test_id == 0:
        logger.info("Running LN")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 1:
        logger.info("Running LN_G")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            gamma_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 2:
        logger.info("Running LN_GB")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            gamma_t,
            beta_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )

    logger.info("Done")

    ttz = ttl.tensor.sharded_to_interleaved(ttz, in0_mem_config)
    t2_data = ttz.cpu().to_torch().float()
    tt_got_back = torch.Tensor(t2_data).reshape(in0_shape)
    tt_got_back = untilize(tt_got_back)

    ref_lnorm = torch.nn.functional.layer_norm(in0 + in1, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

    passing, output = comp_pcc(tt_got_back, ref_lnorm, 0.999)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1),),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "gamma_beta_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=[
        "gb_DRAM",
        "gb_L1",
    ],
)
@pytest.mark.parametrize(
    "out_dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "cb_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2),
    ids=[
        "add_LN",
        "add_LN_G",
        "add_LN_GB",
    ],
)
def test_layernorm_sharded_mix_precision_rm(
    test_id, in_dtype, out_dtype, cb_dtype, gamma_beta_mem_config, out_mem_config, device
):
    torch.manual_seed(1234)
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    if grid_size[1] > 8:
        grid_size[1] = 8
    fidelity = ttl.tensor.MathFidelity.HiFi4

    epsf = 1e-2
    batch = 12

    in0_shape = (batch, 1, 32 * grid_size[0], 1024)
    M = in0_shape[2] * batch
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in0_t_shard = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    in1 = torch.rand(in0_shape) * 2 - 0.8
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in1_t_shard = ttl.tensor.interleaved_to_sharded(
        in1_t,
        grid_size,
        [M // grid_size[0], K // grid_size[1]],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    if test_id == 0:
        gamma = torch.ones(in0_shape[3])
        beta = torch.zeros(in0_shape[3])
    if test_id == 1:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.zeros(in0_shape[3])
    if test_id == 2:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.rand(in0_shape[3]) * 2.0 - 1.1

    gamma = gamma.reshape(1, 1, -1, 32)
    gamma_t = ttl.tensor.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        cb_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    beta = beta.reshape(1, 1, -1, 32)
    beta_t = ttl.tensor.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        cb_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    program_config = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        math_fidelity=fidelity,
        im_data_format=cb_dtype,
        out_data_format=out_dtype,
        inplace=in_dtype == out_dtype,
    )

    if test_id == 0:
        logger.info("Running add_LN")
        ttz = ttl.operations.primary.add_layernorm(
            in0_t_shard,
            in1_t_shard,
            epsf,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 1:
        logger.info("Running add_LN_G")
        ttz = ttl.operations.primary.add_layernorm(
            in0_t_shard,
            in1_t_shard,
            epsf,
            gamma_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 2:
        logger.info("Running add_LN_GB")
        ttz = ttl.operations.primary.add_layernorm(
            in0_t_shard,
            in1_t_shard,
            epsf,
            gamma_t,
            beta_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )

    logger.info("Done")

    ttz = ttl.tensor.sharded_to_interleaved(ttz, in0_mem_config)
    t2_data = ttz.cpu().to_torch().float()
    tt_got_back = torch.Tensor(t2_data).reshape(in0_shape)
    tt_got_back = untilize(tt_got_back)

    ref_lnorm = torch.nn.functional.layer_norm(in0 + in1, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

    passing, output = comp_pcc(tt_got_back, ref_lnorm, 0.999)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "shard_orientation",
    (ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR),
    ids=["RM", "CM"],
)
@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1),),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "gamma_beta_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=[
        "gb_DRAM",
        "gb_L1",
    ],
)
@pytest.mark.parametrize(
    "out_dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "cb_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2, 3, 4, 5),
    ids=[
        "add_LN",
        "add_LN_G",
        "add_LN_GB",
        "LN",
        "LN_G",
        "LN_GB",
    ],
)
def test_layernorm_1d_sharded_mix_precision_rm(
    test_id, in_dtype, out_dtype, cb_dtype, gamma_beta_mem_config, out_mem_config, shard_orientation, device
):
    torch.manual_seed(1234)
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    grid_size = (8, 8)
    fidelity = ttl.tensor.MathFidelity.HiFi4

    epsf = 1e-2

    in0_shape = torch.Size([1, 1, 32, 8192])
    M = in0_shape.numel() // in0_shape[3]
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    in0_t_shard = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        [M, K // (grid_size[0] * grid_size[1])],
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        shard_orientation,
    )

    if test_id <= 2:
        in1 = torch.rand(in0_shape) * 2 - 0.8
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
        in1_t_shard = ttl.tensor.interleaved_to_sharded(
            in1_t,
            grid_size,
            [M, K // (grid_size[0] * grid_size[1])],
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            shard_orientation,
        )

    if test_id == 0 or test_id == 3:
        gamma = torch.ones(in0_shape[3])
        beta = torch.zeros(in0_shape[3])
    if test_id == 1 or test_id == 4:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.zeros(in0_shape[3])
    if test_id == 2 or test_id == 5:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.rand(in0_shape[3]) * 2.0 - 1.1

    gamma = gamma.reshape(1, 1, -1, 32)
    gamma_t = ttl.tensor.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        cb_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    beta = beta.reshape(1, 1, -1, 32)
    beta_t = ttl.tensor.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        cb_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    program_config = ttl.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=M // 32,
        block_w=K // (grid_size[0] * grid_size[1]) // 32,
        math_fidelity=fidelity,
        im_data_format=cb_dtype,
        out_data_format=out_dtype,
        inplace=in_dtype == out_dtype,
    )

    if test_id == 0:
        logger.info("Running add_LN")
        ttz = ttl.operations.primary.add_layernorm(
            in0_t_shard,
            in1_t_shard,
            epsf,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 1:
        logger.info("Running add_LN_G")
        ttz = ttl.operations.primary.add_layernorm(
            in0_t_shard,
            in1_t_shard,
            epsf,
            gamma_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 2:
        logger.info("Running add_LN_GB")
        ttz = ttl.operations.primary.add_layernorm(
            in0_t_shard,
            in1_t_shard,
            epsf,
            gamma_t,
            beta_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 3:
        logger.info("Running LN")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 4:
        logger.info("Running LN_G")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            gamma_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 5:
        logger.info("Running LN_GB")
        ttz = ttl.operations.primary.layernorm(
            in0_t_shard,
            epsf,
            gamma_t,
            beta_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )

    logger.info("Done")

    ttz = ttl.tensor.sharded_to_interleaved(ttz, in0_mem_config)
    t2_data = ttz.cpu().to_torch().float()
    tt_got_back = torch.Tensor(t2_data).reshape(in0_shape)
    tt_got_back = untilize(tt_got_back)

    pt_in = in0 + in1 if test_id <= 2 else in0
    ref_lnorm = torch.nn.functional.layer_norm(pt_in, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

    passing, output = comp_pcc(tt_got_back, ref_lnorm, 0.999)
    logger.info(output)
    assert passing
