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
from models.utility_functions import is_wormhole_b0, skip_for_wormhole_b0, skip_for_grayskull
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


def rmsnorm(x, dim, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean([-i for i in range(1, len(dim) + 1)], keepdim=True) + eps) * gamma + beta


# only use certain tests for CI to reduce run time
# grid_sizes = [(i, j) for i in range(1, 13) for j in range(1, 9)] # (1,1) to (12,8)
grid_sizes = [[1, 1], [1, 8], [12, 1], [12, 8]]
# seq_lens = [int(i*32) for i in range(1, 25)] # 32 - 768
seq_lens = [32, 256, 384]
per_core_ks = [32, 64, 128]


@skip_for_grayskull("disable due to watcher error, see issue #5797")
@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "grid_size",
    grid_sizes,
    ids=[f"grid_size_{x}_{y}" for x, y in grid_sizes],
)
@pytest.mark.parametrize(
    "seq_len",
    seq_lens,
    ids=[f"seq_len_{x}" for x in seq_lens],
)
@pytest.mark.parametrize(
    "per_core_k",
    per_core_ks,
    ids=[f"per_core_k_{x}" for x in per_core_ks],
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
@pytest.mark.parametrize(
    "tt_lib_fn, ref_fn",
    [(ttl.operations.primary.layernorm, torch.nn.functional.layer_norm), (ttl.operations.primary.rmsnorm, rmsnorm)],
    ids=["LayerNorm", "RMSNorm"],
)
def test_layernorm_sharded_rm(test_id, device, grid_size, seq_len, per_core_k, tt_lib_fn, ref_fn):
    torch.manual_seed(1234)

    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1)
    gamma_beta_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    in_dtype = ttl.tensor.DataType.BFLOAT16
    cb_dtype = ttl.tensor.DataType.BFLOAT16
    out_dtype = ttl.tensor.DataType.BFLOAT16
    fidelity = ttl.tensor.MathFidelity.HiFi4

    epsf = 1e-2
    batch = grid_size[0]

    in0_shape = (batch, 1, seq_len, per_core_k * grid_size[1])
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
        subblock_w=per_core_k // 32,
        block_h=seq_len // 32,
        block_w=per_core_k // 32,
        inplace=True,
    )

    if test_id == 0:
        logger.info("Running LN")
        ttz = tt_lib_fn(
            in0_t_shard,
            epsf,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 1:
        logger.info("Running LN_G")
        ttz = tt_lib_fn(
            in0_t_shard,
            epsf,
            gamma_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 2:
        logger.info("Running LN_GB")
        ttz = tt_lib_fn(
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

    ref_lnorm = ref_fn(in0 + in1, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

    passing, output = comp_pcc(tt_got_back, ref_lnorm, 0.999)
    logger.info(output)
    assert passing


# only use certain tests for CI to reduce run time
grid_sizes = [[1, 1], [1, 8], [12, 1], [12, 8]]
seq_lens = [32, 256, 384]
per_core_ks = [32, 64, 128]


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "grid_size",
    grid_sizes,
    ids=[f"grid_size_{x}_{y}" for x, y in grid_sizes],
)
@pytest.mark.parametrize(
    "seq_len",
    seq_lens,
    ids=[f"seq_len_{x}" for x in seq_lens],
)
@pytest.mark.parametrize(
    "per_core_k",
    per_core_ks,
    ids=[f"per_core_k_{x}" for x in per_core_ks],
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
@pytest.mark.parametrize(
    "tt_lib_fn, ref_fn",
    [
        (ttl.operations.primary.add_layernorm, torch.nn.functional.layer_norm),
        (ttl.operations.primary.add_rmsnorm, rmsnorm),
    ],
    ids=["LayerNorm", "RMSNorm"],
)
def test_layernorm_sharded_mix_precision_rm(test_id, device, grid_size, seq_len, per_core_k, tt_lib_fn, ref_fn):
    torch.manual_seed(1234)

    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1)
    gamma_beta_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    in_dtype = ttl.tensor.DataType.BFLOAT16
    cb_dtype = ttl.tensor.DataType.BFLOAT16
    out_dtype = ttl.tensor.DataType.BFLOAT16
    fidelity = ttl.tensor.MathFidelity.HiFi4

    epsf = 1e-2
    batch = grid_size[0]

    in0_shape = (batch, 1, seq_len, per_core_k * grid_size[1])
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
        subblock_w=per_core_k // 32,
        block_h=seq_len // 32,
        block_w=per_core_k // 32,
        inplace=True,
    )

    if test_id == 0:
        logger.info("Running add_LN")
        ttz = tt_lib_fn(
            in0_t_shard,
            in1_t_shard,
            epsf,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 1:
        logger.info("Running add_LN_G")
        ttz = tt_lib_fn(
            in0_t_shard,
            in1_t_shard,
            epsf,
            gamma_t,
            output_mem_config=out_mem_config,
            program_config=program_config,
        )
    if test_id == 2:
        logger.info("Running add_LN_GB")
        ttz = tt_lib_fn(
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

    ref_lnorm = ref_fn(in0 + in1, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

    passing, output = comp_pcc(tt_got_back, ref_lnorm, 0.999)
    logger.info(output)
    assert passing
