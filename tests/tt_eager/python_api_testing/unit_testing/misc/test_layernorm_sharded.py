# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
import pytest
import torch
import math

import tt_lib as ttl

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, skip_for_wormhole_b0, is_grayskull


def rms_norm(x, dim, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean([-i for i in range(1, len(dim) + 1)], keepdim=True) + eps) * gamma + beta


# @skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "out_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1),),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "gamma_beta_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "gb_DRAM",
    ],
)
@pytest.mark.parametrize(
    "gamma_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["FLOAT32", "BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    ids=[
        "add_LN",
        "add_LN_G",
        "add_LN_GB",
        "add_RMSN",
        "add_RMSN_G",
        "add_RMSN_GB",
        "LN",
        "LN_G",
        "LN_GB",
        "RMSN",
        "RMSN_G",
        "RMSN_GB",
    ],
)
@pytest.mark.parametrize("width_padding", [False, True], ids=["no_padding", "padding"])
def test_layernorm_sharded_mix_precision_rm(
    test_id, in_dtype, gamma_dtype, gamma_beta_mem_config, out_mem_config, device, width_padding
):
    if is_grayskull() and in_dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.manual_seed(1234)
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, compute_grid_size.y]
    if grid_size[1] > 8:
        grid_size[1] = 8
    fidelity = ttl.tensor.MathFidelity.HiFi4

    epsf = 1e-2
    batch = grid_size[1]

    width = 128 * grid_size[1]
    if grid_size[1] > 1 and width_padding:
        width = 128 * (grid_size[1] - 1) + 96  # 4 tiles per core, except last one that has 3

    in0_shape = (batch, 1, 32 * grid_size[0], width)
    M = in0_shape[2] * batch
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    shard_shape = [M // grid_size[0], math.ceil(K / grid_size[1] / 32) * 32]
    in0_t_shard = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        shard_shape,
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.ShardOrientation.COL_MAJOR,
    )

    if test_id <= 5:
        in1 = torch.rand(in0_shape) * 2 - 0.8
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
        in1_t_shard = ttl.tensor.interleaved_to_sharded(
            in1_t,
            grid_size,
            shard_shape,
            ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttl.tensor.ShardOrientation.COL_MAJOR,
        )

    if test_id % 3 == 0:
        gamma = torch.ones(in0_shape[3])
        beta = torch.zeros(in0_shape[3])
    if test_id % 3 == 1:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.zeros(in0_shape[3])
    if test_id % 3 == 2:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.rand(in0_shape[3]) * 2.0 - 1.1

    gamma = gamma.reshape(1, 1, -1, 32)
    gamma_t = ttl.tensor.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        gamma_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    beta = beta.reshape(1, 1, -1, 32)
    beta_t = ttl.tensor.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        gamma_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        inplace=True,
    )

    if not is_grayskull():
        compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True
        )

    if test_id == 0:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 1:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 2:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 3:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 4:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 5:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 6:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 7:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 8:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 9:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 10:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )
    if test_id == 11:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config if not is_grayskull() else None,
        )

    ttz = ttl.tensor.sharded_to_interleaved(ttz, in0_mem_config)
    tt_got_back = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().float()

    pt_in = in0 + in1 if test_id <= 5 else in0
    if test_id <= 2 or 6 <= test_id <= 8:
        ref_fn = torch.nn.functional.layer_norm
    else:
        ref_fn = rms_norm
    ref_lnorm = ref_fn(pt_in, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

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
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, ttl.tensor.BufferType.L1),),
    ids=["out_L1"],
)
@pytest.mark.parametrize(
    "gamma_beta_mem_config",
    (ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),),
    ids=[
        "gb_DRAM",
    ],
)
@pytest.mark.parametrize(
    "gamma_dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "in_dtype",
    (
        ttl.tensor.DataType.FLOAT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["FLOAT32", "BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "M, K, subblock_w",
    [
        (64, 8192, 4),
        (64, 8192, 4),  # padding test
        (512, 2048, 1),
    ],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    ids=[
        "add_LN",
        "add_LN_G",
        "add_LN_GB",
        "add_RMSN",
        "add_RMSN_G",
        "add_RMSN_GB",
        "LN",
        "LN_G",
        "LN_GB",
        "RMSN",
        "RMSN_G",
        "RMSN_GB",
    ],
)
def test_layernorm_1d_sharded_mix_precision_rm(
    test_id, M, K, subblock_w, in_dtype, gamma_dtype, gamma_beta_mem_config, out_mem_config, shard_orientation, device
):
    if is_grayskull() and in_dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    torch.manual_seed(1234)
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    device_grid_size = device.compute_with_storage_grid_size()
    if device_grid_size.x >= 8 and device_grid_size.y >= 4:
        if device_grid_size.y >= 8:
            grid_size = (8, 8)
        else:
            grid_size = (8, 4)
    else:
        pytest.skip("Device grid size is too small for this test")

    fidelity = ttl.tensor.MathFidelity.HiFi2

    epsf = 1e-2

    in0_shape = torch.Size([1, 1, M, K])
    M = in0_shape.numel() // in0_shape[3]
    K = in0_shape[3]

    in0 = torch.rand(in0_shape) * 2 - 0.95
    in0_t = torch2tt_tensor(in0, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
    shard_shape = [M, math.ceil(K / (grid_size[0] * grid_size[1]) / 32) * 32]
    in0_t_shard = ttl.tensor.interleaved_to_sharded(
        in0_t,
        grid_size,
        shard_shape,
        ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        shard_orientation,
    )

    if test_id <= 5:
        in1 = torch.rand(in0_shape) * 2 - 0.8
        in1_t = torch2tt_tensor(in1, device, tt_memory_config=in0_mem_config, tt_dtype=in_dtype)
        in1_t_shard = ttl.tensor.interleaved_to_sharded(
            in1_t,
            grid_size,
            shard_shape,
            ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
            shard_orientation,
        )

    if test_id % 3 == 0:
        gamma = torch.ones(in0_shape[3])
        beta = torch.zeros(in0_shape[3])
    if test_id % 3 == 1:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.zeros(in0_shape[3])
    if test_id % 3 == 2:
        gamma = torch.rand(in0_shape[3]) * 2 - 1
        beta = torch.rand(in0_shape[3]) * 2.0 - 1.1

    gamma = gamma.reshape(1, 1, -1, 32)
    gamma_t = ttl.tensor.Tensor(
        gamma.reshape(-1).tolist(),
        gamma.shape,
        gamma_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    beta = beta.reshape(1, 1, -1, 32)
    beta_t = ttl.tensor.Tensor(
        beta.reshape(-1).tolist(),
        beta.shape,
        gamma_dtype,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device, gamma_beta_mem_config)

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=subblock_w,
        block_h=M // 32,
        block_w=shard_shape[1] // 32,
        inplace=True,
    )
    compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    if test_id == 0:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 1:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 2:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 3:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 4:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 5:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            residual_input_tensor=in1_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 6:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 7:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 8:
        ttz = ttnn.layer_norm(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 9:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            epsilon=epsf,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 10:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    if test_id == 11:
        ttz = ttnn.rms_norm(
            in0_t_shard,
            epsilon=epsf,
            weight=gamma_t,
            bias=beta_t,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    ttz = ttl.tensor.sharded_to_interleaved(ttz, in0_mem_config)
    tt_got_back = ttz.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch().float()

    pt_in = in0 + in1 if test_id <= 5 else in0
    if test_id <= 2 or 6 <= test_id <= 8:
        ref_fn = torch.nn.functional.layer_norm
    else:
        ref_fn = rms_norm
    ref_lnorm = ref_fn(pt_in, in0.shape[-1:], gamma.flatten(), beta.flatten(), epsf)

    passing, output = comp_pcc(tt_got_back, ref_lnorm, 0.999)
    logger.info(output)
    assert passing
