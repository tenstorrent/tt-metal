# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn

import tt_lib as ttl
import pytest
from loguru import logger

from models.utility_functions import tt2torch_tensor, comp_pcc


def run_ssm_eltwise_mul_test(batch_size, in0_W, in1_W, dtype, in0_mem_config, in1_mem_config, out_mem_config, device):
    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()
    batch_size = batch_size
    hidden_size = 5120
    latent_size = 32

    B_shape = [1, 1, batch_size, in0_W]
    X_shape = [1, 1, batch_size, in1_W]
    B = torch.randn(B_shape)
    X = torch.randn(X_shape)

    tt_input_tensor_B = ttl.tensor.Tensor(B, dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)
    tt_input_tensor_X = ttl.tensor.Tensor(X, dtype).to(ttl.tensor.Layout.TILE).to(device, in1_mem_config)

    tt_out = ttnn.experimental.repeat_and_interleave_eltwise_mul(
        tt_input_tensor_B, tt_input_tensor_X, memory_config=out_mem_config, dtype=dtype
    )

    assert list(tt_out.get_legacy_shape()) == [1, 1, batch_size, latent_size * hidden_size]

    out = tt2torch_tensor(tt_out)

    # Compute reference on pytorch
    if in0_W == latent_size and in1_W == hidden_size:
        ref_out = B.repeat(1, 1, 1, hidden_size) * X.repeat_interleave(latent_size, dim=-1)
    elif in0_W == latent_size * hidden_size and in1_W == hidden_size:
        ref_out = B * X.repeat_interleave(latent_size, dim=-1)
    elif in0_W == latent_size and in1_W == latent_size * hidden_size:
        ref_out = B.repeat(1, 1, 1, hidden_size) * X
    else:
        raise Exception("Input shapes invalid, use eltwise_mul for same input shapes,", in0_W, in1_W)

    passing_pcc, output_pcc = comp_pcc(out, ref_out, 0.9995)
    logger.debug(f"Out passing={passing_pcc}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing_pcc


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
)
@pytest.mark.parametrize(
    "in1_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.BFLOAT8_B),
)
@pytest.mark.parametrize(
    "in0_W, in1_W",
    (
        (32, 5120),
        (32 * 5120, 5120),
        (32, 32 * 5120),
    ),
)
@pytest.mark.parametrize(
    "batch",
    (32, 64),
)
def test_ssm_eltwise_mul(batch, in0_W, in1_W, dtype, in0_mem_config, in1_mem_config, out_mem_config, device):
    run_ssm_eltwise_mul_test(batch, in0_W, in1_W, dtype, in0_mem_config, in1_mem_config, out_mem_config, device)


def test_ssm_eltwise_mul_with_program_cache(device, use_program_cache):
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    dtype = ttl.tensor.DataType.BFLOAT16

    for _ in range(2):
        batch, in0_W, in1_W = 64, 32, 5120
        run_ssm_eltwise_mul_test(batch, in0_W, in1_W, dtype, mem_config, mem_config, mem_config, device)
        batch, in0_W, in1_W = 64, 32 * 5120, 5120
        run_ssm_eltwise_mul_test(batch, in0_W, in1_W, dtype, mem_config, mem_config, mem_config, device)
        batch, in0_W, in1_W = 64, 32, 32 * 5120
        run_ssm_eltwise_mul_test(batch, in0_W, in1_W, dtype, mem_config, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    assert device.num_program_cache_entries() == 3
