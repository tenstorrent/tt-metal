# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

import tt_lib as ttl
import pytest
from loguru import logger

from models.utility_functions import tt2torch_tensor, comp_pcc


def run_ssm_eltwise_mul_test(H, W, dtype, in0_mem_config, in1_mem_config, out_mem_config, device):
    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()

    B_shape = [1, 1, H, 32]
    X_shape = [1, 1, H, W]
    B = torch.randn(B_shape)
    X = torch.randn(X_shape)

    tt_input_tensor_B = ttl.tensor.Tensor(B, dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)
    tt_input_tensor_X = ttl.tensor.Tensor(X, dtype).to(ttl.tensor.Layout.TILE).to(device, in1_mem_config)

    tt_out = ttl.operations.primary.transformers.ssm_eltwise_mul(
        tt_input_tensor_B, tt_input_tensor_X, output_mem_config=out_mem_config, output_dtype=dtype
    )

    assert list(tt_out.get_legacy_shape()) == [1, 1, H, 32 * W]

    out = tt2torch_tensor(tt_out)

    # Compute reference on pytorch
    ref_out = B.repeat(1, 1, 1, W) * X.repeat_interleave(32, dim=-1)

    passing_pcc, output_pcc = comp_pcc(out, ref_out, 0.9999)
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
    (ttl.tensor.DataType.BFLOAT16,),
)
@pytest.mark.parametrize(
    "H, W",
    (
        (32, 32),
        (32, 5120),
    ),
)
def test_ssm_eltwise_mul(H, W, dtype, in0_mem_config, in1_mem_config, out_mem_config, device):
    run_ssm_eltwise_mul_test(H, W, dtype, in0_mem_config, in1_mem_config, out_mem_config, device)


def test_ssm_eltwise_mul_with_program_cache(device, use_program_cache):
    H, W = 32, 5120
    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    dtype = ttl.tensor.DataType.BFLOAT16

    for _ in range(2):
        run_ssm_eltwise_mul_test(H, W, dtype, mem_config, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttl.tensor.Tensor(py_dummy_tensor, dtype).to(ttl.tensor.Layout.TILE).to(device, mem_config)

    assert device.num_program_cache_entries() == 1
