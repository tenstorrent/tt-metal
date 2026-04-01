# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
import torch
import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

TEST_PADDING_VALUE = -42


def run_eltwise_softmax_in_place_tests(input_shape, dtype, dlayout, in_mem_config, data_seed, device):
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100)
    x_ref = x.detach().clone()

    # get ref result
    ref_value = torch.softmax(x_ref, -1)

    # Same path as ttnn_ops.eltwise_softmax_in_place / setup_ttnn_tensor, plus explicit padding
    t0 = ttnn.from_torch(
        x,
        dtype=dtype,
        layout=dlayout,
        device=device if in_mem_config is not None else None,
        memory_config=in_mem_config,
        pad_value=TEST_PADDING_VALUE,
    )
    if in_mem_config is not None:
        t0 = ttnn.fill_implicit_tile_padding(t0, TEST_PADDING_VALUE)

    t1 = ttnn.softmax(t0, -1, memory_config=None)
    tt_result = ttnn.to_torch(t1)

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


test_sweep_args = [
    (
        (1, 9, 32, 32),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        38346,
    ),
    (
        (4, 7, 32, 96),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        17155532,
    ),
    (
        (4, 7, 32, 96),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        16305027,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_softmax_in_place_test(input_shape, dtype, dlayout, in_mem_config, data_seed, device):
    run_eltwise_softmax_in_place_tests(input_shape, dtype, dlayout, in_mem_config, data_seed, device)
