# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_pt_tt
import tt_lib as ttl
from models.utility_functions import comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE

    torch_input = torch.randint(-100, 100, input_shape, dtype=cpu_dtype)
    torch_output = torch.randint(-100, 100, output_shape, dtype=cpu_dtype)

    tt_input = ttl.tensor.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttl.tensor.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "input_shape",
    (
        ([2, 3, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 7 - 1]),
        ([9, 16, TILE_HEIGHT * 13 - 1, TILE_WIDTH * 19 - 1]),
        ([4, 3, TILE_HEIGHT * 3 - 1, TILE_WIDTH * 11 - 1]),
        ([1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1]),
        ([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1]),
        ([8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "2, 3, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 7 - 1",
        "9, 16, TILE_HEIGHT * 13 - 1, TILE_WIDTH * 19 - 1",
        "4, 3, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 11 - 1",
        "1, 1, TILE_HEIGHT-1,TILE_WIDTH - 1",
        "4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1",
        "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1",
        "8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    (
        [
            0,
        ],
        [
            1,
        ],
    ),
    ids=["0", "1"],
)
# Support for dim 2,3 in composite_ops
def test_prod_dims(input_shape, dims, device):
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device)

    torch_output = torch.prod(torch_input, dims[0], True)

    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    tt_output_cpu = (
        ttl.operations.primary.prod_nc(tt_input, tt_output, dims=dims)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing


mem_configs = [
    ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
    ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
]


@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
@pytest.mark.parametrize(
    "input_shapes",
    (
        ([1, 1, 32, 32]),
        ([2, 2, 32, 32]),
        ([4, 3, 32, 32]),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [0, 1],
)
@pytest.mark.parametrize("all_dimensions", [False])
def test_prod_with_output_nc(input_shapes, all_dimensions, dim, dst_mem_config, device):
    in_data, input_tensor = data_gen_pt_tt(input_shapes, device)
    golden_tensor = torch.prod(in_data, dim, keepdim=True)

    output_shape = input_shapes
    output_shape[dim] = 1
    out_data, output_tensor = data_gen_pt_tt(output_shape, device)

    tt_output_tensor_on_device = ttl.tensor.prod(input_tensor, all_dimensions, dim, dst_mem_config, output_tensor)
    tt_out_tensor = tt_output_tensor_on_device.cpu().to(ttl.tensor.Layout.ROW_MAJOR).to_torch()
    comp_pass, comp_out = comparison_funcs.comp_pcc(golden_tensor, tt_out_tensor)
    assert comp_pass
