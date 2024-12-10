# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    compare_pcc,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Possible accuracy issues with grayskull")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.experimental.gte,
        ttnn.experimental.gt,
        ttnn.experimental.lte,
        ttnn.experimental.lt,
        ttnn.experimental.eq,
        ttnn.experimental.ne,
        ttnn.experimental.logical_and,
        ttnn.experimental.logical_or,
        ttnn.experimental.logical_xor,
        ttnn.experimental.ldexp,
        ttnn.experimental.logaddexp,
        ttnn.experimental.logaddexp2,
        ttnn.experimental.squared_difference,
        ttnn.experimental.add,
        ttnn.experimental.sub,
        ttnn.experimental.mul,
        ttnn.experimental.div,
        ttnn.experimental.bias_gelu,
    ],
)
def test_binary_scalar_ops(input_shapes, ttnn_fn, device):
    a_shape, b_shape = input_shapes
    a_pt = torch.rand(a_shape).bfloat16()
    b_pt = torch.rand(b_shape).bfloat16()

    a_tt = ttnn.from_torch(a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_tt = ttnn.from_torch(b_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cq_id = 0
    out_tt = ttnn_fn(a_tt, b_tt, queue_id=cq_id)
    golden_fn = ttnn.get_golden_function(ttnn_fn)
    out_pt = golden_fn(a_pt, b_pt)

    comp_pass = compare_pcc([out_tt], [out_pt])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 31, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 2, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([2, 3, 128, 1])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.experimental.gte,
        ttnn.experimental.gt,
        ttnn.experimental.lte,
        ttnn.experimental.lt,
        ttnn.experimental.eq,
        ttnn.experimental.ne,
        ttnn.experimental.logical_and,
        ttnn.experimental.logical_or,
        ttnn.experimental.logical_xor,
        ttnn.experimental.ldexp,
        ttnn.experimental.logaddexp,
        ttnn.experimental.logaddexp2,
        ttnn.experimental.squared_difference,
        ttnn.experimental.add,
        ttnn.experimental.sub,
        ttnn.experimental.mul,
        ttnn.experimental.div,
        ttnn.experimental.bias_gelu,
    ],
)
def test_binary_scalar_ops_invalid_bcast(input_shapes, ttnn_fn, device):
    a_shape, b_shape = input_shapes
    a_pt = torch.rand(a_shape).bfloat16()
    b_pt = torch.rand(b_shape).bfloat16()

    a_tt = ttnn.from_torch(a_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    b_tt = ttnn.from_torch(b_pt, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    with pytest.raises(RuntimeError) as e:
        cq_id = 0
        _ = ttnn_fn(a_tt, b_tt, queue_id=cq_id)
        assert "Broadcasting rule violation" in str(e.value)


@pytest.mark.parametrize(
    "shapes",
    [
        [[1, 71, 7, 7], [7, 7]],
        [[920, 1, 256], [256]],
        [[4, 12, 64, 64], [12, 1, 1]],
        [[4, 16, 64, 64], [16, 1, 1]],
        [[64, 3, 64, 64], [3, 1, 1]],
        [[64, 4, 64, 64], [4, 1, 1]],
        [[16, 6, 64, 64], [6, 1, 1]],
        [[16, 8, 64, 64], [8, 1, 1]],
        [[16, 1], [1, 1, 32]],
    ],
)
def test_unequal_ranks(device, shapes):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_tensor = ttnn.experimental.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988


@pytest.mark.parametrize(
    "data",
    [
        ([], [], []),
        ([1], [2], [3]),
        ([1], [], []),
        ([], [1], []),
        ([1, 2], [3], [4, 5]),
        ([1], [2, 3], [3, 4]),
        ([1, 2], [3, 4], [4, 6]),
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_01_volume_tensors(device, data, memory_config):
    (a, b, c_golden) = data
    a = torch.BFloat16Tensor(a)
    b = torch.BFloat16Tensor(b)
    assert torch.add(a, b).tolist() == c_golden

    ttnn_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.experimental.add(ttnn_a, ttnn_b)
    c = ttnn.to_torch(ttnn_c).reshape((-1))

    assert c.tolist() == c_golden
