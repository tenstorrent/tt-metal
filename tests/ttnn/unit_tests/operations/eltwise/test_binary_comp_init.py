# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_dtype,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([64, 64])),
        (torch.Size([2, 32, 32])),
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "mem_configs",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
)
@pytest.mark.parametrize("out_dtype", (ttnn.uint32, ttnn.uint16))
@pytest.mark.parametrize(
    "ttnn_function",
    (ttnn.lt, ttnn.gt, ttnn.eq, ttnn.le, ttnn.ge, ttnn.ne, ttnn.logical_and, ttnn.logical_or, ttnn.logical_xor),
)
def test_binary_comp_ops(input_shapes, out_dtype, mem_configs, ttnn_function, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)

    cq_id = 0
    mem_cfg = mem_configs

    tt_output_tensor_on_device = ttnn_function(
        input_tensor, other_tensor, memory_config=mem_cfg, dtype=out_dtype, queue_id=cq_id
    )

    golden_fn = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_fn(in_data, other_data)
    golden_tensor = golden_tensor.int()

    output_tensor = ttnn.to_torch(tt_output_tensor_on_device)

    are_equal = torch.equal(output_tensor, golden_tensor)
    assert are_equal


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([64, 64])),
        (torch.Size([2, 32, 32])),
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "mem_configs",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
)
@pytest.mark.parametrize("out_dtype", (ttnn.uint32, ttnn.uint16))
@pytest.mark.parametrize(
    "ttnn_function",
    (ttnn.lt, ttnn.gt, ttnn.eq, ttnn.le, ttnn.ge, ttnn.ne, ttnn.logical_and, ttnn.logical_or, ttnn.logical_xor),
)
def test_binary_comp_opt_out(input_shapes, out_dtype, mem_configs, ttnn_function, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)

    cq_id = 0
    mem_cfg = mem_configs
    _, output_tensor = data_gen_with_range_dtype(input_shapes, -1, 1, device, False, False, out_dtype)
    ttnn_function(
        input_tensor, other_tensor, memory_config=mem_cfg, dtype=out_dtype, queue_id=cq_id, output_tensor=output_tensor
    )

    golden_fn = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_fn(in_data, other_data)
    golden_tensor = golden_tensor.int()

    output_tensor = ttnn.to_torch(output_tensor)

    are_equal = torch.equal(output_tensor, golden_tensor)
    assert are_equal


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([64, 64])),
        (torch.Size([2, 32, 32])),
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "mem_configs",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
)
@pytest.mark.parametrize(
    "scalar",
    {2.3, 15.6, 55.4, 72.5, 120.6},
)
@pytest.mark.parametrize("out_dtype", (ttnn.uint32, ttnn.uint16))
@pytest.mark.parametrize(
    "ttnn_function",
    (
        ttnn.lt,
        ttnn.gt,
        ttnn.eq,
        ttnn.le,
        ttnn.ge,
        ttnn.ne,
    ),
)
def test_binary_comp_ops_scalar(input_shapes, scalar, out_dtype, mem_configs, ttnn_function, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    cq_id = 0
    mem_cfg = mem_configs

    tt_output_tensor_on_device = ttnn_function(
        input_tensor, scalar, memory_config=mem_cfg, dtype=out_dtype, queue_id=cq_id
    )

    golden_fn = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_fn(in_data, scalar)
    golden_tensor = golden_tensor.int()

    output_tensor = ttnn.to_torch(tt_output_tensor_on_device)

    are_equal = torch.equal(output_tensor, golden_tensor)
    assert are_equal
