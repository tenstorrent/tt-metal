# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
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
@pytest.mark.parametrize("out_dtype", (ttnn.uint32, ttnn.uint16, ttnn.uint8))
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

    output_tensor = ttnn.to_torch(tt_output_tensor_on_device).to(golden_tensor.dtype)

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
@pytest.mark.parametrize("out_dtype", (ttnn.uint32, ttnn.uint16, ttnn.uint8))
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

    output_tensor = ttnn.to_torch(output_tensor).to(golden_tensor.dtype)

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
    (2.3, 15.6, 55.4, 72.5, 120.6),
)
@pytest.mark.parametrize("out_dtype", (ttnn.uint32, ttnn.uint16, ttnn.uint8))
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

    output_tensor = ttnn.to_torch(tt_output_tensor_on_device).to(golden_tensor.dtype)

    are_equal = torch.equal(output_tensor, golden_tensor)
    assert are_equal


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([100])),
        (torch.Size([64, 64])),
        (torch.Size([2, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 1, 3, 320, 384])),
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
    "ttnn_function",
    (ttnn.eq, ttnn.ne),
)
def test_binary_comp_uint16_ops(input_shapes, mem_configs, ttnn_function, device):
    torch.manual_seed(0)
    in_data = torch.randint(0, 100, input_shapes, dtype=torch.int32)
    in_data[-1] = 65535
    # Make a copy of in_data so 50% of values are the same
    other_data = in_data.clone()
    # Replace half of them with new random values
    mask = torch.rand(input_shapes) > 0.5
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    other_data[mask] = torch.randint(0, 100, input_shapes, dtype=torch.int32)[mask]
    other_data[-1] = 65535
    other_tensor = ttnn.from_torch(other_data, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    cq_id = 0
    mem_cfg = mem_configs

    output_tensor = ttnn_function(input_tensor, other_tensor, memory_config=mem_cfg, queue_id=cq_id)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_fn(in_data, other_data)
    golden_tensor = golden_tensor.int()

    output_tensor = ttnn.to_torch(output_tensor).to(golden_tensor.dtype)
    assert torch.equal(output_tensor, golden_tensor)


@pytest.mark.parametrize(
    "input_dtype", (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.int32, ttnn.uint32)
)
@pytest.mark.parametrize("op", ("add", "subtract", "multiply", "maximum", "gt", "ne", "logical_and"))
def test_binary_uint8_output(device, input_dtype, op):
    torch.manual_seed(0)
    shape = (2, 3, 33, 65)
    # Even values in [0, 14]: results stay within [0, 255], and every value is exact in bfloat16 and on the
    # bfloat8_b / bfloat4_b grid of any block they share, so every dtype gives the same answer.
    a = 2 * torch.randint(4, 8, shape)
    b = 2 * torch.randint(0, 4, shape)
    ta = ttnn.from_torch(a, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    result = getattr(ttnn, op)(ta, tb, dtype=ttnn.uint8)

    assert result.dtype == ttnn.uint8
    expected = getattr(torch, op)(a, b).to(torch.uint8)
    assert torch.equal(ttnn.to_torch(result).to(torch.uint8), expected)


def test_binary_uint8_output_after_explicit_typecast(device):
    # A caller's TYPECAST(FLOAT32, UINT8) already produces the output dtype, so no second cast may follow it.
    torch.manual_seed(0)
    a = torch.randint(8, 16, (64, 64)).float()
    b = torch.randint(0, 8, (64, 64)).float()
    ta = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    result = ttnn.add(
        ta,
        tb,
        dtype=ttnn.uint8,
        activations=[
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.FLOAT32.value, ttnn.DataType.UINT8.value)
        ],
    )

    assert result.dtype == ttnn.uint8
    assert torch.equal(ttnn.to_torch(result).to(torch.uint8), (a + b).to(torch.uint8))


@pytest.mark.parametrize("input_dtype", (ttnn.int32, ttnn.uint32))
@pytest.mark.parametrize("op", ("subtract", "add", "multiply"))
def test_binary_uint8_output_wraps(device, input_dtype, op):
    # Results outside [0, 255] keep their low byte, as torch's conversion to uint8 does: subtract goes below zero
    # (a < b), add and multiply go past 255 (100 * 100).
    torch.manual_seed(0)
    shape = (2, 64, 64)
    a = torch.randint(0, 200, shape)
    b = torch.randint(0, 200, shape)
    if op == "subtract":
        a, b = torch.minimum(a, b), torch.maximum(a, b) + 1
    a[0, 0, 0], b[0, 0, 0] = (5, 9) if op == "subtract" else (100, 100)
    ta = ttnn.from_torch(a, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    result = getattr(ttnn, op)(ta, tb, dtype=ttnn.uint8)

    expected = (getattr(torch, op)(a, b) % 256).to(torch.uint8)
    assert torch.equal(ttnn.to_torch(result).to(torch.uint8), expected)


def test_binary_uint8_output_rejects_unconverted_inputs(device, expect_error):
    a = torch.randint(1, 50, (32, 32))
    ta = ttnn.from_torch(a, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "A UINT8 output is not supported for UINT16 inputs"):
        ttnn.add(ta, ta, dtype=ttnn.uint8)
    ti = ttnn.from_torch(a, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    # ttnn.divide rejects it before the device op, which rejects it too for direct callers.
    with expect_error(RuntimeError, "Incorrect output_dtype value for Integer Division"):
        ttnn.divide(ti, ti, dtype=ttnn.uint8)
