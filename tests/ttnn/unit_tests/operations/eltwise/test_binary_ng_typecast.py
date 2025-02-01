# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.utility_functions import skip_for_grayskull, torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt


binary_fns = {
    "gte",
    "gt",
    "lte",
    "lt",
    "eq",
    "ne",
    "logical_and",
    "logical_or",
    "logical_xor",
    "ldexp",
    "logaddexp",
    "logaddexp2",
    "squared_difference",
    "add",
    "sub",
    "rsub",
    "mul",
    "bias_gelu",
}


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1]), torch.Size([5, 3, 128, 64])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128]), torch.Size([5, 32, 128])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    binary_fns,
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.bfloat16]),
)
# No typecast on inputs and optional output
def test_opt_output_no_typecast(input_shapes, dtype, ttnn_fn, device):
    torch.manual_seed(0)
    a_shape, b_shape, out_shape = input_shapes
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)

    torch_input_tensor_a = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), dtype)(
        a_shape
    )
    torch_input_tensor_b = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), dtype)(
        b_shape
    )
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.bfloat16), dtype)(out_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.from_torch(
        out, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cq_id = 0
    ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1]), torch.Size([5, 3, 128, 64])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128]), torch.Size([5, 32, 128])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    binary_fns,
)
@pytest.mark.parametrize(
    "dtype",
    ([ttnn.bfloat8_b]),
)
# Typecast on both inputs and optional output
def test_opt_output_bf8b(input_shapes, dtype, ttnn_fn, device):
    torch.manual_seed(0)
    a_shape, b_shape, out_shape = input_shapes
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)

    torch_input_tensor_a = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), dtype)(
        a_shape
    )
    torch_input_tensor_b = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), dtype)(
        b_shape
    )
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.bfloat16), dtype)(out_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.from_torch(
        out, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cq_id = 0
    ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128])),
    ),
)
# Typecast on both inputs
def test_sub_typecast(input_shapes, device):
    a_shape, b_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(b_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    output_tensor = ttnn.experimental.sub(input_tensor_a, input_tensor_b, queue_id=cq_id)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128])),
    ),
)
# Typecast on input tensor a
def test_sub_typecast_a(input_shapes, device):
    a_shape, b_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16
    )(b_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    output_tensor = ttnn.experimental.sub(input_tensor_a, input_tensor_b, queue_id=cq_id)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128])),
    ),
)
# Typecast on input tensor b
def test_sub_typecast_b(input_shapes, device):
    a_shape, b_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(b_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    output_tensor = ttnn.experimental.sub(input_tensor_a, input_tensor_b, queue_id=cq_id)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1]), torch.Size([5, 3, 128, 64])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128]), torch.Size([5, 32, 128])),
    ),
)
# Typecast on both inputs
def test_sub_opt_output_typecast_inputs(input_shapes, device):
    a_shape, b_shape, out_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(b_shape)
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.bfloat16), ttnn.bfloat16)(out_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cq_id = 0
    ttnn.experimental.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1]), torch.Size([5, 3, 128, 64])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128]), torch.Size([5, 32, 128])),
    ),
)
# Typecast on output
def test_sub_opt_output_typecast_out(input_shapes, device):
    a_shape, b_shape, out_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16
    )(b_shape)
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.bfloat16), ttnn.bfloat8_b)(out_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cq_id = 0
    ttnn.experimental.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1]), torch.Size([5, 3, 128, 64])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128]), torch.Size([5, 32, 128])),
    ),
)
# Typecast on input tensor a
def test_sub_opt_output_typecast_a(input_shapes, device):
    a_shape, b_shape, out_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16
    )(b_shape)
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.bfloat16), ttnn.bfloat16)(out_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cq_id = 0
    ttnn.experimental.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([5, 3, 32, 32]), torch.Size([5, 3, 32, 32])),
        (torch.Size([5, 1, 64, 1]), torch.Size([1, 3, 1, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([1, 3, 128, 1]), torch.Size([5, 3, 128, 64])),
        (torch.Size([5, 1, 1]), torch.Size([1, 32, 128]), torch.Size([5, 32, 128])),
    ),
)
# Typecast on input tensor b
def test_sub_opt_output_typecast_b(input_shapes, device):
    a_shape, b_shape, out_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(b_shape)
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.bfloat16), ttnn.bfloat16)(out_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cq_id = 0
    ttnn.experimental.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 3, 64, 128]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 3, 128, 64]), torch.Size([1, 1, 128, 1])),
        (torch.Size([5, 32, 128]), torch.Size([5, 1, 1])),
    ),
)
# Typecast on both inputs
def test_inplace_sub_typecast(input_shapes, device):
    a_shape, b_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(b_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    ttnn.experimental.sub_(input_tensor_a, input_tensor_b, queue_id=cq_id)
    output_tensor = ttnn.to_torch(input_tensor_a)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub_)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 3, 64, 128]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 3, 128, 64]), torch.Size([1, 1, 128, 1])),
        (torch.Size([5, 32, 128]), torch.Size([5, 1, 1])),
    ),
)
# Typecast on input tensor a
def test_inplace_sub_typecast_a(input_shapes, device):
    a_shape, b_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16
    )(b_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    ttnn.experimental.sub_(input_tensor_a, input_tensor_b, queue_id=cq_id)
    output_tensor = ttnn.to_torch(input_tensor_a)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub_)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([5, 3, 32, 32]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 3, 64, 128]), torch.Size([1, 3, 1, 128])),
        (torch.Size([5, 3, 128, 64]), torch.Size([1, 1, 128, 1])),
        (torch.Size([5, 32, 128]), torch.Size([5, 1, 1])),
    ),
)
# Typecast on input tensor b
def test_inplace_sub_typecast_b(input_shapes, device):
    a_shape, b_shape = input_shapes

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat16
    )(a_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(b_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    ttnn.experimental.sub_(input_tensor_a, input_tensor_b, queue_id=cq_id)
    output_tensor = ttnn.to_torch(input_tensor_a)

    golden_fn = ttnn.get_golden_function(ttnn.experimental.sub_)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 1]), torch.Size([1, 1, 1, 1])),
        (torch.Size([5, 3, 64, 128]), torch.Size([5, 3, 64, 128])),
        (torch.Size([5, 1, 1, 64]), torch.Size([5, 1, 1, 64])),
        (torch.Size([5, 32, 32]), torch.Size([5, 32, 32])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        "add",
        "sub",
        "mul",
        "div",
        "rsub",
        "gt",
        "lt",
        "lte",
        "gte",
        "eq",
        "ne",
        "squared_difference",
    ],
)
@pytest.mark.parametrize("scalar", [-0.25, -16.5, 0.0, 0.05, 1.7, 19.0])
# Typecast on both input and optional tensor
def test_opt_output_scalar(input_shapes, ttnn_fn, scalar, device):
    torch.manual_seed(0)
    a_shape, out_shape = input_shapes
    ttnn_op = getattr(ttnn.experimental, ttnn_fn)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), ttnn.bfloat8_b
    )(a_shape)
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.bfloat16), ttnn.bfloat8_b)(out_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    cq_id = 0
    ttnn_op(input_tensor_a, scalar, queue_id=cq_id, output_tensor=out_tt)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, scalar)

    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999
