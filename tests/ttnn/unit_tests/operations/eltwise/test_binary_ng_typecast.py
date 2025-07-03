# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.utility_functions import torch_random
from functools import partial
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_with_pcc


binary_fns = {
    "ge",
    "gt",
    "le",
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
@pytest.mark.parametrize(
    "layout",
    ([ttnn.TILE_LAYOUT]),
)
# No typecast on inputs and optional output
def test_opt_output_no_typecast(input_shapes, dtype, layout, ttnn_fn, device):
    torch.manual_seed(0)
    a_shape, b_shape, out_shape = input_shapes
    ttnn_op = getattr(ttnn, ttnn_fn)

    torch_input_tensor_a = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), dtype)(
        a_shape
    )
    torch_input_tensor_b = gen_func_with_cast_tt(partial(torch_random, low=-50, high=50, dtype=torch.bfloat16), dtype)(
        b_shape
    )
    out = gen_func_with_cast_tt(partial(torch_random, low=0, high=1, dtype=torch.bfloat16), dtype)(out_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, dtype=dtype, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, dtype=dtype, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.from_torch(out, dtype=dtype, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cq_id = 0
    ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    ttnn_op = getattr(ttnn, ttnn_fn)

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
    ttnn_op(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    ttnn.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    ttnn.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    ttnn.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    ttnn.sub(input_tensor_a, input_tensor_b, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    ttnn.sub_(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=False)
    output_tensor = ttnn.to_torch(input_tensor_a)

    golden_fn = ttnn.get_golden_function(ttnn.sub_)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    ttnn.sub_(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=False)
    output_tensor = ttnn.to_torch(input_tensor_a)

    golden_fn = ttnn.get_golden_function(ttnn.sub_)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
    ttnn.sub_(input_tensor_a, input_tensor_b, queue_id=cq_id, use_legacy=False)
    output_tensor = ttnn.to_torch(input_tensor_a)

    golden_fn = ttnn.get_golden_function(ttnn.sub_)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)
    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


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
        "divide",
        "rsub",
        "gt",
        "lt",
        "le",
        "ge",
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
    ttnn_op = getattr(ttnn, ttnn_fn)

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
    ttnn_op(input_tensor_a, scalar, queue_id=cq_id, output_tensor=out_tt, use_legacy=False)
    output_tensor = ttnn.to_torch(out_tt)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, scalar)

    status = ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor)
    assert status >= 0.999


@pytest.mark.parametrize("input_shape", [(1, 1, 1, 1), (3, 3, 15, 15), (3, 3, 17, 17), (3, 3, 33, 33)])
@pytest.mark.parametrize(
    "memory_config",
    ([ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]),
)
@pytest.mark.parametrize("scalar", [-0.25, -16.5, 0.0, 0.05, 1.7, 19.0])
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        "add",
        "sub",
        "mul",
        "divide",
        "rsub",
        "squared_difference",
    ],
)
@pytest.mark.parametrize(
    "layout",
    ([ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]),
)
def test_edgecase_dims_eltwise_scalar_matrix_math(input_shape, scalar, ttnn_fn, memory_config, layout, device):
    torch.manual_seed(0)
    a_shape = input_shape

    ttnn_op = getattr(ttnn, ttnn_fn)
    torch_input_tensor_a = torch.randn(a_shape, dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )

    output = ttnn_op(input_tensor_a, scalar, use_legacy=False)
    tt_output_tensor = ttnn.to_torch(output)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, scalar)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.999)


@pytest.mark.parametrize("input_shape", [(1, 1, 1, 1), (3, 3, 15, 15), (3, 3, 17, 17), (3, 3, 33, 33)])
@pytest.mark.parametrize(
    "memory_config",
    ([ttnn.DRAM_MEMORY_CONFIG]),
)
@pytest.mark.parametrize("scalar", [-1.0, -2.0, 0.0, 1.0, 2.0, 19.0])
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        "gt",
        "lt",
        "le",
        "ge",
        "eq",
        "ne",
    ],
)
@pytest.mark.parametrize(
    "layout",
    ([ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]),
)
def test_edgecase_dims_eltwise_scalar_logical(input_shape, scalar, ttnn_fn, memory_config, layout, device):
    torch.manual_seed(0)
    a_shape = input_shape

    ttnn_op = getattr(ttnn, ttnn_fn)
    torch_input_tensor_a = torch.randint(low=-50, high=50, size=a_shape, dtype=torch.bfloat16)
    # guarantee a few equal values
    if (ttnn_fn == "eq" or ttnn_fn == "ne" or ttnn_fn == "ge" or ttnn_fn == "le") and input_shape != (1, 1, 1, 1):
        torch_input_tensor_a[0, 0, 0, 0] = scalar
        torch_input_tensor_a[-1, -1, -1, -1] = scalar

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )

    output = ttnn_op(input_tensor_a, scalar, dtype=ttnn.uint32, use_legacy=False)
    tt_output_tensor = ttnn.to_torch(output)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, scalar)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        ((1, 7, 1, 1), (7, 7, 33, 33)),
        ((7, 1, 1, 1), (7, 7, 49, 49)),
        ((7, 7, 65, 65), (7, 7, 65, 65)),
        ((2, 2, 10, 1), (2, 2, 10, 2)),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    ([ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        "add",
        "sub",
        "mul",
        "divide",
        "rsub",
        "squared_difference",
    ],
)
@pytest.mark.parametrize(
    "layout",
    ([ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]),
)
def test_edgecase_dims_eltwise_broadcast_matrix_math(input_shapes, ttnn_fn, memory_config, layout, device):
    torch.manual_seed(0)
    a_shape, b_shape = input_shapes

    ttnn_op = getattr(ttnn, ttnn_fn)
    torch_input_tensor_a = torch.randn(a_shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(b_shape, dtype=torch.bfloat16)

    if ttnn_fn == "divide":
        torch_input_tensor_b[torch_input_tensor_b.abs() < 0.001] = 0.001

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )

    output = ttnn_op(input_tensor_a, input_tensor_b, dtype=ttnn.float32, use_legacy=False)
    tt_output_tensor = ttnn.to_torch(output)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        ((1, 7, 1, 1), (7, 7, 33, 33)),
        ((7, 1, 1, 1), (7, 7, 49, 49)),
        ((7, 7, 65, 65), (7, 7, 65, 65)),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    ([ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        "gt",
        "lt",
        "le",
        "ge",
        "eq",
        "ne",
    ],
)
@pytest.mark.parametrize(
    "layout",
    ([ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]),
)
def test_edgecase_dims_eltwise_broadcast_logical(input_shapes, ttnn_fn, memory_config, layout, device):
    torch.manual_seed(0)
    a_shape, b_shape = input_shapes

    ttnn_op = getattr(ttnn, ttnn_fn)
    torch_input_tensor_a = torch.randn(a_shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn(b_shape, dtype=torch.bfloat16)
    # guarantee at least one equal value
    if ttnn_fn == "eq" or ttnn_fn == "ne" or ttnn_fn == "ge" or ttnn_fn == "le":
        torch_input_tensor_a[0, 0, 0, 0] = torch_input_tensor_b[0, 0, 0, 0]

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )

    output = ttnn_op(input_tensor_a, input_tensor_b, dtype=ttnn.float32, use_legacy=False)
    tt_output_tensor = ttnn.to_torch(output)

    golden_fn = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.999)


@pytest.mark.parametrize(
    "input_shape, input_layout, input_shard_grid, input_shard_orientation, input_sharding_scheme",
    [
        (
            [1, 1, 64, 64],
            ttnn.TILE_LAYOUT,
            ttnn.CoreGrid(y=1, x=2),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardStrategy.WIDTH,
        ),
    ],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.bfloat16])
def test_binary_div(
    device,
    input_shape,
    input_layout,
    input_shard_grid,
    input_shard_orientation,
    input_sharding_scheme,
    input_dtype,
    output_dtype,
):
    memory_config = ttnn.create_sharded_memory_config(
        input_shape,
        core_grid=input_shard_grid,
        strategy=input_sharding_scheme,
        orientation=input_shard_orientation,
        use_height_and_width_as_shard_shape=False,
    )

    torch_input_a = torch.rand(input_shape, dtype=torch.bfloat16) + 1
    torch_input_b = torch.rand(input_shape, dtype=torch.bfloat16) + 1
    torch_output = torch_input_a / torch_input_b

    input_tensor_a = ttnn.from_torch(
        torch_input_a, layout=input_layout, memory_config=memory_config, dtype=input_dtype, device=device
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_b, layout=input_layout, memory_config=memory_config, dtype=input_dtype, device=device
    )
    output_tensor = ttnn.divide(input_tensor_a, input_tensor_b, dtype=output_dtype, use_legacy=False)
    assert_with_pcc(torch_output, ttnn.to_torch(output_tensor), 0.999)
