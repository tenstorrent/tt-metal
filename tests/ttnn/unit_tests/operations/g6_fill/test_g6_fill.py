# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_fill.

Shapes are the production cases only (BB-22): float32 TILE [512], [256], [207],
[196], [185] and int32 ROW_MAJOR [1]. The grade is bit equality of the logical
elements, the exact gate of eval/golden_tests/g6_fill/helpers.py:TOLERANCES.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_fill import SUPPORTED, g6_fill

FLOAT32_SHAPES = [(512,), (256,), (207,), (196,), (185,)]
INT32_SHAPE = (1,)
FLT_MAX = 3.4028234663852886e38


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


def reference_fill(shape, torch_dtype, value):
    """torch.full_like: the value converted once, from its exact value, to the dtype."""
    scalar = torch.tensor(value, dtype=torch.float64 if isinstance(value, float) else torch.int64)
    return scalar.to(torch_dtype).expand(shape).clone()


def _bits(tensor):
    # float32 compared as int32 so -0.0, NaN payloads and subnormals count.
    return tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor


def assert_bit_exact(actual, expected, what):
    assert actual.dtype == expected.dtype and tuple(actual.shape) == tuple(
        expected.shape
    ), f"{what}: got {actual.dtype} {tuple(actual.shape)}, expected {expected.dtype} {tuple(expected.shape)}"
    wrong = (_bits(actual.contiguous()) != _bits(expected.contiguous())).flatten().nonzero().flatten()
    if wrong.numel():
        i = int(wrong[0])
        got = int(_bits(actual.contiguous()).flatten()[i]) & 0xFFFFFFFF
        want = int(_bits(expected.contiguous()).flatten()[i]) & 0xFFFFFFFF
        raise AssertionError(
            f"{what}: {wrong.numel()} of {expected.numel()} logical elements differ; "
            f"first at {i}: got 0x{got:08x}, expected 0x{want:08x}"
        )


def values_avoiding(shape, torch_dtype, value, offset=0.0):
    """Random values, none with the fill element's bit pattern, so an unwritten position shows."""
    if torch_dtype == torch.int32:
        values = torch.randint(1, 100, shape, dtype=torch.int32)
    else:
        values = torch.randn(shape, dtype=torch.float32) + offset
    clash = _bits(values) == _bits(reference_fill((), torch_dtype, value))
    values[clash] += 1
    return values


def to_device(device, values, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(values, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def overwrite(device_tensor, values, dtype, layout):
    """New values into the existing buffer, as the model's preallocated output keeps its buffer."""
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(values, dtype=dtype, layout=layout), device_tensor)


def assert_returned_output(result, output_tensor):
    assert isinstance(result, ttnn.Tensor), f"returned {type(result).__name__}, not output_tensor"
    assert (
        result.buffer_address() == output_tensor.buffer_address()
    ), "returned tensor does not use output_tensor's buffer (the result was reallocated)"
    assert (tuple(result.shape), result.dtype, result.layout) == (
        tuple(output_tensor.shape),
        output_tensor.dtype,
        output_tensor.layout,
    )


def _float32_pair(device, shape, value):
    x_values = torch.randn(shape, dtype=torch.float32)
    x = to_device(device, x_values, ttnn.float32, ttnn.TILE_LAYOUT)
    out = to_device(device, values_avoiding(shape, torch.float32, value, offset=10.0), ttnn.float32, ttnn.TILE_LAYOUT)
    return x_values, x, out


@pytest.mark.parametrize("shape", FLOAT32_SHAPES, ids=_shape_id)
def test_fill_production(device, shape):
    """The model's call: float32 TILE, 1.0, separate output; the input stays unchanged."""
    torch.manual_seed(42)
    x_values, x, out = _float32_pair(device, shape, 1.0)

    result = g6_fill(x, 1.0, output_tensor=out)

    assert_returned_output(result, out)
    assert_bit_exact(ttnn.to_torch(out), reference_fill(shape, torch.float32, 1.0), "output_tensor")
    assert_bit_exact(ttnn.to_torch(x), x_values, "input_tensor after the call")


# Values a 16-bit route, a flushed subnormal, a hard-coded constant or a double
# rounding of an int gets wrong while 1.0 survives them.
VALUE_PROBES = [
    pytest.param(1 / 3, id="one_third"),
    pytest.param(-0.0, id="neg_zero"),
    pytest.param(1e-40, id="subnormal"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="neg_inf"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(FLT_MAX, id="flt_max"),
    pytest.param(0, id="int_zero"),
    pytest.param(16777217, id="int_2p24_plus_1"),
    pytest.param(2**60 + 2**36 + 1, id="int_double_rounding_trap"),
]


@pytest.mark.parametrize("value", VALUE_PROBES)
@pytest.mark.parametrize("shape", [(196,), (185,)], ids=_shape_id)
def test_fill_value_bits(device, shape, value):
    """Every logical element is cast_float32(value) bit for bit."""
    torch.manual_seed(42)
    x_values, x, out = _float32_pair(device, shape, value)

    result = g6_fill(x, value, output_tensor=out)

    assert_returned_output(result, out)
    assert_bit_exact(ttnn.to_torch(out), reference_fill(shape, torch.float32, value), f"fill_value={value!r}")
    assert_bit_exact(ttnn.to_torch(x), x_values, "input_tensor after the call")


def test_fill_repeat_new_value(device):
    """Repeated calls into one output with a cached program: each call holds its own value."""
    torch.manual_seed(42)
    shape = (512,)
    x_values, x, out = _float32_pair(device, shape, 1.0)

    for value in [1.0, 1.0, 0.1, 1.0, -2.5]:
        overwrite(out, values_avoiding(shape, torch.float32, value, offset=10.0), ttnn.float32, ttnn.TILE_LAYOUT)
        result = g6_fill(x, value, output_tensor=out)
        assert_returned_output(result, out)
        assert_bit_exact(ttnn.to_torch(out), reference_fill(shape, torch.float32, value), f"fill_value={value!r}")

    assert_bit_exact(ttnn.to_torch(x), x_values, "input_tensor after the calls")


@pytest.mark.parametrize("shape", [(512,), (207,)], ids=_shape_id)
def test_fill_in_place(device, shape):
    """output_tensor is input_tensor."""
    torch.manual_seed(42)
    x = to_device(device, values_avoiding(shape, torch.float32, 1.0), ttnn.float32, ttnn.TILE_LAYOUT)

    result = g6_fill(x, 1.0, output_tensor=x)

    assert_returned_output(result, x)
    assert_bit_exact(ttnn.to_torch(x), reference_fill(shape, torch.float32, 1.0), "in-place result")


def test_fill_int32_row_major(device, expect_error):
    """The model's int32 ROW_MAJOR [1] call. Until int32 ROW_MAJOR is in SUPPORTED the op
    must refuse it with a typed support refusal and leave output_tensor untouched."""
    torch.manual_seed(42)
    supported = ttnn.int32 in SUPPORTED["dtype"] and ttnn.ROW_MAJOR_LAYOUT in SUPPORTED["layout"]
    x_values = torch.tensor([7], dtype=torch.int32)
    x = to_device(device, x_values, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    out = to_device(device, torch.tensor([5], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    for value in [0, 16777217, -(2**31), 2**31 - 1]:
        overwrite(out, values_avoiding(INT32_SHAPE, torch.int32, value), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        if not supported:
            before = ttnn.to_torch(out)
            with expect_error(NotImplementedError, ""):
                g6_fill(x, value, output_tensor=out)
            assert_bit_exact(ttnn.to_torch(out), before, f"output_tensor after the refused fill_value={value!r}")
            continue
        result = g6_fill(x, value, output_tensor=out)
        assert_returned_output(result, out)
        assert_bit_exact(ttnn.to_torch(out), reference_fill(INT32_SHAPE, torch.int32, value), f"fill_value={value!r}")

    assert_bit_exact(ttnn.to_torch(x), x_values, "input_tensor after the calls")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool_true"),
        pytest.param(False, id="bool_false"),
        pytest.param(1e39, id="above_flt_max"),
        pytest.param(-1e39, id="below_neg_flt_max"),
        pytest.param(2**128, id="int_above_flt_max"),
        pytest.param("1.0", id="str"),
        pytest.param(None, id="none"),
    ],
)
def test_fill_refuses_float32_value(device, value, expect_error):
    """A value float32 cannot take is refused with ValueError, after an admitted call, and
    output_tensor keeps what that call wrote."""
    torch.manual_seed(42)
    shape = (196,)
    _, x, out = _float32_pair(device, shape, 1.0)
    g6_fill(x, 1.0, output_tensor=out)
    before = ttnn.to_torch(out)

    with expect_error(ValueError, ""):
        g6_fill(x, value, output_tensor=out)

    assert_bit_exact(ttnn.to_torch(out), before, f"output_tensor after the refused fill_value={value!r}")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1.5, id="float"),
        pytest.param(0.0, id="integral_float"),
        pytest.param(2**31, id="above_int32_max"),
        pytest.param(-(2**31) - 1, id="below_int32_min"),
        pytest.param(True, id="bool"),
    ],
)
def test_fill_refuses_int32_value(device, value, expect_error):
    """A value int32 cannot take exactly is refused with ValueError, whatever SUPPORTED says."""
    x = to_device(device, torch.tensor([7], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    out = to_device(device, torch.tensor([5], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    before = ttnn.to_torch(out)

    with expect_error(ValueError, ""):
        g6_fill(x, value, output_tensor=out)

    assert_bit_exact(ttnn.to_torch(out), before, f"output_tensor after the refused fill_value={value!r}")


def test_fill_refuses_missing_output(device, expect_error):
    torch.manual_seed(42)
    _, x, _ = _float32_pair(device, (196,), 1.0)
    with expect_error(ValueError, ""):
        g6_fill(x, 1.0, output_tensor=None)


def _tensor_mismatches(device):
    """(name, input_tensor, output_tensor) triples the contract refuses."""
    f32, tile, l1 = ttnn.float32, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG
    a196 = torch.randn((196,), dtype=torch.float32)
    a512, a256, a1x196 = torch.randn((512,)), torch.randn((256,)), a196.reshape(1, 196)
    good = to_device(device, a196, f32, tile)
    host = ttnn.from_torch(a196, dtype=f32, layout=tile)
    return [
        ("output_shape", to_device(device, a512, f32, tile), to_device(device, a256, f32, tile)),
        ("output_layout", good, to_device(device, a196, f32, ttnn.ROW_MAJOR_LAYOUT)),
        ("output_dtype", good, to_device(device, torch.zeros((196,), dtype=torch.int32), ttnn.int32, tile)),
        ("output_l1", good, to_device(device, a196, f32, tile, memory_config=l1)),
        ("rank_2", to_device(device, a1x196, f32, tile), to_device(device, a1x196, f32, tile)),
        ("bfloat16", to_device(device, a196, ttnn.bfloat16, tile), to_device(device, a196, ttnn.bfloat16, tile)),
        ("input_l1", to_device(device, a196, f32, tile, memory_config=l1), good),
        ("host_tensors", host, ttnn.from_torch(a196, dtype=f32, layout=tile)),
    ]


def test_fill_refuses_tensors(device, expect_error):
    """Rank, dtype, placement and output mismatches are refused before anything is written."""
    torch.manual_seed(42)
    for name, x, out in _tensor_mismatches(device):
        on_device = ttnn.is_tensor_storage_on_device(out)
        before = ttnn.to_torch(out) if on_device else None
        with expect_error((ValueError, RuntimeError), ""):
            g6_fill(x, 1.0, output_tensor=out)
        if on_device:
            assert torch.equal(_bits(ttnn.to_torch(out)), _bits(before)), f"{name}: output_tensor was written"
