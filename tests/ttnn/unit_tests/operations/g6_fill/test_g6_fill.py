# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_fill: out[i] = cast_T(fill_value) for every logical position.

Shapes are the production cases only (BB-22). The grade is bit equality of the
logical values: a fill has no precision to lose. int32 ROW_MAJOR is a
refinement: its cases xfail while the op refuses them as unsupported, and must
pass once it is in SUPPORTED.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_fill import g6_fill
from ttnn.operations._op_contract import SupportRefusal

FLOAT32_TILE_SHAPES = [(512,), (256,), (207,), (196,), (185,)]
INT32_ROW_MAJOR_SHAPE = (1,)
FLT_MAX = 3.4028234663852886e38

_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.int32: torch.int32}


def _element(dtype, value):
    """e = cast_T(value) as a 0-d tensor, converted once from the exact value.

    torch's int64 -> float32 cast rounds once; float(int) would round twice.
    """
    source = torch.float64 if isinstance(value, float) else torch.int64
    return torch.tensor(value, dtype=source).to(_TORCH_DTYPE[dtype])


def _bits(tensor):
    return tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor


def _assert_bits_equal(actual, expected, what):
    assert actual.dtype == expected.dtype, f"{what}: dtype {actual.dtype}, expected {expected.dtype}"
    assert tuple(actual.shape) == tuple(
        expected.shape
    ), f"{what}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    got, want = _bits(actual.contiguous()).flatten(), _bits(expected.contiguous()).flatten()
    wrong = (got != want).nonzero().flatten()
    if wrong.numel():
        i = int(wrong[0])
        raise AssertionError(
            f"{what}: {wrong.numel()} of {want.numel()} elements differ; first at {i}: "
            f"got 0x{int(got[i]) & 0xFFFFFFFF:08x}, expected 0x{int(want[i]) & 0xFFFFFFFF:08x}"
        )


def _draw(shape, dtype, avoid, offset=0.0):
    """Random contents, none bit-equal to the fill element, so an unwritten position shows."""
    if dtype == ttnn.int32:
        values = torch.randint(1, 100, shape, dtype=torch.int32)
    else:
        values = torch.randn(shape, dtype=torch.float32) + offset
    values[_bits(values) == _bits(avoid.to(values.dtype))] += 1
    return values


def _to_device(values, device, dtype, layout, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(values, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _overwrite(device_tensor, values, dtype, layout):
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(values, dtype=dtype, layout=layout), device_tensor)


def _fill(input_tensor, value, output_tensor):
    try:
        return g6_fill(input_tensor, value, output_tensor=output_tensor)
    except SupportRefusal as refusal:
        if input_tensor.dtype == ttnn.float32 and input_tensor.layout == ttnn.TILE_LAYOUT:
            raise
        pytest.xfail(f"not yet in SUPPORTED: {refusal}")


def _make_case(device, shape, dtype, layout, first_value, *, alias):
    torch_input = _draw(shape, dtype, avoid=_element(dtype, first_value))
    input_tensor = _to_device(torch_input, device, dtype, layout)
    if alias:
        return torch_input, input_tensor, input_tensor
    output_tensor = _to_device(torch.zeros(shape, dtype=_TORCH_DTYPE[dtype]), device, dtype, layout)
    return torch_input, input_tensor, output_tensor


def _fill_and_check(input_tensor, output_tensor, value, shape, dtype, layout, *, offset):
    element = _element(dtype, value)
    _overwrite(output_tensor, _draw(shape, dtype, avoid=element, offset=offset), dtype, layout)
    result = _fill(input_tensor, value, output_tensor)
    assert isinstance(result, ttnn.Tensor), f"returned {type(result).__name__}, not output_tensor"
    assert result.buffer_address() == output_tensor.buffer_address(), "result does not use output_tensor's buffer"
    assert (tuple(result.shape), result.dtype, result.layout) == (
        tuple(output_tensor.shape),
        output_tensor.dtype,
        output_tensor.layout,
    )
    _assert_bits_equal(ttnn.to_torch(output_tensor), element.expand(shape), f"output after fill_value={value!r}")


def _run(device, shape, dtype, layout, values, *, alias=False):
    torch.manual_seed(42)
    torch_input, input_tensor, output_tensor = _make_case(device, shape, dtype, layout, values[0], alias=alias)
    for value in values:
        _fill_and_check(input_tensor, output_tensor, value, shape, dtype, layout, offset=0.0 if alias else 10.0)
    if not alias:
        _assert_bits_equal(ttnn.to_torch(input_tensor), torch_input, "input_tensor after the calls")


@pytest.mark.parametrize("shape", FLOAT32_TILE_SHAPES, ids=lambda s: f"N{s[0]}")
def test_fill_float32_tile(device, shape):
    _run(device, shape, ttnn.float32, ttnn.TILE_LAYOUT, [1.0])


def test_fill_int32_row_major(device):
    _run(device, INT32_ROW_MAJOR_SHAPE, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, [0])


@pytest.mark.parametrize("shape", [(512,), (207,)], ids=lambda s: f"N{s[0]}")
def test_fill_in_place(device, shape):
    _run(device, shape, ttnn.float32, ttnn.TILE_LAYOUT, [1.0], alias=True)


def test_fill_float32_value_bits(device):
    # Values a 16-bit route, a subnormal flush, a sign-dropping shortcut or a
    # double-rounded int conversion gets wrong while 1.0 survives them.
    values = [
        1 / 3,
        -0.0,
        1e-40,
        float("inf"),
        float("-inf"),
        float("nan"),
        FLT_MAX,
        16777217,
        2**60 + 2**36 + 1,
    ]
    _run(device, (196,), ttnn.float32, ttnn.TILE_LAYOUT, values)


def test_fill_int32_value_bits(device):
    _run(device, INT32_ROW_MAJOR_SHAPE, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, [16777217, -(2**31), 2**31 - 1])


def test_new_value_reuses_cached_program(device):
    shape, dtype, layout = (512,), ttnn.float32, ttnn.TILE_LAYOUT
    torch.manual_seed(42)
    torch_input, input_tensor, output_tensor = _make_case(device, shape, dtype, layout, 1.0, alias=False)
    for call, value in enumerate([1.0, 1.0, 0.1, 1.0]):
        entries_before = device.num_program_cache_entries()
        _fill_and_check(input_tensor, output_tensor, value, shape, dtype, layout, offset=10.0)
        if call > 0:
            assert device.num_program_cache_entries() == entries_before, (
                f"call {call} (fill_value={value!r}) built a new program: the value must reach the cached "
                "program as a runtime argument"
            )
    _assert_bits_equal(ttnn.to_torch(input_tensor), torch_input, "input_tensor after the calls")


def test_refused_fill_values_float32(device, expect_error):
    shape, dtype, layout = (196,), ttnn.float32, ttnn.TILE_LAYOUT
    torch.manual_seed(42)
    _, input_tensor, output_tensor = _make_case(device, shape, dtype, layout, 1.0, alias=False)
    _fill_and_check(input_tensor, output_tensor, 1.0, shape, dtype, layout, offset=10.0)
    before = ttnn.to_torch(output_tensor)
    for value in [True, False, 1e39, -1e39, 3.4028235e38, -(2**128), "1.0", None]:
        with expect_error(ValueError, ""):
            g6_fill(input_tensor, value, output_tensor=output_tensor)
        _assert_bits_equal(ttnn.to_torch(output_tensor), before, f"output after refused fill_value={value!r}")


def test_refused_fill_values_int32(device, expect_error):
    shape, dtype, layout = INT32_ROW_MAJOR_SHAPE, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
    torch.manual_seed(42)
    _, input_tensor, output_tensor = _make_case(device, shape, dtype, layout, 0, alias=False)
    before = ttnn.to_torch(output_tensor)
    for value in [1.5, 0.0, 2**31, -(2**31) - 1, True]:
        with expect_error(ValueError, ""):
            g6_fill(input_tensor, value, output_tensor=output_tensor)
        _assert_bits_equal(ttnn.to_torch(output_tensor), before, f"output after refused fill_value={value!r}")


def _assert_refused(expect_error, input_tensor, output_tensor, what):
    # A contract violation is not a support refusal: SupportRefusal subclasses
    # NotImplementedError (a RuntimeError) and must not satisfy this check.
    with expect_error((ValueError, RuntimeError), "") as info:
        g6_fill(input_tensor, 1.0, output_tensor=output_tensor)
    assert not isinstance(info.value, SupportRefusal), f"{what}: refused as unsupported, not as invalid"


def test_refused_tensors(device, expect_error):
    torch.manual_seed(42)
    shape = (196,)
    values = torch.randn(shape, dtype=torch.float32)
    tile, row_major = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
    x = _to_device(values, device, ttnn.float32, tile)
    out = _to_device(torch.zeros(shape), device, ttnn.float32, tile)
    before = ttnn.to_torch(out)

    x_rank2 = _to_device(values.reshape(1, 196), device, ttnn.float32, tile)
    out_rank2 = _to_device(torch.zeros(1, 196), device, ttnn.float32, tile)
    _assert_refused(expect_error, x_rank2, out_rank2, "rank 2")

    x_bf16 = _to_device(values, device, ttnn.bfloat16, tile)
    out_bf16 = _to_device(torch.zeros(shape), device, ttnn.bfloat16, tile)
    _assert_refused(expect_error, x_bf16, out_bf16, "bfloat16 tensors")

    _assert_refused(expect_error, x, None, "output_tensor None")
    _assert_refused(expect_error, x, _to_device(torch.zeros(256), device, ttnn.float32, tile), "output shape differs")
    _assert_refused(expect_error, x, out_bf16, "output dtype differs")
    _assert_refused(
        expect_error, x, _to_device(torch.zeros(shape), device, ttnn.float32, row_major), "output layout differs"
    )
    _assert_refused(
        expect_error,
        x,
        _to_device(torch.zeros(shape), device, ttnn.float32, tile, ttnn.L1_MEMORY_CONFIG),
        "output placement differs",
    )

    x_l1 = _to_device(values, device, ttnn.float32, tile, ttnn.L1_MEMORY_CONFIG)
    out_l1 = _to_device(torch.zeros(shape), device, ttnn.float32, tile, ttnn.L1_MEMORY_CONFIG)
    _assert_refused(expect_error, x_l1, out_l1, "L1 interleaved tensors")

    x_host = ttnn.from_torch(values, dtype=ttnn.float32, layout=tile)
    _assert_refused(expect_error, x_host, out, "input on host")
    _assert_refused(
        expect_error, x, ttnn.from_torch(torch.zeros(shape), dtype=ttnn.float32, layout=tile), "output on host"
    )

    _assert_bits_equal(ttnn.to_torch(out), before, "output after the refused calls")
