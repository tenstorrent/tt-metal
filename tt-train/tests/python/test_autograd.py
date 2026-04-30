# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ml_dtypes
import numpy as np
import pytest
import os
import sys

import ttnn
import ttml  # noqa: E402


def supported_autograd_types_except(*except_types):
    return tuple(
        data_type
        for data_type in ttnn.DataType.__members__.values()
        if data_type
        not in (
            ttnn.DataType.INVALID,
            ttnn.DataType.BFLOAT8_B,
            ttnn.DataType.BFLOAT4_B,
            ttnn.DataType.UINT8,
            ttnn.DataType.UINT16,
        )
        + tuple(except_type for except_type in except_types)
    )


def do_test_numpy_autograd_conversion(
    tensor_data,
    numpy_type,
    autograd_type,
    layout,
    expect_type_exception,
    expect_runtime_exception,
):
    numpy_tensor = np.array(tensor_data, dtype=numpy_type)
    type_error = False
    runtime_error = False
    autograd_tensor = None

    def handle_error(exception, expected, encountered):
        if (not expected) or encountered:
            raise exception
        return True

    if autograd_type:
        if layout:
            try:
                autograd_tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, layout=layout, new_type=autograd_type)
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
        else:
            try:
                autograd_tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, new_type=autograd_type)
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
    else:
        if layout:
            try:
                autograd_tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, layout=layout)
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
        else:
            try:
                autograd_tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor)
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)

    if autograd_tensor:
        try:
            assert (autograd_tensor.to_numpy() == numpy_tensor).all()
            assert (autograd_tensor.to_numpy(new_type=autograd_type) == numpy_tensor).all()
        except RuntimeError as e:
            runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
        else:
            for new_type in supported_autograd_types_except(autograd_type):
                try:
                    assert (autograd_tensor.to_numpy(new_type=new_type) == numpy_tensor).all()
                except TypeError as e:
                    type_error = handle_error(e, expect_type_exception, type_error)
                except RuntimeError as e:
                    runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
    # sanity check: the occurrence of an exception implies we were expecting it
    assert (not type_error) or (type_error and expect_type_exception)
    assert (not runtime_error) or (runtime_error and expect_runtime_exception)


default_tensor_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
numpy_data_types = [np.float32, np.int32, np.uint32, ml_dtypes.bfloat16]
metal_data_types = [
    None,
    ttnn.DataType.BFLOAT16,
    ttnn.DataType.BFLOAT4_B,
    ttnn.DataType.BFLOAT8_B,
    ttnn.DataType.FLOAT32,
    ttnn.DataType.INT32,
    ttnn.DataType.UINT32,
]
layouts = [None, ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE]


"""cases which violate format conversion rules codified in C++ nanobind python bindings"""
unsupported_format_cases = [
    (default_tensor_data, ml_dtypes.bfloat16, ttnn.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, ml_dtypes.bfloat16, ttnn.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, ml_dtypes.bfloat16, ttnn.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, ml_dtypes.bfloat16, ttnn.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, np.float32, ttnn.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, np.float32, ttnn.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, np.float32, ttnn.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, np.float32, ttnn.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, np.int32, ttnn.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        np.int32,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttnn.DataType.BFLOAT4_B, ttnn.Layout.TILE),
    (default_tensor_data, np.int32, ttnn.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        np.int32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttnn.DataType.BFLOAT8_B, ttnn.Layout.TILE),
    (default_tensor_data, np.uint32, ttnn.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.BFLOAT4_B,
        ttnn.Layout.TILE,
    ),
    (default_tensor_data, np.uint32, ttnn.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.BFLOAT8_B,
        ttnn.Layout.TILE,
    ),
]

"""cases which violate typecast rules codified in TTNN C++"""
typecast_issue_cases = [
    (default_tensor_data, np.float32, None, ttnn.Layout.ROW_MAJOR),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.FLOAT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.int32,
        ttnn.DataType.FLOAT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.FLOAT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.FLOAT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    # Row-major typecast requires padded_shape()[-1] to be a multiple of 32.
    # default_tensor_data is 3x3, so these are expected runtime failures.
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.BFLOAT16,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.int32,
        ttnn.DataType.BFLOAT16,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.BFLOAT16,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.BFLOAT16,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        None,
        ttnn.Layout.ROW_MAJOR,
    ),
]

"""TODO: multiplication cases which get the wrong answer"""
multiplication_issue_cases = [
    (default_tensor_data, np.float32, ttnn.DataType.UINT32, None),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.UINT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.float32, ttnn.DataType.UINT32, ttnn.Layout.TILE),
    (default_tensor_data, np.int32, ttnn.DataType.UINT32, None),
    (
        default_tensor_data,
        np.int32,
        ttnn.DataType.UINT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttnn.DataType.UINT32, ttnn.Layout.TILE),
    (default_tensor_data, np.uint32, None, None),
    (default_tensor_data, np.uint32, None, ttnn.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, None, ttnn.Layout.TILE),
    (default_tensor_data, np.uint32, ttnn.DataType.UINT32, None),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.UINT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.uint32, ttnn.DataType.UINT32, ttnn.Layout.TILE),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.UINT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.UINT32,
        ttnn.Layout.TILE,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.UINT32,
        None,
    ),
]

"""TODO: division cases which get the wrong answer"""
division_issue_cases = [
    (default_tensor_data, np.float32, ttnn.DataType.INT32, None),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.INT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.float32, ttnn.DataType.INT32, ttnn.Layout.TILE),
    (default_tensor_data, np.float32, ttnn.DataType.UINT32, None),
    (
        default_tensor_data,
        np.float32,
        ttnn.DataType.UINT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.float32, ttnn.DataType.UINT32, ttnn.Layout.TILE),
    (default_tensor_data, np.int32, None, None),
    (default_tensor_data, np.int32, None, ttnn.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, None, ttnn.Layout.TILE),
    (default_tensor_data, np.int32, ttnn.DataType.INT32, None),
    (
        default_tensor_data,
        np.int32,
        ttnn.DataType.INT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttnn.DataType.INT32, ttnn.Layout.TILE),
    (default_tensor_data, np.int32, ttnn.DataType.UINT32, None),
    (
        default_tensor_data,
        np.int32,
        ttnn.DataType.UINT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttnn.DataType.UINT32, ttnn.Layout.TILE),
    (default_tensor_data, np.uint32, None, None),
    (default_tensor_data, np.uint32, None, ttnn.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, None, ttnn.Layout.TILE),
    (default_tensor_data, np.uint32, ttnn.DataType.INT32, None),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.INT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.uint32, ttnn.DataType.INT32, ttnn.Layout.TILE),
    (default_tensor_data, np.uint32, ttnn.DataType.UINT32, None),
    (
        default_tensor_data,
        np.uint32,
        ttnn.DataType.UINT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.uint32, ttnn.DataType.UINT32, ttnn.Layout.TILE),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.UINT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.UINT32,
        ttnn.Layout.TILE,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.UINT32,
        None,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.INT32,
        ttnn.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttnn.DataType.INT32,
        None,
    ),
]


@pytest.mark.parametrize("tensor_data", [default_tensor_data])
@pytest.mark.parametrize("numpy_type", numpy_data_types)
@pytest.mark.parametrize("autograd_type", metal_data_types)
@pytest.mark.parametrize("layout", layouts)
def test_numpy_autograd_conversion(tensor_data, numpy_type, autograd_type, layout):
    # Skip unsupported format cases
    if (tensor_data, numpy_type, autograd_type, layout) in unsupported_format_cases:
        pytest.skip("Unsupported format combination")
    # Skip typecast issue cases
    if (tensor_data, numpy_type, autograd_type, layout) in typecast_issue_cases:
        pytest.skip("Known typecast issue")

    return do_test_numpy_autograd_conversion(
        tensor_data=tensor_data,
        numpy_type=numpy_type,
        autograd_type=autograd_type,
        layout=layout,
        expect_type_exception=False,
        expect_runtime_exception=False,
    )


@pytest.mark.parametrize(
    "tensor_data, numpy_type, autograd_type, layout",
    unsupported_format_cases,
)
def test_numpy_autograd_conversion_expecting_type_error(tensor_data, numpy_type, autograd_type, layout):
    return do_test_numpy_autograd_conversion(
        tensor_data=tensor_data,
        numpy_type=numpy_type,
        autograd_type=autograd_type,
        layout=layout,
        expect_type_exception=True,
        expect_runtime_exception=False,
    )


@pytest.mark.parametrize(
    "tensor_data, numpy_type, autograd_type, layout",
    typecast_issue_cases,
)
def test_numpy_autograd_conversion_expecting_runtime_error(tensor_data, numpy_type, autograd_type, layout):
    return do_test_numpy_autograd_conversion(
        tensor_data=tensor_data,
        numpy_type=numpy_type,
        autograd_type=autograd_type,
        layout=layout,
        expect_type_exception=False,
        expect_runtime_exception=True,
    )


def test_to_numpy_no_device_cast_returns_bfloat16_dtype():
    # cast_on_device=False on a bf16 tensor should return ml_dtypes.bfloat16 — the raw
    # storage is returned without triggering a device-side typecast.
    numpy_tensor = np.array(default_tensor_data, dtype=ml_dtypes.bfloat16)
    tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, new_type=ttnn.DataType.BFLOAT16)
    result = tensor.to_numpy(cast_on_device=False)
    assert result.dtype == ml_dtypes.bfloat16, f"Expected ml_dtypes.bfloat16, got {result.dtype}"


def test_to_numpy_no_device_cast_matches_float32_path():
    # cast_on_device=False + .astype(float32) must be numerically identical to the
    # default cast_on_device=True path with new_type=FLOAT32.
    numpy_tensor = np.array(default_tensor_data, dtype=ml_dtypes.bfloat16)
    tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, new_type=ttnn.DataType.BFLOAT16)
    via_cpu_cast = tensor.to_numpy(cast_on_device=False).astype(np.float32)
    via_device_cast = tensor.to_numpy(ttnn.DataType.FLOAT32)
    np.testing.assert_array_equal(via_cpu_cast, via_device_cast)


def test_to_numpy_no_device_cast_with_new_type_float32():
    # cast_on_device=False + new_type=FLOAT32: bf16 fetched from device without caching a
    # float32 copy in AutocastTensor; conversion to float32 happens on the CPU side.
    numpy_tensor = np.array(default_tensor_data, dtype=ml_dtypes.bfloat16)
    tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, new_type=ttnn.DataType.BFLOAT16)
    result = tensor.to_numpy(cast_on_device=False, new_type=ttnn.DataType.FLOAT32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, tensor.to_numpy(ttnn.DataType.FLOAT32))


def test_to_numpy_cast_on_device_true_unchanged():
    # cast_on_device=True (default) must behave identically to the pre-existing default path.
    numpy_tensor = np.array(default_tensor_data, dtype=ml_dtypes.bfloat16)
    tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, new_type=ttnn.DataType.BFLOAT16)
    default_result = tensor.to_numpy(ttnn.DataType.FLOAT32)
    explicit_true_result = tensor.to_numpy(cast_on_device=True, new_type=ttnn.DataType.FLOAT32)
    np.testing.assert_array_equal(default_result, explicit_true_result)


def make_tensors(tensor_data, numpy_type, autograd_type, layout):
    numpy_tensor = np.array(tensor_data, dtype=numpy_type)

    if autograd_type:
        if layout:
            autograd_tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, layout=layout, new_type=autograd_type)
        else:
            autograd_tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, new_type=autograd_type)
    else:
        if layout:
            autograd_tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor, layout=layout)
        else:
            autograd_tensor = ttml.autograd.Tensor.from_numpy(numpy_tensor)
    return (numpy_tensor, autograd_tensor)


@pytest.mark.parametrize("tensor_data", [default_tensor_data])
@pytest.mark.parametrize("numpy_type", numpy_data_types)
@pytest.mark.parametrize("autograd_type", metal_data_types)
@pytest.mark.parametrize("layout", layouts)
def test_binary_operators_add(tensor_data, numpy_type, autograd_type, layout):
    # Skip unsupported format cases
    if (tensor_data, numpy_type, autograd_type, layout) in unsupported_format_cases:
        pytest.skip("Unsupported format combination")
    # Skip typecast issue cases
    if (tensor_data, numpy_type, autograd_type, layout) in typecast_issue_cases:
        pytest.skip("Known typecast issue")

    numpy_tensor, autograd_tensor = make_tensors(tensor_data, numpy_type, autograd_type, layout)

    sum = autograd_tensor + autograd_tensor
    assert (sum.to_numpy() == (numpy_tensor + numpy_tensor)).all()


@pytest.mark.parametrize("tensor_data", [default_tensor_data])
@pytest.mark.parametrize("numpy_type", numpy_data_types)
@pytest.mark.parametrize("autograd_type", metal_data_types)
@pytest.mark.parametrize("layout", layouts)
def test_binary_operators_diff(tensor_data, numpy_type, autograd_type, layout):
    # Skip unsupported format cases
    if (tensor_data, numpy_type, autograd_type, layout) in unsupported_format_cases:
        pytest.skip("Unsupported format combination")
    # Skip typecast issue cases
    if (tensor_data, numpy_type, autograd_type, layout) in typecast_issue_cases:
        pytest.skip("Known typecast issue")

    numpy_tensor, autograd_tensor = make_tensors(tensor_data, numpy_type, autograd_type, layout)

    diff = autograd_tensor - autograd_tensor

    assert (diff.to_numpy() == (numpy_tensor - numpy_tensor)).all()


@pytest.mark.parametrize("tensor_data", [default_tensor_data])
@pytest.mark.parametrize("numpy_type", numpy_data_types)
@pytest.mark.parametrize("autograd_type", metal_data_types)
@pytest.mark.parametrize("layout", layouts)
def test_binary_operators_mul(tensor_data, numpy_type, autograd_type, layout):
    # Skip unsupported format cases
    if (tensor_data, numpy_type, autograd_type, layout) in unsupported_format_cases:
        pytest.skip("Unsupported format combination")
    # Skip typecast issue cases
    if (tensor_data, numpy_type, autograd_type, layout) in typecast_issue_cases:
        pytest.skip("Known typecast issue")
    # Skip multiplication issue cases
    if (tensor_data, numpy_type, autograd_type, layout) in multiplication_issue_cases:
        pytest.skip("Known multiplication issue")

    numpy_tensor, autograd_tensor = make_tensors(tensor_data, numpy_type, autograd_type, layout)

    mul = autograd_tensor * autograd_tensor
    mul_float = autograd_tensor * 10.0

    assert (mul.to_numpy() == (numpy_tensor * numpy_tensor)).all()
    assert (mul_float.to_numpy() == (numpy_tensor * 10.0)).all()


@pytest.mark.parametrize("tensor_data", [default_tensor_data])
@pytest.mark.parametrize("numpy_type", numpy_data_types)
@pytest.mark.parametrize("autograd_type", metal_data_types)
@pytest.mark.parametrize("layout", layouts)
def test_binary_operators_div(tensor_data, numpy_type, autograd_type, layout):
    # Skip unsupported format cases
    if (tensor_data, numpy_type, autograd_type, layout) in unsupported_format_cases:
        pytest.skip("Unsupported format combination")
    # Skip typecast issue cases
    if (tensor_data, numpy_type, autograd_type, layout) in typecast_issue_cases:
        pytest.skip("Known typecast issue")
    # Skip division issue cases
    if (tensor_data, numpy_type, autograd_type, layout) in division_issue_cases:
        pytest.skip("Known division issue")

    numpy_tensor, autograd_tensor = make_tensors(tensor_data, numpy_type, autograd_type, layout)

    div = autograd_tensor.__div__(autograd_tensor)

    assert (div.to_numpy() == (numpy_tensor / numpy_tensor)).all()
