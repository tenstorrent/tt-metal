# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
                autograd_tensor = ttml.autograd.Tensor.from_numpy(
                    numpy_tensor, layout=layout, new_type=autograd_type
                )
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
        else:
            try:
                autograd_tensor = ttml.autograd.Tensor.from_numpy(
                    numpy_tensor, new_type=autograd_type
                )
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
    else:
        if layout:
            try:
                autograd_tensor = ttml.autograd.Tensor.from_numpy(
                    numpy_tensor, layout=layout
                )
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
        assert (autograd_tensor.to_numpy() == numpy_tensor).all()
        assert (autograd_tensor.to_numpy(new_type=autograd_type) == numpy_tensor).all()
        for new_type in supported_autograd_types_except(autograd_type):
            try:
                assert (
                    autograd_tensor.to_numpy(new_type=new_type) == numpy_tensor
                ).all()
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
    # sanity check: the occurence of an exception implies we were expecting it
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
def test_numpy_autograd_conversion_expecting_type_error(
    tensor_data, numpy_type, autograd_type, layout
):
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
def test_numpy_autograd_conversion_expecting_runtime_error(
    tensor_data, numpy_type, autograd_type, layout
):
    return do_test_numpy_autograd_conversion(
        tensor_data=tensor_data,
        numpy_type=numpy_type,
        autograd_type=autograd_type,
        layout=layout,
        expect_type_exception=False,
        expect_runtime_exception=True,
    )


def make_tensors(tensor_data, numpy_type, autograd_type, layout):
    numpy_tensor = np.array(tensor_data, dtype=numpy_type)

    if autograd_type:
        if layout:
            autograd_tensor = ttml.autograd.Tensor.from_numpy(
                numpy_tensor, layout=layout, new_type=autograd_type
            )
        else:
            autograd_tensor = ttml.autograd.Tensor.from_numpy(
                numpy_tensor, new_type=autograd_type
            )
    else:
        if layout:
            autograd_tensor = ttml.autograd.Tensor.from_numpy(
                numpy_tensor, layout=layout
            )
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

    numpy_tensor, autograd_tensor = make_tensors(
        tensor_data, numpy_type, autograd_type, layout
    )

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

    numpy_tensor, autograd_tensor = make_tensors(
        tensor_data, numpy_type, autograd_type, layout
    )

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

    numpy_tensor, autograd_tensor = make_tensors(
        tensor_data, numpy_type, autograd_type, layout
    )

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

    numpy_tensor, autograd_tensor = make_tensors(
        tensor_data, numpy_type, autograd_type, layout
    )

    div = autograd_tensor.__div__(autograd_tensor)

    assert (div.to_numpy() == (numpy_tensor / numpy_tensor)).all()
