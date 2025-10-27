# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ml_dtypes
import numpy as np
import pytest
import os
import sys

import ttml  # noqa: E402


def supported_autograd_types_except(*except_types):
    return tuple(
        data_type
        for data_type in ttml.autograd.DataType.__members__.values()
        if data_type
        not in (
            ttml.autograd.DataType.INVALID,
            ttml.autograd.DataType.BFLOAT8_B,
            ttml.autograd.DataType.BFLOAT4_B,
            ttml.autograd.DataType.UINT8,
            ttml.autograd.DataType.UINT16,
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
        assert (autograd_tensor.to_numpy() == numpy_tensor).all()
        assert (autograd_tensor.to_numpy(new_type=autograd_type) == numpy_tensor).all()
        for new_type in supported_autograd_types_except(autograd_type):
            try:
                assert (autograd_tensor.to_numpy(new_type=new_type) == numpy_tensor).all()
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
    ttml.autograd.DataType.BFLOAT16,
    ttml.autograd.DataType.BFLOAT4_B,
    ttml.autograd.DataType.BFLOAT8_B,
    ttml.autograd.DataType.FLOAT32,
    ttml.autograd.DataType.INT32,
    ttml.autograd.DataType.UINT32,
]
layouts = [None, ttml.Layout.ROW_MAJOR, ttml.Layout.TILE]


def generate_all_test_cases(
    tensor_data=None,
    numpy_types=None,
    metal_types=None,
    layout_types=None,
):
    """
    Generate test cases with optional filtering.

    Args:
        tensor_data: Tensor data to use (default: default_tensor_data)
        numpy_types: List of numpy types to test (default: all numpy_data_types)
        metal_types: List of metal types to test (default: all metal_data_types)
        layout_types: List of layouts to test (default: all layouts)

    Examples:
        # Generate all cases (default)
        all_test_cases = generate_all_test_cases()

        # Only test float32 with all metal types and layouts
        float32_cases = generate_all_test_cases(numpy_types=[np.float32])

        # Only test with TILE layout
        tile_cases = generate_all_test_cases(layout_types=[ttml.Layout.TILE])

        # Combine filters: only float32 + bfloat16 with TILE layout
        custom_cases = generate_all_test_cases(
            numpy_types=[np.float32],
            metal_types=[ttml.autograd.DataType.BFLOAT16],
            layout_types=[ttml.Layout.TILE]
        )
    """
    tensor_data = tensor_data if tensor_data is not None else default_tensor_data
    numpy_types = numpy_types if numpy_types is not None else numpy_data_types
    metal_types = metal_types if metal_types is not None else metal_data_types
    layout_types = layout_types if layout_types is not None else layouts

    ret = []
    for numpy_data_type in numpy_types:
        for metal_data_type in metal_types:
            for layout in layout_types:
                ret.append((tensor_data, numpy_data_type, metal_data_type, layout))
    return ret


# Generate all test cases (you can customize this by passing parameters)
all_test_cases = generate_all_test_cases()

# Example: If you want to test only specific combinations, you can create additional test case sets:
# float32_only_cases = generate_all_test_cases(numpy_types=[np.float32])
# tile_layout_cases = generate_all_test_cases(layout_types=[ttml.Layout.TILE])


def join_lists(*lists):
    ret = []
    for l in lists:
        for o in l:
            ret.append(o)
    return ret


def cases_except(cases, *cases_to_skip):
    actual_cases_to_skip = []
    for case_or_cases in cases_to_skip:
        if type(case_or_cases) == list:
            for case in case_or_cases:
                actual_cases_to_skip.append(case)
        else:
            actual_cases_to_skip.append(case_or_cases)
    return [test_case for test_case in cases if test_case not in actual_cases_to_skip]


"""cases which violate format conversion rules codified in C++ nanobind python bindings"""
unsupported_format_cases = [
    (default_tensor_data, ml_dtypes.bfloat16, ttml.autograd.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, ml_dtypes.bfloat16, ttml.autograd.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, ml_dtypes.bfloat16, ttml.autograd.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, ml_dtypes.bfloat16, ttml.autograd.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, np.float32, ttml.autograd.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, np.float32, ttml.autograd.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, np.float32, ttml.autograd.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, np.float32, ttml.autograd.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, np.int32, ttml.autograd.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        np.int32,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttml.autograd.DataType.BFLOAT4_B, ttml.Layout.TILE),
    (default_tensor_data, np.int32, ttml.autograd.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        np.int32,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttml.autograd.DataType.BFLOAT8_B, ttml.Layout.TILE),
    (default_tensor_data, np.uint32, ttml.autograd.DataType.BFLOAT4_B, None),
    (
        default_tensor_data,
        np.uint32,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.uint32,
        ttml.autograd.DataType.BFLOAT4_B,
        ttml.Layout.TILE,
    ),
    (default_tensor_data, np.uint32, ttml.autograd.DataType.BFLOAT8_B, None),
    (
        default_tensor_data,
        np.uint32,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.uint32,
        ttml.autograd.DataType.BFLOAT8_B,
        ttml.Layout.TILE,
    ),
]

"""cases which violate typecast rules codified in TTNN C++"""
typecast_issue_cases = [
    (default_tensor_data, np.float32, None, ttml.Layout.ROW_MAJOR),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.FLOAT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.int32,
        ttml.autograd.DataType.FLOAT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        np.uint32,
        ttml.autograd.DataType.FLOAT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.FLOAT32,
        ttml.Layout.ROW_MAJOR,
    ),
]

"""TODO: multiplication cases which get the wrong answer"""
multiplication_issue_cases = [
    (default_tensor_data, np.float32, ttml.autograd.DataType.UINT32, None),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.float32, ttml.autograd.DataType.UINT32, ttml.Layout.TILE),
    (default_tensor_data, np.int32, ttml.autograd.DataType.UINT32, None),
    (
        default_tensor_data,
        np.int32,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttml.autograd.DataType.UINT32, ttml.Layout.TILE),
    (default_tensor_data, np.uint32, None, None),
    (default_tensor_data, np.uint32, None, ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, None, ttml.Layout.TILE),
    (default_tensor_data, np.uint32, ttml.autograd.DataType.UINT32, None),
    (
        default_tensor_data,
        np.uint32,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.uint32, ttml.autograd.DataType.UINT32, ttml.Layout.TILE),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.TILE,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.UINT32,
        None,
    ),
]

"""TODO: division cases which get the wrong answer"""
division_issue_cases = [
    (default_tensor_data, np.float32, ttml.autograd.DataType.INT32, None),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.INT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.float32, ttml.autograd.DataType.INT32, ttml.Layout.TILE),
    (default_tensor_data, np.float32, ttml.autograd.DataType.UINT32, None),
    (
        default_tensor_data,
        np.float32,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.float32, ttml.autograd.DataType.UINT32, ttml.Layout.TILE),
    (default_tensor_data, np.int32, None, None),
    (default_tensor_data, np.int32, None, ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, None, ttml.Layout.TILE),
    (default_tensor_data, np.int32, ttml.autograd.DataType.INT32, None),
    (
        default_tensor_data,
        np.int32,
        ttml.autograd.DataType.INT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttml.autograd.DataType.INT32, ttml.Layout.TILE),
    (default_tensor_data, np.int32, ttml.autograd.DataType.UINT32, None),
    (
        default_tensor_data,
        np.int32,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.int32, ttml.autograd.DataType.UINT32, ttml.Layout.TILE),
    (default_tensor_data, np.uint32, None, None),
    (default_tensor_data, np.uint32, None, ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, None, ttml.Layout.TILE),
    (default_tensor_data, np.uint32, ttml.autograd.DataType.INT32, None),
    (
        default_tensor_data,
        np.uint32,
        ttml.autograd.DataType.INT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.uint32, ttml.autograd.DataType.INT32, ttml.Layout.TILE),
    (default_tensor_data, np.uint32, ttml.autograd.DataType.UINT32, None),
    (
        default_tensor_data,
        np.uint32,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (default_tensor_data, np.uint32, ttml.autograd.DataType.UINT32, ttml.Layout.TILE),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.UINT32,
        ttml.Layout.TILE,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.UINT32,
        None,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.INT32,
        ttml.Layout.ROW_MAJOR,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.INT32,
        ttml.Layout.TILE,
    ),
    (
        default_tensor_data,
        ml_dtypes.bfloat16,
        ttml.autograd.DataType.INT32,
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
