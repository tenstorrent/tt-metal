# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/build/tt-train/sources/ttml')
import _ttml  # noqa: E402


def supported_autograd_types_except(*except_types):
    return tuple(
        data_type
        for data_type in _ttml.autograd.DataType.__members__.values()
        if data_type
        not in (
            _ttml.autograd.DataType.INVALID,
            _ttml.autograd.DataType.BFLOAT8_B,
            _ttml.autograd.DataType.BFLOAT4_B,
            _ttml.autograd.DataType.UINT8,
            _ttml.autograd.DataType.UINT16,
        )
        + tuple(except_type for except_type in except_types)
    )


def do_test_numpy_autograd_conversion(
    tensor_data, numpy_type, autograd_type, layout, expect_type_exception, expect_runtime_exception
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
                autograd_tensor = _ttml.autograd.Tensor.from_numpy(numpy_tensor, layout=layout, new_type=autograd_type)
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
        else:
            try:
                autograd_tensor = _ttml.autograd.Tensor.from_numpy(numpy_tensor, new_type=autograd_type)
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
    else:
        if layout:
            try:
                autograd_tensor = _ttml.autograd.Tensor.from_numpy(numpy_tensor, layout=layout)
            except TypeError as e:
                type_error = handle_error(e, expect_type_exception, type_error)
            except RuntimeError as e:
                runtime_error = handle_error(e, expect_runtime_exception, runtime_error)
        else:
            try:
                autograd_tensor = _ttml.autograd.Tensor.from_numpy(numpy_tensor)
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
all_test_cases = [
    (default_tensor_data, np.float32, None, None),
    (default_tensor_data, np.float32, None, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, None, _ttml.Layout.TILE),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT16, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.TILE),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT4_B, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT8_B, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.FLOAT32, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.TILE),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.INT32, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.INT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.INT32, _ttml.Layout.TILE),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.UINT32, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.UINT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.UINT32, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, None, None),
    (default_tensor_data, np.int32, None, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, None, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT16, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT16, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT4_B, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT4_B, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT8_B, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.FLOAT32, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.INT32, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.INT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.INT32, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.UINT32, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.UINT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.UINT32, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, None, None),
    (default_tensor_data, np.uint32, None, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, None, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT16, None),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT16, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT4_B, None),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT8_B, None),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.FLOAT32, None),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.INT32, None),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.INT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.INT32, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.UINT32, None),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.UINT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.UINT32, _ttml.Layout.TILE),
]


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
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT4_B, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT4_B, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT8_B, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT8_B, None),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT4_B, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT8_B, None),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT4_B, None),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT4_B, _ttml.Layout.TILE),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT8_B, None),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.BFLOAT8_B, _ttml.Layout.TILE),
]

"""cases which violate typecast rules codified in TTNN C++"""
typecast_issue_cases = [
    (default_tensor_data, np.float32, None, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.float32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.int32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.ROW_MAJOR),
    (default_tensor_data, np.uint32, _ttml.autograd.DataType.FLOAT32, _ttml.Layout.ROW_MAJOR),
]


@pytest.mark.parametrize(
    "tensor_data, numpy_type, autograd_type, layout",
    cases_except(
        all_test_cases,
        unsupported_format_cases,
        typecast_issue_cases,
    ),
)
def test_numpy_autograd_conversion(tensor_data, numpy_type, autograd_type, layout):
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
