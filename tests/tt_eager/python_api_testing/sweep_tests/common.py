# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
import os
import torch
from itertools import product
from functools import partial
import functools
import operator
from collections import deque
from loguru import logger

from tests.tt_eager.python_api_testing.sweep_tests import generation_funcs
import pytest

def get_test_fieldnames(test_args=[]):
    return [
        "test_name",
        "input_shapes",
        *test_args,
        "data_seed",
        "env_vars",
        "status",
        "test_output",
        "pass/fail",
    ]

# TODO: Deprecate pcie_slot arg after run_pytorch_test is uplifted to pytest and device fixture
def run_tt_lib_test(
    tt_lib_op,
    pytorch_op,
    input_shapes,
    data_gen_funcs,
    output_comparison_func,
    device_id,
    test_args,
    device=None,
    plot_func=None,
):
    logger.info(f"Running with args: {test_args}")

    tensor_inputs = []

    for input_shape, data_gen_func in zip(input_shapes, data_gen_funcs):
        tensor_input = data_gen_func(input_shape)
        tensor_inputs.append(tensor_input)

    tt_lib_out = tt_lib_op(
        *tensor_inputs, device_id=device_id, device=device, **test_args
    )
    pytorch_out = pytorch_op(*tensor_inputs, **test_args)

    result, output = output_comparison_func(pytorch_out, tt_lib_out)

    if plot_func is not None:
        test_name = str(pytorch_op).split()[1]
        plot_func(test_name, *tensor_inputs, pytorch_out, tt_lib_out)
    return result, output


def run_test_and_save_results(
    results_csv_writer,
    test_name,
    input_shapes,
    data_seed,
    env_vars,
    test_args,
    *run_test_args,
):
    def _try_except_wrapper(func, *args, **kwargs):
        try:
            test_pass, test_output = func(*args, **kwargs)
            if test_pass:
                test_result = "pass"
            else:
                test_result = "fail"
            test_status = "completed"

        except Exception as err:
            test_pass = False
            test_status = "error"
            test_output = err
            test_result = "fail"

        # test_pass and test_output comes from actual test
        # test_status is completed/error (ie. runtime)
        # test_result is pass/fail depending on test_pass bool
        # test_args is a dict of specific args required by ops to run
        return test_pass, test_status, test_output, test_result

    test_pass, test_status, test_output, test_result = _try_except_wrapper(
        run_tt_lib_test, *run_test_args
    )

    if results_csv_writer is not None:
        results_csv_writer.writerow(
            {
                "test_name": test_name,
                "input_shapes": input_shapes,
                "args": test_args,
                "data_seed": data_seed,
                "env_vars": env_vars,
                "status": test_status,
                "test_output": test_output,
                "pass/fail": test_result,
            }
        )

    return test_pass


def align_to_interval(x, start_val, interval):
    dx = x - start_val
    dx = (dx // interval) * interval
    return start_val + dx


def get_all_shapes(start_shape, end_shape, interval):
    num_dims = len(start_shape)

    dim_ranges = [
        range(start_shape[i], end_shape[i] + interval[i], interval[i])
        for i in range(num_dims)
    ]

    return list(product(*dim_ranges))


def get_random_shape(start_shape, end_shape, interval):
    num_dims = len(start_shape)
    shape = []

    for i in range(num_dims):
        x = random.randint(start_shape[i], end_shape[i])
        shape.append(align_to_interval(x, start_shape[i], interval[i]))

    return shape


def shapes_and_datagen(shape_dict, datagen_dict, test_args_gen, test_tt_dtypes, test_tt_layouts, test_buffer_types):
    num_shapes = shape_dict["num-shapes"]

    # Datagen functions
    if isinstance(datagen_dict, dict):
        datagen_funcs = [
            generation_funcs.gen_func_with_cast(
                partial(
                    getattr(generation_funcs, datagen_dict["function"]),
                    **datagen_dict["args"],
                ),
                generation_funcs.supported_dtypes[datagen_dict.get("dtype", "float32")],
                datagen_dict.get("tilize", False),
            )
        ] * num_shapes
    elif isinstance(datagen_dict, list):
        datagen_funcs = [
            generation_funcs.gen_func_with_cast(
                partial(
                    getattr(generation_funcs, _datagen_dict["function"]),
                    **_datagen_dict["args"],
                ),
                generation_funcs.supported_dtypes[datagen_dict.get("dtype", "float32")],
                datagen_dict.get("tilize", False),
            )
            for _datagen_dict in datagen_dict
        ]

    # Helper
    def _gen_args(input_shapes):
        args = test_args_gen(input_shapes, test_tt_dtypes, test_tt_layouts, test_buffer_types)
        args = list(args)

        if shape_dict.get("args-sampling-strategy", "random") == "random" and len(args) > 0:
            generated_test_args = random.choice(args)
            args = [generated_test_args]

        return args

    if "shape-list" in shape_dict:
        # Path for running hardcoded shapes; ignore all other parameters
        shape_list = shape_dict["shape-list"]
        for shape in shape_list:
            assert len(shape) == num_shapes

            for generated_test_args in _gen_args(shape):
                yield shape, datagen_funcs, generated_test_args
    else:
        start_shape = shape_dict["start-shape"]
        end_shape = shape_dict["end-shape"]
        interval = shape_dict["interval"]

        method = shape_dict.get("method", "default")
        num_samples = shape_dict.get("num-samples", "all")

        # Helper method. Generates shaeps and argument. Used in various methods.
        def _gen_shapes_and_args(start_shape, end_shape, interval, shape_transformator):
            num_dims = len(start_shape)

            if num_samples == "all":
                dim_ranges = [
                    range(start_shape[i], end_shape[i] + interval[i], interval[i])
                    for i in range(num_dims)
                ]

                for shape in product(*dim_ranges):
                    input_shapes = shape_transformator(shape)

                    for generated_test_args in _gen_args(input_shapes):
                        yield input_shapes, datagen_funcs, generated_test_args

            else:
                sample_id = 0
                while sample_id < num_samples:
                    shape = []

                    for i in range(num_dims):
                        x = random.randint(start_shape[i], end_shape[i])
                        shape.append(align_to_interval(x, start_shape[i], interval[i]))

                    input_shapes = shape_transformator(shape)
                    args = _gen_args(input_shapes)

                    if len(args) == 0:
                        sample_id += 1
                        continue

                    for generated_test_args in args:
                        sample_id += 1
                        yield input_shapes, datagen_funcs, generated_test_args

        if method == "default":
            # Sweep across start-shape to end-shape
            # Duplicate the shape num_shapes times
            num_dims = len(start_shape)
            assert len(end_shape) == num_dims

            if not isinstance(interval, list):
                interval = [interval] * num_dims

            assert len(interval) == num_dims

            def _default_shape_tr(shape):
                return [list(shape)] * num_shapes

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(start_shape, end_shape, interval, _default_shape_tr):
                yield shapes, datagen_funcs, test_args

        elif method in ("bcast_h", "bcast_w", "bcast_hw"):
            # Like default, but yield a specific second bcast_shape
            assert num_shapes == 2

            num_dims = len(start_shape)
            bcast_batch = shape_dict.get("bcast-batch", False)

            assert len(end_shape) == num_dims

            if not isinstance(interval, list):
                interval = [interval] * num_dims

            assert len(interval) == num_dims

            def _gen_bcast_shape(shape):
                b, c, h, w = shape
                if method == "bcast_h":
                    bcast_shape = [b, c, 1, w]
                elif method == "bcast_w":
                    bcast_shape = [b, c, h, 1]
                elif method == "bcast_hw":
                    bcast_shape = [b, c, 1, 1]
                if bcast_batch:
                    bcast_shape[:-2] = [1] * len(bcast_shape[:-2])
                return [shape, bcast_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(start_shape, end_shape, interval, _gen_bcast_shape):
                yield shapes, datagen_funcs, test_args

        elif method == "matmul":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used
            assert len(start_shape) == len(end_shape) == 2
            assert num_shapes == 2

            shape1_start, shape2_start = start_shape
            shape1_end, shape2_end = end_shape
            num_dims = 4

            assert (
                len(shape1_start)
                == len(shape1_end)
                == len(shape2_start)
                == len(shape2_end)
                == num_dims
            )

            shape1_start.append(shape2_start[-1])
            shape1_end.append(shape2_end[-1])

            def _gen_matmul_shapes(shape):
                shape1 = [shape[0], shape[1], shape[2], shape[3]]
                shape2 = [shape[0], shape[1], shape[3], shape[4]]
                return [shape1, shape2]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(shape1_start, shape1_end, interval, _gen_matmul_shapes):
                yield shapes, datagen_funcs, test_args

        elif method == "layernorm":
            assert (len(start_shape) == 4)
            assert (len(end_shape) == 4)

            def _gen_layernorm_shapes(shape):
                normalized_shape = [1, 1, 1, shape[3]]
                return [shape, normalized_shape, normalized_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(start_shape, end_shape, interval, _gen_layernorm_shapes):
                yield shapes, datagen_funcs, test_args

        elif method == "add_layernorm":
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_add_layernorm_shapes(shape):
                normalized_shape = [1, 1, 1, shape[3]]
                return [shape, shape, normalized_shape, normalized_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(start_shape, end_shape, interval, _gen_add_layernorm_shapes):
                yield shapes, datagen_funcs, test_args

        elif method == "conv":
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_conv_shapes(shape):
                conv_shape = [0, 0, 0, 0]
                conv_shape[0] = 1
                conv_shape[1] = shape[1]
                conv_shape[2] = random.randint(1, 4)
                conv_shape[3] = random.randint(1, 4)

                return [shape, conv_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(start_shape, end_shape, interval, _gen_conv_shapes):
                yield shapes, datagen_funcs, test_args

        elif method == "linear":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used
            assert len(start_shape) == len(end_shape) == 2
            assert num_shapes == 2 or num_shapes == 3

            shape1_start, shape2_start = start_shape
            shape1_end, shape2_end = end_shape

            num_dims = 4
            assert (
                len(shape1_start)
                == len(shape1_end)
                == len(shape2_start)
                == len(shape2_end)
                == num_dims
            )

            if not isinstance(interval, list):
                interval = [interval] * (num_dims + 1)

            assert len(interval) == (num_dims + 1)

            shape1_start.append(shape2_start[-1])
            shape1_end.append(shape2_end[-1])

            def _gen_linear_shapes(shape):
                b, c, h, w, outer_dim = shape
                shape1 = [b, c, h, w]
                shape2 = [1, 1, outer_dim, w]
                shapes = [shape1, shape2]

                if num_shapes == 3:
                    shape3 = [1, 1, 1, outer_dim]
                    shapes.append(shape3)

                return shapes

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(shape1_start, shape1_end, interval, _gen_linear_shapes):
                yield shapes, datagen_funcs, test_args

        else:
            raise NotImplementedError("Method {method} is not a valid choice")


ARCH_NAME = os.environ.get("ARCH_NAME",os.environ.get("TT_ARCH_NAME","")).lower()

def is_wormhole_b0():
    return "wormhole_b0" in ARCH_NAME

def is_grayskull():
    return "grayskull" in ARCH_NAME

def skip_for_wormhole_b0(fn):
    @pytest.mark.skipif(is_wormhole_b0(),reason="not working for Wormhole B0")
    @functools.wraps(fn)
    def _caller_fn(*args,**kwargs):
        return fn(*args,**kwargs)
    return _caller_fn

def skip_for_grayskull(fn):
    @pytest.mark.skipif(is_grayskull(),reason="not working for Grayskull")
    @functools.wraps(fn)
    def _caller_fn(*args,**kwargs):
        return fn(*args,**kwargs)
    return _caller_fn
