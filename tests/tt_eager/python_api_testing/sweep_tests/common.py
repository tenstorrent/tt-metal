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
    def _get_sample_indices(total_shapes, num_shapes):
        if num_samples == "all":
            idx_list = list(range(total_shapes))
        else:
            assert num_samples <= total_shapes
            idx_list = sorted(random.sample(range(total_shapes), num_samples))
        return idx_list

    if "shape-list" in shape_dict:
        # Path for running hardcoded shapes; ignore all other parameters
        shape_list = shape_dict["shape-list"]
        for shape in shape_list:
            assert len(shape) == num_shapes
            yield shape, datagen_funcs

    else:
        start_shape = shape_dict["start-shape"]
        end_shape = shape_dict["end-shape"]
        interval = shape_dict["interval"]

        method = shape_dict.get("method", "default")
        num_samples = shape_dict.get("num-samples", "all")
        bcast_batch = shape_dict.get("bcast-batch", False)

        if method == "default":
            # Sweep across start-shape to end-shape
            # Duplicate the shape num_shapes times
            num_dims = len(start_shape)
            assert len(end_shape) == num_dims

            if not isinstance(interval, list):
                interval = [interval] * num_dims

            assert len(interval) == num_dims
            sample_id = 0

            while sample_id < num_samples:
                shape = []

                for i in range(num_dims):
                    x = random.randint(start_shape[i], end_shape[i])
                    shape.append(align_to_interval(x, start_shape[i], interval[i]))

                input_shapes = [shape] * num_shapes
                args = test_args_gen(input_shapes, test_tt_dtypes, test_tt_layouts, test_buffer_types)

                if shape_dict.get("args-sampling-strategy", "all") == "random":
                    generated_test_args = random.choice(args)
                    args = [generated_test_args]

                for generated_test_args in args:
                    sample_id += 1
                    yield input_shapes, datagen_funcs, generated_test_args

        elif method in ("bcast_h", "bcast_w", "bcast_hw"):
            # Like default, but yield a specific second bcast_shape
            assert num_shapes == 2

            num_dims = len(start_shape)
            assert len(end_shape) == num_dims

            if not isinstance(interval, list):
                interval = [interval] * num_dims

            assert len(interval) == num_dims

            dim_ranges = [
                range(start_shape[i], end_shape[i] + interval[i], interval[i])
                for i in range(num_dims)
            ]

            sweeps_generator = product(*dim_ranges)
            total_shapes = functools.reduce(operator.mul, map(len, dim_ranges), 1)
            idx_list = _get_sample_indices(total_shapes, num_shapes)

            if "split" in shape_dict:
                split_params = shape_dict["split"]
                assert len(split_params) == 2

                split_id, num_splits = split_params
                assert len(idx_list) % num_splits == 0
                samples_per_split = len(idx_list) // num_splits
                idx_list = idx_list[
                    (split_id - 1) * samples_per_split : split_id * samples_per_split
                ]

            idx_list = deque(idx_list)
            for i, shape in enumerate(sweeps_generator):
                if i == idx_list[0]:
                    idx_list.popleft()
                    if len(idx_list) == 0:
                        break
                else:
                    continue
                shape = list(shape)
                b, c, h, w = shape
                if method == "bcast_h":
                    bcast_shape = [b, c, 1, w]
                elif method == "bcast_w":
                    bcast_shape = [b, c, h, 1]
                elif method == "bcast_hw":
                    bcast_shape = [b, c, 1, 1]
                if bcast_batch:
                    bcast_shape[:-2] = [1] * len(bcast_shape[:-2])
                yield [shape, bcast_shape], datagen_funcs

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

            for _ in range(num_samples):
                shape1 = []

                for i in range(num_dims):
                    x = random.randint(shape1_start[i], shape1_end[i])
                    shape1.append(align_to_interval(x, shape1_start[i], interval[i]))

                x = random.randint(shape2_start[-1], shape2_end[-1])
                outer_dim = align_to_interval(x, shape2_start[-1], interval[-1])

                shape2 = [shape1[0], shape1[1], shape1[3], outer_dim]
                yield [shape1, shape2], datagen_funcs


        elif method == "layernorm":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used

            shape1_start = start_shape
            shape1_end = end_shape
            num_dims = 4

            assert (
                len(shape1_start)
                == num_dims
            )

            for _ in range(num_samples):
                shape1 = []

                for i in range(num_dims):
                    x = random.randint(shape1_start[i], shape1_end[i])
                    shape1.append(align_to_interval(x, shape1_start[i], interval[i]))

                normalized_shape = [1, 1, 1, shape1[3]]

                yield [shape1, normalized_shape, normalized_shape], datagen_funcs


        elif method == "add_layernorm":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used

            shape1_start = start_shape
            shape1_end = end_shape
            num_dims = 4

            assert (
                len(shape1_start)
                == num_dims
            )

            for _ in range(num_samples):
                shape1 = []

                for i in range(num_dims):
                    x = random.randint(shape1_start[i], shape1_end[i])
                    shape1.append(align_to_interval(x, shape1_start[i], interval[i]))

                normalized_shape = [1, 1, 1, shape1[3]]

                yield [shape1, shape1, normalized_shape, normalized_shape], datagen_funcs

        elif method == "conv":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used

            shape1_start = start_shape
            shape1_end = end_shape
            num_dims = 4

            assert (
                len(shape1_start)
                == num_dims
            )

            for _ in range(num_samples):
                shape1 = []

                for i in range(num_dims):
                    x = random.randint(shape1_start[i], shape1_end[i])
                    shape1.append(align_to_interval(x, shape1_start[i], interval[i]))

                conv_shape = [0, 0, 0, 0]
                conv_shape[0] = 1
                conv_shape[1] = shape1[1]
                conv_shape[2] = random.randint(1, 4)
                conv_shape[3] = random.randint(1, 4)

                yield [shape1, conv_shape], datagen_funcs

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

            dim_ranges = [
                range(shape1_start[i], shape1_end[i] + interval[i], interval[i])
                for i in range(num_dims)
            ]
            # Add outer dim from last dim of second shape
            dim_ranges.append(
                range(shape2_start[-1], shape2_end[-1] + interval[-1], interval[-1])
            )

            sweeps_generator = product(*dim_ranges)
            total_shapes = functools.reduce(operator.mul, map(len, dim_ranges), 1)
            idx_list = _get_sample_indices(total_shapes, num_shapes)

            if "split" in shape_dict:
                split_params = shape_dict["split"]
                assert len(split_params) == 2

                split_id, num_splits = split_params
                assert len(idx_list) % num_splits == 0
                samples_per_split = len(idx_list) // num_splits
                idx_list = idx_list[
                    (split_id - 1) * samples_per_split : split_id * samples_per_split
                ]

            idx_list = deque(idx_list)
            for i, shape in enumerate(sweeps_generator):
                if i == idx_list[0]:
                    idx_list.popleft()
                    if len(idx_list) == 0:
                        break
                else:
                    continue
                shape = list(shape)
                b, c, h, w, outer_dim = shape
                shape1 = [b, c, h, w]
                shape2 = [1, 1, outer_dim, w]

                shapes = [shape1, shape2]
                if num_shapes == 3:
                    shape3 = [1, 1, 1, outer_dim]
                    shapes.append(shape3)

                yield shapes, datagen_funcs

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
