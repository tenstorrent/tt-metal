# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random
import os
import torch
from itertools import product
from functools import partial
import functools
import math
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


def run_tt_lib_test(
    tt_lib_op,
    pytorch_op,
    input_shapes,
    data_gen_funcs,
    output_comparison_func,
    test_args,
    device=None,
    plot_func=None,
):
    logger.debug(f"Running with args: {test_args}")

    tensor_inputs = []

    for input_shape, data_gen_func in zip(input_shapes, data_gen_funcs):
        tensor_input = data_gen_func(input_shape)
        tensor_inputs.append(tensor_input)

    tt_lib_out = tt_lib_op(*tensor_inputs, device=device, **test_args)
    pytorch_out = pytorch_op(*tensor_inputs, **test_args)

    result, output = output_comparison_func(pytorch_out, tt_lib_out)

    logger.info(f"{result} {output}")

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

    test_pass, test_status, test_output, test_result = _try_except_wrapper(run_tt_lib_test, *run_test_args)

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


def shapes_and_datagen(
    shape_dict,
    datagen_dict,
    test_args_gen,
    test_tt_dtypes,
    test_tt_layouts,
    test_buffer_types,
    sanitize_args=True,
    coregrid=[],
):
    num_shapes = shape_dict["num-shapes"]

    # Helper
    def _gen_args(input_shapes):
        args = test_args_gen(
            input_shapes,
            test_tt_dtypes,
            test_tt_layouts,
            test_buffer_types,
            do_sanitize_args=sanitize_args,
            coregrid=coregrid,
        )
        args = list(args)

        # Default "args-sampling-strategy" is "all". If random is not specified test will be run for all generated args
        if shape_dict.get("args-sampling-strategy", "random") == "random" and len(args) > 0:
            generated_test_args = random.choice(args)
            args = [generated_test_args]

        result = []

        # Add datagen_funcs
        for generated_test_args in args:
            datagen_funcs = []

            for i in range(len(generated_test_args["dtype"])):
                _datagen_dict = datagen_dict[i] if isinstance(datagen_dict, list) else datagen_dict

                datagen_funcs.append(
                    generation_funcs.gen_func_with_cast_tt(
                        partial(
                            getattr(generation_funcs, _datagen_dict["function"]),
                            **_datagen_dict["args"],
                        ),
                        generated_test_args["dtype"][i],
                    )
                )

            result.append((datagen_funcs, generated_test_args))

        return result

    if "shape-list" in shape_dict:
        # Path for running hardcoded shapes; ignore all other parameters
        shape_list = shape_dict["shape-list"]
        for shape in shape_list:
            assert len(shape) == num_shapes

            for datagen_funcs, generated_test_args in _gen_args(shape):
                yield shape, datagen_funcs, generated_test_args
    else:
        start_shape = shape_dict["start-shape"]
        end_shape = shape_dict["end-shape"]
        interval = shape_dict["interval"]

        num_dims_settings = shape_dict.get("num-dims", [])
        method = shape_dict.get("method", "default")
        num_samples = shape_dict.get("num-samples", "all")

        # Helper method. Generates shapes and arguments. Used in various methods.
        def _gen_shapes_and_args(start_shape, end_shape, interval, shape_transformator):
            all_num_dims = [len(start_shape)] if len(num_dims_settings) == 0 else num_dims_settings

            if num_samples == "all":
                for num_dims in all_num_dims:
                    dim_ranges = [
                        range(start_shape[i], end_shape[i] + interval[i], interval[i]) for i in range(num_dims)
                    ]

                    for shape in product(*dim_ranges):
                        input_shapes = shape_transformator(shape)

                        for datagen_funcs, generated_test_args in _gen_args(input_shapes):
                            yield input_shapes, datagen_funcs, generated_test_args

            else:
                sample_id = 0

                while sample_id < num_samples:
                    for num_dims in all_num_dims:
                        shape = []

                        for i in range(-num_dims, 0):
                            x = random.randint(start_shape[i], end_shape[i])
                            shape.append(align_to_interval(x, start_shape[i], interval[i]))

                        input_shapes = shape_transformator(shape)
                        args = _gen_args(input_shapes)

                        if len(args) == 0:
                            sample_id += 1
                            continue

                        for datagen_funcs, generated_test_args in args:
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

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _default_shape_tr
            ):
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

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_bcast_shape
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "matmul":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used
            assert len(start_shape) == len(end_shape) == 2
            assert num_shapes == 2

            shape1_start, shape2_start = start_shape
            shape1_end, shape2_end = end_shape
            num_dims = 4

            assert len(shape1_start) == len(shape1_end) == len(shape2_start) == len(shape2_end) == num_dims

            shape1_start = shape1_start + [shape2_start[-1]]
            shape1_end = shape1_end + [shape2_end[-1]]

            def _gen_matmul_shapes(shape):
                shape1 = [shape[0], shape[1], shape[2], shape[3]]
                shape2 = [shape[0], shape[1], shape[3], shape[4]]
                return [shape1, shape2]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                shape1_start, shape1_end, interval, _gen_matmul_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "bert_large_mul":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used
            assert len(start_shape) == 1
            assert len(end_shape) == 1

            def _gen_bert_large_shapes(shape):
                a_shape = [9, 1, 384, 1024]
                b_shape = [1, 1, 1024, 4096]
                bias_shape = [1, 1, 1, 4096]
                return [a_shape, b_shape, bias_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_bert_large_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "concat":
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            shape_start = start_shape + start_shape
            shape_end = end_shape + end_shape
            new_interval = interval + interval

            def _concat_shapes(shape):
                a_shape = [shape[0], shape[1], shape[2], shape[3]]
                b_shape = [shape[0], shape[1], shape[2], shape[3]]

                dim = random.randint(0, 3)

                # Concatinating dim is different
                b_shape[dim] = shape[4 + dim]

                return [a_shape, b_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                shape_start, shape_end, new_interval, _concat_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "layernorm":
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_layernorm_shapes(shape):
                normalized_shape = [1, 1, 1, shape[3]]
                return [shape, normalized_shape, normalized_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_layernorm_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "add_layernorm":
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_add_layernorm_shapes(shape):
                normalized_shape = [1, 1, 1, shape[3]]
                return [shape, shape, normalized_shape, normalized_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_add_layernorm_shapes
            ):
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

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_conv_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "embeddings":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_embeddings_shapes(shape):
                batch_size = shape[0]
                num_rows = shape[1]
                num_embeddings = shape[2]
                embedding_dim = shape[3]

                input_rows_shape = [batch_size, 1, 1, num_rows]
                weights_shape = [1, 1, num_embeddings, embedding_dim]

                return [input_rows_shape, weights_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_embeddings_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "embeddings-bw":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_embeddings_shapes(shape):
                batch_size = shape[0]
                # num_rows = shape[3]
                num_embeddings = shape[2]
                embedding_dim = shape[3]
                no_of_embeddings = shape[1] * shape[2]

                input_rows_shape = [batch_size, 1, 1, no_of_embeddings]
                weights_shape = [batch_size, 1, no_of_embeddings, embedding_dim]
                grad_shape = [1, 1, batch_size * no_of_embeddings, embedding_dim]

                return [grad_shape, input_rows_shape, weights_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_embeddings_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "rmsnorm":
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_rmsnorm_shapes(shape):
                normalized_shape = [1, 1, 1, shape[3]]
                return [shape, normalized_shape, normalized_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_rmsnorm_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "complex_bin":
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_complex_bin_shapes(shape):
                complex_shape = [1, shape[1], shape[2], shape[3]]
                return [complex_shape, complex_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_complex_bin_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "linear":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used
            assert len(start_shape) == len(end_shape) == 2
            assert num_shapes == 2 or num_shapes == 3

            shape1_start, shape2_start = start_shape
            shape1_end, shape2_end = end_shape
            num_dims = 4

            assert len(shape1_start) == len(shape1_end) == len(shape2_start) == len(shape2_end) == num_dims

            if not isinstance(interval, list):
                interval = [interval] * (num_dims + 1)

            assert len(interval) == (num_dims + 1)

            shape1_start = shape1_start + [shape2_start[-1]]
            shape1_end = shape1_end + [shape2_end[-1]]

            def _gen_linear_shapes(shape):
                b, c, h, w, outer_dim = shape
                shape1 = [b, c, h, w]
                shape2 = [1, 1, outer_dim, w]
                shapes = [shape1, shape2]

                if num_shapes == 3:
                    shape3 = [1, 1, 1, outer_dim]
                    shapes.append(shape3)

                return shapes

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                shape1_start, shape1_end, interval, _gen_linear_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "ttnn-linear":
            # Only supports dim = 5;
            assert len(start_shape) == 5
            assert len(end_shape) == 5

            def _gen_tt_nn_linear_shapes(shape):
                b, c, h, w, outer_dim = shape

                b, c, h, w, outer_dim = shape
                shape1 = [b, c, h, w]
                shape2 = [1, 1, outer_dim, w]
                shapes = [shape1, shape2]

                if num_shapes == 3:
                    shape3 = [1, 1, 1, outer_dim]
                    shapes.append(shape3)

                return shapes

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_linear_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "ttnn-embeddings":
            # Only supports dim = 4;
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_tt_nn_embeddings_shapes(shape):
                batch_size = shape[0]
                num_rows = shape[1]
                num_embeddings = shape[2]
                embedding_dim = shape[3]

                input_rows_shape = [batch_size, num_rows]
                weights_shape = [num_embeddings, embedding_dim]

                return [input_rows_shape, weights_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_embeddings_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "ttnn-matmul":
            # start-shape and end-shape are lists of two shapes
            # Only supports dim = 4; for the second shape, only the last dim is used

            def _gen_tt_nn_matmul_shapes(shape):
                if len(shape) == 5:
                    n, c, h, w, x = shape
                    shape_type = random.randint(0, 3)

                    if shape_type == 0:
                        shape1 = [n, c, h, w]
                        shape2 = [n, c, w, x]
                    elif shape_type == 1:
                        shape1 = [n, c, h, w]
                        shape2 = [1, 1, w, x]
                    elif shape_type == 2:
                        shape1 = [n, c, h, w]
                        shape2 = [1, w, x]
                    else:
                        shape1 = [n, c, h, w]
                        shape2 = [w, x]
                elif len(shape) == 4:
                    c, h, w, x = shape
                    shape_type = random.randint(0, 2)

                    if shape_type == 0:
                        shape1 = [c, h, w]
                        shape2 = [c, w, x]
                    elif shape_type == 1:
                        shape1 = [c, h, w]
                        shape2 = [1, w, x]
                    else:
                        shape1 = [c, h, w]
                        shape2 = [w, x]
                elif len(shape) == 3:
                    m, k, n = shape
                    shape1 = [m, k]
                    shape2 = [k, n]
                elif len(shape) == 2:
                    k, n = shape
                    shape1 = [1, k]
                    shape2 = [k, n]
                else:
                    assert False, f"Bad shape for tt nn matmult sweep {shape}"

                return [shape1, shape2]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_matmul_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "ttnn-layernorm":
            assert len(start_shape) == 2
            assert len(end_shape) == 2

            def _gen_tt_nn_layernorm_shapes(shape):
                input_shape = [shape[0], shape[1]]
                weights_shape = [shape[1]]
                bias_shape = [shape[1]]

                return [input_shape, weights_shape, bias_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_layernorm_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "ttnn-layernorm_residual":
            assert len(start_shape) == 2
            assert len(end_shape) == 2

            def _gen_tt_nn_layernorm_res_shapes(shape):
                input_shape = [shape[0], shape[1]]
                residual_shape = [shape[0], shape[1]]
                weights_shape = [shape[1]]
                bias_shape = [shape[1]]

                return [input_shape, residual_shape, weights_shape, bias_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_layernorm_res_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "ttnn-layernorm_noweights":
            assert len(start_shape) == 2
            assert len(end_shape) == 2

            def _gen_tt_nn_layernorm_nw_shapes(shape):
                input_shape = [shape[0], shape[1]]

                return [input_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_layernorm_nw_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "ttnn-softmax":
            assert len(start_shape) == 4
            assert len(end_shape) == 4

            def _gen_tt_nn_softmax_shapes(shape):
                input_shape = [shape[0], shape[1], shape[2], shape[3]]
                mask_shape = [shape[0], shape[1], 32, shape[3]]
                return [input_shape, mask_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_softmax_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "ttnn-rmsnorm":
            assert len(start_shape) == 2
            assert len(end_shape) == 2

            def _gen_tt_nn_rmsnorm_shapes(shape):
                input_shape = [shape[0], shape[1]]
                weights_shape = [1, shape[1]]

                return [input_shape, weights_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_rmsnorm_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "tt_nn-bcast":

            def _gen_tt_nn_bcast_shapes(shape):
                shape_type = random.randint(0, 2)
                second_shape = shape.copy()

                if shape_type == 0:
                    second_shape[-2] = 1
                    second_shape[-1] = 1
                elif shape_type == 1:
                    second_shape[-2] = shape[-2]
                    second_shape[-1] = 1
                elif shape_type == 2:
                    second_shape[-2] = 1
                    second_shape[-1] = shape[-1]
                # elif shape_type == 2:
                #     second_shape = [shape[-2], shape[-1]]
                # elif shape_type == 3:
                #     second_shape = [shape[-3], shape[-2], shape[-1]]
                else:
                    second_shape = shape

                return [shape, second_shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_tt_nn_bcast_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "concat_bw":

            def _gen_concat_bw_shapes(shape):
                shape1 = []
                shape2 = []
                grad_shape = []

                num_dims = len(shape)

                dim = random.randint(0, num_dims - 1)

                for i in range(num_dims):
                    shape1.append(shape[i])
                    shape2.append(shape[i])
                    if i == dim:
                        grad_shape.append(shape1[i] + shape2[i])
                    else:
                        grad_shape.append(shape[i])

                return [grad_shape, shape1, shape2]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_concat_bw_shapes
            ):
                yield shapes, datagen_funcs, test_args

        elif method == "topk":
            # at the moment, topk only works on last dim
            # last dim must be a multiple of 64 and a pow of 2
            def _gen_topk_shapes(shape):
                num_dims = len(shape)
                last_dim = shape[num_dims - 1]
                if not (last_dim & (last_dim - 1) == 0) and last_dim != 0:
                    last_dim = 2 ** math.ceil(math.log2(last_dim))
                    last_dim = last_dim + last_dim % 64

                shape[num_dims - 1] = last_dim

                return [shape]

            for shapes, datagen_funcs, test_args in _gen_shapes_and_args(
                start_shape, end_shape, interval, _gen_topk_shapes
            ):
                yield shapes, datagen_funcs, test_args

        else:
            raise NotImplementedError("Method {method} is not a valid choice")


def set_slow_dispatch_mode(set_var):
    prev_value = os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)

    if set_var != "" and set_var is not None:
        os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = set_var
        logger.info("Setting slow dispatch mode")
    else:
        logger.info("Setting fast dispatch mode")

    return prev_value
