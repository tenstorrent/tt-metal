import random
import torch
from itertools import product
from functools import partial

from python_api_testing.sweep_tests import generation_funcs


TEST_FIELDNAMES = {
    "pad": [
        "test_name",
        "input_shapes",
        "output_tensor_shape",
        "input_tensor_start",
        "pad_value",
        "data_seed",
        "env_vars",
        "status",
        "test_output",
        "pass/fail",
    ],
    "unpad": [
        "test_name",
        "input_shapes",
        "output_tensor_start",
        "output_tensor_end",
        "data_seed",
        "env_vars",
        "status",
        "test_output",
        "pass/fail",
    ],
}


TT_DNN_TESTS = [
    # Sanity
    "datacopy",
    # Eltwise unary
    "eltwise-exp",
    "eltwise-recip",
    "eltwise-sqrt",
    "eltwise-gelu",
    "eltwise-relu",
    "eltwise-sigmoid",
    "eltwise-log",
    "eltwise-tanh",
    # Eltwise binary
    "eltwise-add",
    "eltwise-sub",
    "eltwise-mul",
    # Matmul
    "matmul",
    "bmm",
    # Broadcast
    "bcast-add-h",
    "bcast-add-w",
    "bcast-add-hw",
    "bcast-sub-h",
    "bcast-sub-w",
    "bcast-sub-hw",
    "bcast-mul-h",
    "bcast-mul-w",
    "bcast-mul-hw",
    # Reduce
    "reduce-max-h",
    "reduce-max-w",
    "reduce-max-hw",
    "reduce-sum-h",
    "reduce-sum-w",
    "reduce-sum-hw",
    # Transpose
    "transpose-wh",
    "transpose-hc",
]


TT_TENSOR_TESTS = ["pad", "unpad"]


def get_test_fieldnames(test_name):
    return TEST_FIELDNAMES.get(
        test_name,
        [
            "test_name",
            "input_shapes",
            "data_seed",
            "env_vars",
            "status",
            "test_output",
            "pass/fail",
        ],
    )


def run_tt_dnn_test(
    ttlib_op,
    pytorch_op,
    input_shapes,
    data_gen_funcs,
    output_comparison_func,
    pcie_slot,
):
    tensor_inputs = []

    for input_shape, data_gen_func in zip(input_shapes, data_gen_funcs):
        tensor_input = data_gen_func(input_shape)
        tensor_inputs.append(tensor_input)

    ttlib_out = ttlib_op(*tensor_inputs, pcie_slot)
    pytorch_out = pytorch_op(*tensor_inputs)

    result, output = output_comparison_func(pytorch_out, ttlib_out)
    return result, output


def run_tt_tensor_test(
    ttlib_op,
    pytorch_op,
    input_shapes,
    data_gen_funcs,
    output_comparison_func,
    pcie_slot,
    **test_args,
):
    tensor_inputs = []

    for input_shape, data_gen_func in zip(input_shapes, data_gen_funcs):
        tensor_input = data_gen_func(input_shape)
        tensor_inputs.append(tensor_input)

    ttlib_out = ttlib_op(*tensor_inputs, pcie_slot, **test_args)
    pytorch_out = pytorch_op(*tensor_inputs, **test_args)

    result, output = output_comparison_func(pytorch_out, ttlib_out)
    return result, output


def run_test_and_save_results(
    results_csv_writer, test_name, input_shapes, data_seed, env_vars, *run_test_args
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
        return test_pass, test_status, test_output, test_result

    if test_name in TT_DNN_TESTS:
        test_pass, test_status, test_output, test_result = _try_except_wrapper(
            run_tt_dnn_test, *run_test_args
        )
        results_csv_writer.writerow(
            {
                "test_name": test_name,
                "input_shapes": input_shapes,
                "data_seed": data_seed,
                "env_vars": env_vars,
                "status": test_status,
                "test_output": test_output,
                "pass/fail": test_result,
            }
        )

    elif test_name in TT_TENSOR_TESTS:
        if test_name == "pad":
            assert len(input_shapes) == 1
            assert len(input_shapes[0]) == 4

            pad_sizes = (64, 64, 64, 64)
            output_tensor_shape = [
                random.randint(input_shapes[0][i], input_shapes[0][i] + pad_sizes[i])
                for i in range(4)
            ]
            input_tensor_start = [
                random.randint(0, output_tensor_shape[i] - input_shapes[0][i])
                for i in range(4)
            ]
            pad_value = random.uniform(-100, 100)
            # Cast to bfloat16 then back to float for exact match
            pad_value = (
                torch.Tensor([pad_value]).to(torch.bfloat16).to(torch.float).item()
            )

            test_args = {
                "output_tensor_shape": output_tensor_shape,
                "input_tensor_start": input_tensor_start,
                "pad_value": pad_value,
            }
            test_pass, test_status, test_output, test_result = _try_except_wrapper(
                run_tt_tensor_test, *run_test_args, **test_args
            )
            results_csv_writer.writerow(
                {
                    "test_name": test_name,
                    "input_shapes": input_shapes,
                    "output_tensor_shape": output_tensor_shape,
                    "input_tensor_start": input_tensor_start,
                    "pad_value": pad_value,
                    "data_seed": data_seed,
                    "env_vars": env_vars,
                    "status": test_status,
                    "test_output": test_output,
                    "pass/fail": test_result,
                }
            )

        elif test_name == "unpad":
            assert len(input_shapes) == 1
            assert len(input_shapes[0]) == 4

            output_tensor_start = [
                random.randint(0, input_shapes[0][i] - 1) for i in range(4)
            ]
            output_tensor_end = [
                random.randint(output_tensor_start[i], input_shapes[0][i] - 1)
                for i in range(4)
            ]

            test_args = {
                "output_tensor_start": output_tensor_start,
                "output_tensor_end": output_tensor_end,
            }
            test_pass, test_status, test_output, test_result = _try_except_wrapper(
                run_tt_tensor_test, *run_test_args, **test_args
            )
            results_csv_writer.writerow(
                {
                    "test_name": test_name,
                    "input_shapes": input_shapes,
                    "output_tensor_start": output_tensor_start,
                    "output_tensor_end": output_tensor_end,
                    "data_seed": data_seed,
                    "env_vars": env_vars,
                    "status": test_status,
                    "test_output": test_output,
                    "pass/fail": test_result,
                }
            )

        else:
            test_pass, test_status, test_output, test_result = _try_except_wrapper(
                run_tt_tensor_test, *run_test_args
            )
            results_csv_writer.writerow(
                {
                    "test_name": test_name,
                    "input_shapes": input_shapes,
                    "data_seed": data_seed,
                    "env_vars": env_vars,
                    "status": test_status,
                    "test_output": test_output,
                    "pass/fail": test_result,
                }
            )

    return test_pass


def shapes_and_datagen(shape_dict, datagen_dict):
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

            dim_ranges = [
                range(start_shape[i], end_shape[i] + interval[i], interval[i])
                for i in range(num_dims)
            ]

            sweeps_generator = list(product(*dim_ranges))
            total_shapes = len(sweeps_generator)
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

            for idx in idx_list:
                shape = list(sweeps_generator[idx])
                yield [shape] * num_shapes, datagen_funcs

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

            sweeps_generator = list(product(*dim_ranges))
            total_shapes = len(sweeps_generator)
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

            for idx in idx_list:
                shape = list(sweeps_generator[idx])
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

            sweeps_generator = list(product(*dim_ranges))
            total_shapes = len(sweeps_generator)
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

            for idx in idx_list:
                b, c, h, w, outer_dim = sweeps_generator[idx]
                shape1 = [b, c, h, w]
                shape2 = [b, c, w, outer_dim]
                if bcast_batch:
                    shape2[:-2] = [1] * len(shape2[:-2])
                yield [shape1, shape2], datagen_funcs

        else:
            raise NotImplementedError("Method {method} is not a valid choice")
