import argparse
import random
from itertools import product
from functools import partial

from python_api_testing.sweep_tests import generation_funcs

fieldnames = [
    "test_name",
    "input_shapes",
    "data_seed",
    "status",
    "test_output",
    "pass/fail",
]


def run_test(
    ttmetal_op,
    pytorch_op,
    input_shapes,
    data_gen_funcs,
    output_comparison_func,
    pcie_slot=0,
):
    tensor_inputs = []

    for input_shape, data_gen_func in zip(input_shapes, data_gen_funcs):
        tensor_input = data_gen_func(input_shape)
        tensor_inputs.append(tensor_input)

    ttmetal_out = ttmetal_op(*tensor_inputs, pcie_slot)
    pytorch_out = pytorch_op(*tensor_inputs)

    result, output = output_comparison_func(pytorch_out, ttmetal_out)
    return result, output


def run_test_and_save_results(
    results_csv_writer, test_name, input_shapes, data_seed, *run_test_args
):
    try:
        test_pass, test_output = run_test(*run_test_args)

        if test_pass:
            test_result = "pass"
        else:
            test_result = "fail"

        test_status = "completed"

    except Exception as err:
        test_status = "error"
        test_result = "fail"
        test_output = err

    results_csv_writer.writerow(
        {
            "test_name": test_name,
            "input_shapes": input_shapes,
            "data_seed": data_seed,
            "status": test_status,
            "test_output": test_output,
            "pass/fail": test_result,
        }
    )


def shapes_and_datagen(shape_dict, datagen_dict):
    num_shapes = shape_dict["num-shapes"]

    # Datagen functions
    if isinstance(datagen_dict, dict):
        datagen_funcs = [
            generation_funcs.gen_func_with_cast(
                partial(
                    getattr(generation_funcs, datagen_dict["function"]),
                    **datagen_dict["args"]
                ),
                generation_funcs.supported_dtypes[datagen_dict.get("dtype", "float32")],
            )
        ] * num_shapes
    elif isinstance(datagen_dict, list):
        datagen_funcs = [
            generation_funcs.gen_func_with_cast(
                partial(
                    getattr(generation_funcs, _datagen_dict["function"]),
                    **_datagen_dict["args"]
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

            for idx in idx_list:
                b, c, h, w, outer_dim = sweeps_generator[idx]
                shape1 = [b, c, h, w]
                shape2 = [b, c, w, outer_dim]
                if bcast_batch:
                    shape2[:-2] = [1] * len(shape2[:-2])
                yield [shape1, shape2], datagen_funcs

        else:
            raise NotImplementedError("Method {method} is not a valid choice")
