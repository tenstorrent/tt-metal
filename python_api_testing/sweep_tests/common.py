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
    ll_buda_op,
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

    ll_buda_out = ll_buda_op(pcie_slot, *tensor_inputs)
    pytorch_out = pytorch_op(*tensor_inputs)

    result, output = output_comparison_func(pytorch_out, ll_buda_out)
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
    start_shape = shape_dict["start-shape"]
    end_shape = shape_dict["end-shape"]
    interval = shape_dict["interval"]
    num_shapes = shape_dict["num-shapes"]

    method = shape_dict.get("method", "default")
    num_samples = shape_dict.get("num-samples", "all")

    # Datagen functions
    if isinstance(datagen_dict, dict):
        datagen_funcs = [
            partial(
                getattr(generation_funcs, datagen_dict["function"]),
                **datagen_dict["args"]
            )
        ] * num_shapes
    elif isinstance(datagen_dict, list):
        datagen_funcs = [
            partial(
                getattr(generation_funcs, _datagen_dict["function"]),
                **_datagen_dict["args"]
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
            interval = [interval] * num_dims

        assert len(interval) == num_dims

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
            yield [shape1, shape2], datagen_funcs

    else:
        raise NotImplementedError("Method {method} is not a valid choice")


# Used by old standalone pytorch test scripts
def shape_sweeps(
    start_shape, end_shape, interval=32, square=False, num_shapes=1, num_samples=None
):
    assert len(start_shape) == 4
    assert len(end_shape) == 4

    b_start, c_start, h_start, w_start = start_shape
    b_end, c_end, h_end, w_end = end_shape

    assert b_start == b_end
    assert c_start == c_end

    if square:
        assert h_start == w_start
        assert h_end == w_end

    if square:
        sweeps_generator = list(range(h_start, h_end + interval, interval))
    else:
        h_range = range(h_start, h_end + interval, interval)
        w_range = range(w_start, w_end + interval, interval)
        sweeps_generator = list(product(h_range, w_range))

    total_shapes = len(sweeps_generator)

    if num_samples:
        idx_list = sorted(random.sample(range(total_shapes), num_samples))
    else:
        idx_list = list(range(total_shapes))

    for idx in idx_list:
        if square:
            h = sweeps_generator[idx]
            shape = [b_start, c_start, h, h]
            yield [shape] * num_shapes
        else:
            h, w = sweeps_generator[idx]
            shape = [b_start, c_start, h, w]
            yield [shape] * num_shapes


def generic_shape_sweeps(
    start_shape, end_shape, interval=32, num_shapes=1, num_samples=None
):
    num_dims = len(start_shape)

    assert len(end_shape) == num_dims

    if not isinstance(interval, list):
        interval = [interval] * num_dims

    assert len(interval) == num_dims

    ranges = [
        range(start_shape[i], end_shape[i] + interval[i], interval[i])
        for i in range(num_dims)
    ]

    sweeps_generator = list(product(*ranges))

    total_shapes = len(sweeps_generator)

    if num_samples:
        idx_list = sorted(random.sample(range(total_shapes), num_samples))
    else:
        idx_list = list(range(total_shapes))

    for idx in idx_list:
        shape = sweeps_generator[idx]
        yield [shape] * num_shapes


def matmul_shape_sweeps(
    dim1_tuple, dim2_tuple, dim3_tuple, interval=32, num_samples=None
):
    assert len(dim1_tuple) == 2
    assert len(dim2_tuple) == 2
    assert len(dim3_tuple) == 2

    b, c = 1, 1

    dim1_range = range(dim1_tuple[0], dim1_tuple[1] + interval, interval)
    dim2_range = range(dim2_tuple[0], dim2_tuple[1] + interval, interval)
    dim3_range = range(dim3_tuple[0], dim3_tuple[1] + interval, interval)
    sweeps_generator = list(product(dim1_range, dim2_range, dim3_range))

    total_shapes = len(sweeps_generator)

    if num_samples:
        idx_list = sorted(random.sample(range(total_shapes), num_samples))
    else:
        idx_list = list(range(total_shapes))

    for idx in idx_list:
        dim1, dim2, dim3 = sweeps_generator[idx]
        shape1 = [b, c, dim1, dim2]
        shape2 = [b, c, dim2, dim3]
        yield [shape1, shape2]


def get_args_from_argparser():
    parser = argparse.ArgumentParser(description="Pytorch testing infra")
    parser.add_argument(
        "-o",
        "--output-csv-file-path",
        default="pytorch_test_results.csv",
        help="output csv file path",
    )
    parser.add_argument(
        "-s",
        "--pcie-slot",
        default=0,
        type=int,
        help="Virtual PCIE slot of GS device to run on",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=None,
        help="Number of samples to sweep for test; by default, run all possible shapes",
    )

    args = parser.parse_args()

    return args
