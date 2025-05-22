import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_dtype,
    compare_pcc,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from functools import partial
import random
from models.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_equal
import os
import numpy as np

from tests.ttnn.utils_for_testing import check_with_pcc
import csv
import datetime


def compare_tensors(input_tensor, calculated_tensor, expected_tensor):
    assert input_tensor.shape == calculated_tensor.shape == expected_tensor.shape, "Tensors must have the same shape"
    mismatch_indices = torch.nonzero(calculated_tensor != expected_tensor)
    for idx in mismatch_indices:
        idx_tuple = tuple(idx.tolist())
        print(f"Mismatch at index {idx_tuple}:")
        print(f"  Input tensor value: {input_tensor[idx_tuple]}")
        print(f"  Calculated tensor value: {calculated_tensor[idx_tuple]}")
        print(f"  Expected tensor value: {expected_tensor[idx_tuple]}")
        print("=" * 50)


def file_handler():
    current_directory = os.getcwd()
    current_timestamp = datetime.datetime.now()
    str_timestamp = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(current_directory, "results", f"test_fail_{str_timestamp}.txt")
    csv_file_path = os.path.join(current_directory, "results", f"test_results_{str_timestamp}.csv")
    print(f"******** Results will be saved to: {csv_file_path} ********")

    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
    return file_path, csv_file_path


txt_file, csv_file = file_handler()


def write_to_csv_file(file_path, test_vector, status, message):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        f.seek(0, 2)
        if f.tell() == 0:
            writer.writerow(
                [
                    "op",
                    "input_a_shape",
                    "input_b_shape",
                    "input_a_dtype",
                    "input_b_dtype",
                    "input_a_layout",
                    "input_b_layout",
                    "a_mem",
                    "b_mem",
                    "message",
                ]
            )

        binary_op = test_vector.get("binary_op", {})
        input_shape = test_vector.get("input_shape", {})
        input_dtype = test_vector.get("input_dtype", {})
        input_mem_config = test_vector.get("input_mem_config", {})
        input_a_layout = test_vector.get("input_a_layout", "unknown")
        input_b_layout = test_vector.get("input_b_layout", "unknown")

        op = binary_op.get("tt_op", "unknown")
        input_a_shape = input_shape.get("self", "unknown")
        input_b_shape = input_shape.get("other", "unknown")
        input_a_dtype = input_dtype.get("input_a_dtype", "unknown")
        input_b_dtype = input_dtype.get("input_b_dtype", "unknown")
        a_mem = input_mem_config.get("a_mem", "unknown")
        b_mem = input_mem_config.get("b_mem", "unknown")

        writer.writerow(
            [
                op,
                input_a_shape,
                input_b_shape,
                input_a_dtype,
                input_b_dtype,
                input_a_layout,
                input_b_layout,
                a_mem,
                b_mem,
                message,
            ]
        )


def write_status_to_csv_file(file_path, message):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        f.seek(0, 2)
        if f.tell() == 0:
            writer.writerow(
                [
                    "status",
                ]
            )

        writer.writerow([message])


def return_mem_config(mem_config_string, H=256, W=256, ncores=8, y=2, x=4):
    if mem_config_string == "l1_interleaved":
        return ttnn.L1_MEMORY_CONFIG
    elif mem_config_string == "dram_interleaved":
        return ttnn.DRAM_MEMORY_CONFIG
    elif mem_config_string == "l1_height_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(H // ncores, W),
            core_grid=ttnn.CoreGrid(y=y, x=x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_height_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(H, W // ncores),
            core_grid=ttnn.CoreGrid(y=y, x=x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(H, W // ncores),
            core_grid=ttnn.CoreGrid(y=y, x=x),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_width_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(H // ncores, W),
            core_grid=ttnn.CoreGrid(y=y, x=x),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_rm":
        return ttnn.create_sharded_memory_config(
            shape=(H // y, W // x),
            core_grid=ttnn.CoreGrid(y=y, x=x),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    elif mem_config_string == "l1_block_sharded_cm":
        return ttnn.create_sharded_memory_config(
            shape=(H // y, W // x),
            core_grid=ttnn.CoreGrid(y=y, x=x),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.COL_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    raise ("Input mem_config_string is not valid!")


# Test parameters


@pytest.mark.parametrize(
    "input_shape",
    [
        {"self": [1, 1, 256, 256], "other": [1, 1, 256, 256]},
    ],
)
@pytest.mark.parametrize(
    "binary_op",
    [{"tt_op": "gcd", "a_high": 1000, "b_high": 2000, "a_low": -1000, "b_low": -2000}],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        {"input_a_dtype": ttnn.int32, "input_b_dtype": ttnn.int32},
    ],
)
@pytest.mark.parametrize(
    "input_a_layout",
    [ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize(
    "input_b_layout",
    [ttnn.TILE_LAYOUT],
)
@pytest.mark.parametrize(
    "input_mem_config",
    [
        # {"a_mem": "l1_interleaved", "b_mem": "l1_interleaved"},
        # {"a_mem": "l1_interleaved", "b_mem": "dram_interleaved"},
        # {"a_mem": "dram_interleaved", "b_mem": "l1_interleaved"},
        # {"a_mem": "dram_interleaved", "b_mem": "dram_interleaved"},  # l1 - dram combination
        {"a_mem": "l1_height_sharded_rm", "b_mem": "l1_height_sharded_rm"},  # Failed
        {"a_mem": "dram_interleaved", "b_mem": "l1_height_sharded_rm"},
        {"a_mem": "l1_height_sharded_rm", "b_mem": "dram_interleaved"},  # HS #Failed
        {"a_mem": "l1_width_sharded_rm", "b_mem": "l1_width_sharded_rm"},  # Failed
        {"a_mem": "dram_interleaved", "b_mem": "l1_width_sharded_rm"},
        {"a_mem": "l1_width_sharded_rm", "b_mem": "dram_interleaved"},  # WS #Failed
        {"a_mem": "l1_block_sharded_rm", "b_mem": "l1_block_sharded_rm"},  # Failed
        {"a_mem": "dram_interleaved", "b_mem": "l1_block_sharded_rm"},
        {"a_mem": "l1_block_sharded_rm", "b_mem": "dram_interleaved"},  # BS #row_major orientation #Failed
        {"a_mem": "l1_height_sharded_cm", "b_mem": "l1_height_sharded_cm"},  # Failed
        {"a_mem": "dram_interleaved", "b_mem": "l1_height_sharded_cm"},
        {"a_mem": "l1_height_sharded_cm", "b_mem": "dram_interleaved"},  # HS #Failed
        {"a_mem": "l1_width_sharded_cm", "b_mem": "l1_width_sharded_cm"},  # Failed
        {"a_mem": "dram_interleaved", "b_mem": "l1_width_sharded_cm"},
        {"a_mem": "l1_width_sharded_cm", "b_mem": "dram_interleaved"},  # WS #Failed
        {"a_mem": "l1_block_sharded_cm", "b_mem": "l1_block_sharded_cm"},  # Failed
        {"a_mem": "dram_interleaved", "b_mem": "l1_block_sharded_cm"},
        {"a_mem": "l1_block_sharded_cm", "b_mem": "dram_interleaved"},  # Failed
    ],
)
def test_non_bcast(
    binary_op,
    input_shape,
    input_dtype,
    input_a_layout,
    input_b_layout,
    input_mem_config,
    *,
    device,
) -> list:
    torch.manual_seed(0)

    # Create test vector
    test_vector = {
        "binary_op": binary_op,
        "input_shape": input_shape,
        "input_dtype": input_dtype,
        "input_mem_config": input_mem_config,
        "input_a_layout": input_a_layout,
        "input_b_layout": input_b_layout,
    }

    ttnn_fn = getattr(ttnn, binary_op["tt_op"])
    a_high = binary_op["a_high"]
    b_high = binary_op["b_high"]
    a_low = binary_op["a_low"]
    b_low = binary_op["b_low"]
    input_a_dtype = input_dtype["input_a_dtype"]  # change
    input_b_dtype = input_dtype["input_b_dtype"]  # change
    torch_input_tensor_b = None
    input_tensor_b = None

    torch_dtype = torch.float32
    if input_a_dtype == ttnn.int32 and input_b_dtype == ttnn.int32:
        torch_dtype = torch.int32
    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=a_low, high=a_high, dtype=torch_dtype), input_a_dtype
    )(input_shape["self"])

    if isinstance(input_shape["other"], list) and input_b_dtype != None:
        torch_input_tensor_b = gen_func_with_cast_tt(
            partial(torch_random, low=b_low, high=b_high, dtype=torch_dtype), input_b_dtype
        )(input_shape["other"])
    else:
        torch_input_tensor_b = torch.tensor(input_shape["other"], dtype=torch_dtype)

    if input_b_dtype == None and input_a_dtype == ttnn.int32:
        torch_input_tensor_b = random.randint(b_low, b_high)
        input_tensor_b = torch_input_tensor_b
        # torch logical don't have b-scalar
        torch_input_tensor_b = torch.full_like(torch_input_tensor_a, input_tensor_b)
    elif input_b_dtype == None:
        torch_input_tensor_b = random.uniform(b_low, b_high)
        input_tensor_b = torch_input_tensor_b
        # torch logical don't have b-scalar
        torch_input_tensor_b = torch.full_like(torch_input_tensor_a, input_tensor_b)

    input_a_memory_config = input_mem_config["a_mem"]
    input_b_memory_config = input_mem_config["b_mem"]

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=return_mem_config(input_a_memory_config),
    )

    if input_b_dtype != None:
        input_tensor_b = ttnn.from_torch(
            torch_input_tensor_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=return_mem_config(input_b_memory_config),
        )
        torch_input_tensor_b = ttnn.to_torch(input_tensor_b)

    torch_input_tensor_a = ttnn.to_torch(input_tensor_a)

    golden_function = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    ttnn.set_printoptions(profile="full")
    torch.set_printoptions(threshold=float("inf"), linewidth=1000)
    # start_time = start_measuring_time()
    status = False
    message = " "

    try:
        result = ttnn_fn(input_tensor_a, input_tensor_b)
        output_tensor = ttnn.to_torch(result)
        status, message = check_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
        # compare_tensors(torch_input_tensor_a, output_tensor, torch_output_tensor)
        if not status:
            message += "\n--- Debug Info ---"
            message += f"\nInput Tensor A: {input_tensor_a}"
            message += f"\nInput Tensor B: {input_tensor_b}"
            message += f"\nOutput TTNN Tensor: {output_tensor}"
            message += f"\nOutput Torch Tensor: {torch_output_tensor}"

    except Exception as e:
        status, message = False, str(e)

    if status:
        write_to_csv_file(csv_file, test_vector, status, "PASSED")
    if not status:
        write_to_csv_file(csv_file, test_vector, status, message)
