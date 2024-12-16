# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import ttnn


def load_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        exit(1)

    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def extract_tensor_info(tensor_str):
    match = re.search(r"tensor<\[(.*?)\]>", tensor_str)

    if match:
        content = match.group(1)
        parts = content.split(",")
        dimensions = [int(part.strip()) for part in parts[:-1]]
        data_type = parts[-1].strip()
        return dimensions, data_type
    else:
        return None, None


def extract_memconfig_info(layout_str):
    smem_config_a = layout_str[0]["memory_config"]
    smem_config_b = layout_str[1]["memory_config"]
    smem_config_o = layout_str[2]["memory_config"]
    layout_a = "ttnn.TILE_LAYOUT"
    layout_b = "ttnn.TILE_LAYOUT"
    mem_config_a = "ttnn.DRAM_MEMORY_CONFIG"
    mem_config_b = "ttnn.DRAM_MEMORY_CONFIG"
    mem_config_o = "ttnn.DRAM_MEMORY_CONFIG"
    if "tile" in smem_config_a:
        layout_a = "ttnn.TILE_LAYOUT"
    if "tile" in smem_config_b:
        layout_b = "ttnn.TILE_LAYOUT"
    if "dram" in smem_config_a:
        mem_config_a = "ttnn.DRAM_MEMORY_CONFIG"
    if "dram" in smem_config_b:
        mem_config_b = "ttnn.DRAM_MEMORY_CONFIG"
    if "dram" in smem_config_o:
        mem_config_o = "ttnn.DRAM_MEMORY_CONFIG"
    if "l1" in smem_config_a:
        mem_config_a = "ttnn.L1_MEMORY_CONFIG"
    if "l1" in smem_config_b:
        mem_config_b = "ttnn.L1_MEMORY_CONFIG"
    if "l1" in smem_config_o:
        mem_config_o = "ttnn.L1_MEMORY_CONFIG"

    return layout_a, layout_b, mem_config_a, mem_config_b, mem_config_o


def generate_unary_test_file(test_data, filename):
    ops = test_data[0]["name"]
    input_shapes = list()

    def get_dtype(dtype):
        if dtype == "i32":
            return "ttnn.int32"
        if dtype == "f32":
            return "ttnn.float32"
        if dtype == "bf16":
            return "ttnn.bfloat16"
        return "ttnn.bfloat16"

    for i in range(len(test_data)):
        if test_data[i]["runs_on_ttnn"] != "yes":
            continue
        input_a_shape, input_a_dtype = extract_tensor_info(test_data[i]["input_shapes"][0])
        input_b_shape, input_b_dtype = extract_tensor_info(test_data[i]["input_shapes"][1])
        output_shape, output_dtype = extract_tensor_info(test_data[i]["output_shapes"][0])
        # oshape, odtype = extract_tensor_info(test_data[i]["output_shapes"][0])
        print(input_a_shape, input_a_dtype)
        print(input_b_shape, input_b_dtype)
        print(output_shape, output_dtype)
        input_dtype = get_dtype(input_a_dtype)

        (
            input_a_layout,
            input_b_layout,
            input_a_mem_config,
            input_b_mem_config,
            output_mem_config,
        ) = extract_memconfig_info(test_data[i]["input_layouts"])

        input_shapes.append(
            {
                "self": input_a_shape,
                "other": input_b_shape,
                "input_dtype": input_dtype,
            }
        )

    shapes_str = ",\n\t\t\t".join([str(shape) for shape in input_shapes])

    param_string = f"""
parameters = {{
    "nightly": {{
        "input_shape":
           [{shapes_str}],
        "input_a_layout": [{input_a_layout}],
        "input_b_layout": [{input_b_layout}],
        "input_a_memory_config": [{input_a_mem_config}],
        "input_b_memory_config": [{input_b_mem_config}],
        "output_memory_config": [{output_mem_config}],
    }},
}}
"""
    code_string = f"""
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

{param_string}

# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT or test_vector["input_b_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row Major layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_layout,
    input_b_layout,
    input_a_memory_config,
    input_b_memory_config,
    output_memory_config,
    *,
    device,
) -> list:

    torch.manual_seed(0)
    if input_shape["input_dtype"] == "ttnn.bfloat16":
        input_dtype = ttnn.bfloat16
    elif input_shape["input_dtype"] == "ttnn.float32":
        input_dtype = ttnn.float32
    elif input_shape["input_dtype"] == "ttnn.int32":
        input_dtype = ttnn.int32

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape["self"])
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_dtype
    )(input_shape["other"])

    golden_function = ttnn.get_golden_function({ops})
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )

    start_time = start_measuring_time()
    result = {ops}(input_tensor_a, input_tensor_b, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result, torch_rank=len(input_shape["self"]))
    e2e_perf = stop_measuring_time(start_time)

    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
"""

    try:
        with open(filename, "w") as f:
            f.write(code_string)
        print(f"Python file '{filename}' has been created successfully.")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process a JSON file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to output sweep file")

    args = parser.parse_args()
    json_data = load_json_file(args.input)

    generate_unary_test_file(json_data, args.output)


if __name__ == "__main__":
    main()
