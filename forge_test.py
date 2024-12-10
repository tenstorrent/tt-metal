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
    smem_config = layout_str[0]["memory_config"]
    layout = "ttnn.TILE_LAYOUT"
    mem_config = "ttnn.DRAM_MEMORY_CONFIG"
    if "tile" in smem_config:
        layout = "ttnn.TILE_LAYOUT"
    if "dram" in smem_config:
        mem_config = "ttnn.DRAM_MEMORY_CONFIG"
    if "l1" in smem_config:
        mem_config = "ttnn.L1_MEMORY_CONFIG"
    return layout, mem_config


def generate_unary_test_file(test_data, filename):
    ops = test_data[0]["name"]
    shapes = list()
    for i in range(len(test_data)):
        if test_data[i]["runs_on_ttnn"] != "yes":
            continue
        shape, sdtype = extract_tensor_info(test_data[i]["input_shapes"][0])
        oshape, odtype = extract_tensor_info(test_data[i]["output_shapes"][0])
        print(oshape, odtype)
        dtype = "ttnn.bfloat16"
        if sdtype == "i32":
            dtype = "ttnn.int32"
        if sdtype == "f32":
            dtype = "ttnn.float32"
        if sdtype == "bf16":
            dtype = "ttnn.bfloat16"

        in_layout, in_mem_connfig = extract_memconfig_info(test_data[i]["input_layouts"])
        out_layout, out_mem_connfig = extract_memconfig_info(test_data[i]["output_layouts"])
        shapes = shapes + [shape]

    param_string = f"""
parameters = {{
    "nightly": {{
        "input_shape":
           {shapes},
        "input_a_dtype": [{dtype}],
        "input_a_layout": [{in_layout}],
        "input_a_memory_config": [{in_mem_connfig}],
        "output_memory_config": [{out_mem_connfig}],
    }},
}}
"""
    code_string = f"""
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial

import torch
import random
import ttnn
from tests.sweep_framework.sweep_utils.utils import gen_shapes
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

# Override the default timeout in seconds for hang detection.
# TIMEOUT = 30

random.seed(0)


# Parameters provided to the test vector generator are defined here.
# They are defined as dict-type suites that contain the arguments to the run function as keys, and lists of possible inputs as values.
# Each suite has a key name (in this case "suite_1" and "suite_2") which will associate the test vectors to this specific suite of inputs.
# Developers can create their own generator functions and pass them to the parameters as inputs.

{param_string}

# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["input_a_layout"] == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Row Major layout is not supported"
    return False, None


# This is the run instructions for the test, defined by the developer.
# The run function must take the above-defined parameters as inputs.
# The runner will call this run function with each test vector, and the returned results from this function will be stored.
# If you defined a mesh_device_fixture above, the object you yielded will be passed into this function as 'device'. Otherwise, it will be the default ttnn device opened by the infra.
def run(
    input_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:

    torch.manual_seed(0)

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(input_shape)

    golden_function = ttnn.get_golden_function({ops})
    torch_output_tensor = golden_function(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=input_a_memory_config,
    )

    start_time = start_measuring_time()
    result = {ops}(input_tensor_a, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(result)
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
