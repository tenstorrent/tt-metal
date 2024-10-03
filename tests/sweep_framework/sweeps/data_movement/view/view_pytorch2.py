# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 10
# seed for random
random.seed(0)


def extract_brackets_content(line):
    # Function to extract the content inside brackets
    brackets_content = []
    open_brackets = 0
    current_content = ""

    for char in line:
        if char == "[":
            open_brackets += 1
            if open_brackets > 0:
                current_content = ""  # Reset content inside the brackets
        elif char == "]":
            if open_brackets > 0:
                brackets_content.append(current_content.strip())
            open_brackets -= 1
        elif open_brackets > 0:
            current_content += char

    return brackets_content


def parse_md_file_simple_no_regex(file_path):
    view_specs = []
    i = 0
    with open(file_path, "r") as file:
        for line in file:
            # Extract all sets of content inside brackets
            brackets_content = extract_brackets_content(line)

            if len(brackets_content) >= 3:  # Ensure we have both shape and size
                shape_str = brackets_content[0]  # First set of brackets for shape
                size_str = brackets_content[2]  # Third set of brackets for size

                # Convert the shape and size strings to lists of integers
                if "s" in shape_str or "s" in size_str:
                    continue
                shape = list(map(int, shape_str.split(",")))
                size = list(map(int, size_str.split(",")))

                # Append the dictionary to the list
                view_specs.append({"shape": shape, "size": size})
            i += 1

    return view_specs


parameters = {
    "nightly": {
        "view_specs": parse_md_file_simple_no_regex("sweeps/data_movement/view/view_trace.md"),
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    if test_vector["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        if test_vector["dtype"] == ttnn.bfloat8_b:
            return True, "bfloat8_b not supported with ROW_MAJOR_LAYOUT"

    return False, None


def run(
    view_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the shape and new size (target shape) from view_specs
    shape = view_specs["shape"]
    size = view_specs["size"]  # New shape for the view/reshape operation

    # Create a random tensor of the specified shape
    tensor = torch_random(shape, -0.1, 0.1, dtype=torch.bfloat16)

    # Apply view using PyTorch's view function to reshape the tensor
    torch_output_tensor = tensor.view(*size)

    # Convert the tensor to the ttnn tensor format
    ttnn_tensor = ttnn.from_torch(tensor, device=device, layout=layout, dtype=dtype)

    # Measure performance of the reshape operation in ttnn
    start_time = start_measuring_time()

    # Apply reshape in ttnn
    ttnn_output_tensor = ttnn.reshape(ttnn_tensor, size)

    e2e_perf = stop_measuring_time(start_time)

    # Convert the ttnn tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(torch_output_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]
