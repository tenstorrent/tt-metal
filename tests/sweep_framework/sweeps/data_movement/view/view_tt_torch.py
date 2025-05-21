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
        for line in file.readlines():
            parsed_line = line.split("|")
            if parsed_line[0] == " ttnn.reshape ":
                tensor = parsed_line[2].split("[")[1].split("]")[0]
                target = parsed_line[5].split("[")[1].split("]")[0]
                tensor_shape = list(map(int, tensor.split(",")[:-1]))
                target_shape = list(map(int, target.split(",")[:-1]))
                addition = {"shape": tensor_shape, "size": target_shape}
                view_specs.append(addition)
    return view_specs


parameters = {
    "nightly": {
        "view_specs": parse_md_file_simple_no_regex(
            "./tests/sweep_framework/sweeps/data_movement/view/tt_torch_trace.md"
        ),
        "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
        "dtype": [ttnn.bfloat16, ttnn.float32],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    view_specs,
    layout,
    dtype,
    *,
    device,
):
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
