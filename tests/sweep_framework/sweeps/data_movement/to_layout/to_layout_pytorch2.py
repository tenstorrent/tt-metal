from typing import Optional, Tuple

import torch
import random
import ttnn
import traceback

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 50
# seed for random
random.seed(0)


def generate_to_layout_config_ND(num_samples=200):
    # Iterate to generate 'num_samples' configurations
    for _ in range(num_samples):
        # Randomly determine the number of dimensions (between 1 and 8)
        num_dims = random.randint(1, 7)

        # Generate random shape with dimensions
        # shape = [random.randint(1, 9) for _ in range(num_dims)]

        # Each dimension should be rounded to a power of 2
        # shape = [2 ** (shape[i]) for i in range(num_dims)]

        # Also try shapes that are not power of 2 for tilize/untilize with padding
        shape = [random.randint(1, 256) for _ in range(num_dims)]

        # Yield the configuration as a dictionary
        yield {
            "shape": shape,
        }


dtype_dict = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint32: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint16: torch.int16,
}

parameters = {
    "nightly": {
        "to_layout_specs": list(generate_to_layout_config_ND(num_samples=5)),
        "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.uint32],
        "input_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        # "output_layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    }
}


# Invalidate vector is called during the generation phase where each vector will be passed in.
# If invalidated, the vector will still be stored but will be skipped.
# Returns False, None if the vector is valid, and True, str with a reason for invalidation if it is invalid.
# def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
#    if test_vector["input_layout"] == test_vector["output_layout"]:
#        return True, "only testing conversion to different layout"

#    return False, None


def run(
    to_layout_specs,
    dtype,
    input_layout,
    # output_layout,
    *,
    device,
):
    device.enable_async(False)

    if input_layout == ttnn.TILE_LAYOUT:
        output_layout = ttnn.ROW_MAJOR_LAYOUT
    else:
        output_layout = ttnn.TILE_LAYOUT
    torch_dtype = dtype_dict[dtype]
    ttnn_dtype = dtype
    print("Start a sweep test set! llong")
    print(dtype)
    print(torch_dtype)
    print(input_layout)
    print(output_layout)

    tensor_shape = tuple(to_layout_specs["shape"])
    print(tensor_shape)
    torch_input_tensor = None
    # create random tensor in PyTorch
    # if torch dtype is interger, use randint to create tensor
    if torch_dtype in [torch.int32, torch.int16]:
        torch_input_tensor = torch.randint(0, 64, tensor_shape, dtype=torch_dtype)
    else:
        torch_input_tensor = torch_random(tensor_shape, -0.1, 0.1, dtype=torch_dtype)

    # create ttnn tensor from torch tensor
    try:
        ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=input_layout, dtype=ttnn_dtype)

        start_time = start_measuring_time()
        # create output tensor using to_layout
        ttnn_tensor = ttnn.to_layout(ttnn_input_tensor, output_layout)
        e2e_perf = stop_measuring_time(start_time)

        # convert back to torch
        ttnn_converted_tensor = ttnn.to_torch(ttnn_tensor)
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        e2e_perf = TIMEOUT
        print(
            f"Error occurred with parameters: to_layout_specs={to_layout_specs}, dtype={dtype}, input_layout={input_layout}, output_layout={output_layout}"
        )
        return [False, None]

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(ttnn_converted_tensor, torch_input_tensor)

    return [result, e2e_perf]
