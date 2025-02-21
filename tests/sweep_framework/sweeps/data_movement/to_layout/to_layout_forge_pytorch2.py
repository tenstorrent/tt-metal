from typing import Optional, Tuple

import torch
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random

TIMEOUT = 50

bf16 = "bf16"
f32 = "f32"
i32 = "i32"
tile = "tile"
row_major = "row_major"

dtype_dict = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint32: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint16: torch.int16,
}

ttnn_type_dict = {
    bf16: ttnn.bfloat16,
    i32: ttnn.int32,
    f32: ttnn.float32,
}

layout_dict = {
    tile: ttnn.TILE_LAYOUT,
    row_major: ttnn.ROW_MAJOR_LAYOUT,
}
# few params because file is too large
# use extract params to extract all of them
parameters = {
    "default": {
        "params": [
            ((1, 12, 10, 10), f32, tile),
            ((1, 12, 10, 10), f32, row_major),
            ((1, 12, 10, 10), f32, tile),
            ((9, 3072), f32, row_major),
            ((9, 3072), bf16, tile),
            ((9, 3072), f32, row_major),
            ((9, 4096), f32, tile),
            ((9, 4096), f32, tile),
            ((9, 4096), bf16, row_major),
            ((9, 4096), f32, row_major),
            ((9, 4096), bf16, tile),
            ((9, 4096), f32, row_major),
            ((9, 768), f32, tile),
            ((9, 768), bf16, row_major),
            ((9, 768), f32, tile),
            ((9, 768), bf16, row_major),
            ((9, 768), bf16, tile),
            ((9, 768), f32, row_major),
            ((9, 8192), f32, tile),
            ((9, 8192), f32, tile),
            ((9, 8192), bf16, row_major),
            ((9, 8192), f32, row_major),
            ((9, 8192), bf16, tile),
            ((9, 8192), f32, row_major),
            ((1,), i32, tile),
            ((1,), bf16, row_major),
            ((1,), i32, tile),
            ((1,), bf16, row_major),
        ],
    }
}


def run(
    params,
    *,
    device,
) -> list:
    [input_shape, ttnn_type, layout] = params

    ttnn_dtype = ttnn_type_dict[ttnn_type]
    torch_dtype = dtype_dict[ttnn_dtype]
    input_layout = layout_dict[layout]

    if input_layout == ttnn.TILE_LAYOUT:
        output_layout = ttnn.ROW_MAJOR_LAYOUT
    else:
        output_layout = ttnn.TILE_LAYOUT
    print(ttnn_dtype)
    print(torch_dtype)
    print(input_layout)
    print(output_layout)

    tensor_shape = input_shape  # tuple(to_layout_specs["shape"])
    print(type(tensor_shape))
    print(tensor_shape)
    torch_input_tensor = None
    # create random tensor in PyTorch
    # if torch dtype is interger, use randint to create tensor
    if torch_dtype == torch.int32:
        torch_input_tensor = torch.randint(1, 1024, tensor_shape, dtype=torch_dtype)
    else:
        torch_input_tensor = torch_random(tensor_shape, -0.1, 0.1, dtype=torch_dtype)

    # create ttnn tensor from torch tensor
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=input_layout, dtype=ttnn_dtype)

    start_time = start_measuring_time()
    # create output tensor using to_layout
    ttnn_tensor = ttnn.to_layout(ttnn_input_tensor, output_layout)
    e2e_perf = stop_measuring_time(start_time)

    # convert back to torch
    ttnn_converted_tensor = ttnn.to_torch(ttnn_tensor)

    # Compare the results and return performance and accuracy check
    result = check_with_pcc(ttnn_converted_tensor, torch_input_tensor)

    return [result, e2e_perf]
