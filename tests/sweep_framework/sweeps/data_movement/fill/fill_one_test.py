from typing import Optional, Tuple

import torch

torch.manual_seed(42)
import random
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
import pdb


def run(
    fill_specs,
    dtype,
    layout,
    *,
    device,
):
    device.enable_async(False)

    # Extract the shape and fill_value from the test case
    shape = fill_specs["shape"]
    fill_value = fill_specs["fill_value"]

    # Create a tensor filled with `fill_value` using torch.full
    torch_tensor = torch.full(shape, fill_value, dtype=torch.float32)

    # Create a random tensor in PyTorch of shape `shape` and dtype float32
    rand_tensor = torch_random(shape, -256, 256, dtype=torch.float32)

    # Convert the torch tensor to the `ttnn` format
    ttnn_tensor = ttnn.from_torch(rand_tensor, device=device, layout=layout, dtype=dtype)

    # Measure performance of the full operation in `ttnn`
    start_time = start_measuring_time()

    # Apply the `ttnn.full` operation
    ttnn_filled_tensor = ttnn.fill(fill_value, any=ttnn_tensor, memory_config=None)

    e2e_perf = stop_measuring_time(start_time)

    # Convert the `ttnn` tensor back to PyTorch for comparison
    ttnn_output_tensor = ttnn.to_torch(ttnn_filled_tensor)

    # Compare the PyTorch and `ttnn` tensors
    result = check_with_pcc(torch_tensor, ttnn_output_tensor, 0.999)

    return [result, e2e_perf]


if __name__ == "__main__":
    # Run the test

    fill_specs = {"shape": [128, 128, 300, 300], "fill_value": 64}

    dtype = ttnn.bfloat16
    # layout = ttnn.TILE_LAYOUT
    layout = ttnn.ROW_MAJOR_LAYOUT
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    print(run(fill_specs, dtype, layout, device=device))
