import torch
import matplotlib.pyplot as plt
import os
import ttnn
import pandas as pd
from tests.ttnn.utils_for_testing import assert_with_ulp

# from models.common.utility_functions import comp_ulp_check

import models.utility_functions as util


def test_softmax_arange(device):
    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16).to(torch.float32)

    # Mask -inf, +inf, and NaN values before softmax is necessary because they propagate into the denominator and can produce invalid (NaN) probabilities.
    valid_input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    # Mask with exp Working range - Overflow from 88.5(inf), Underflow till -87(<0)
    low = -87.0
    high = 88.5
    mask = (valid_input_tensor >= low) & (valid_input_tensor <= high)
    valid_input_tensor = valid_input_tensor[mask]

    # print(f"Testing {len(valid_input_tensor)} valid bf16 values in chunks...")

    # Test all dimensions
    test_dims = [0, 1]
    results_by_dim = {}

    # Process 65K Inputs in chunks by processing 1K values at a time
    chunk_size = 1024
    total_chunks = (len(valid_input_tensor) + chunk_size - 1) // chunk_size

    for dim in test_dims:
        print(f"\nTesting softmax with dim{dim} ")

        # Lists to collect results from all chunks
        all_golden_results = []
        all_ttnn_results = []
        all_input_values = []  # Track input values for debug purpose

        for i in range(total_chunks):
            # Get the chunk of inputs : Extract the i-th slice (or smaller if at the end) from the input tensor
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(valid_input_tensor))
            chunk = valid_input_tensor[start_idx:end_idx]

            # Pad chunk to 1024 values and reshape to [32, 32]. creating a zero tesnor to avoid errors. That is if chunk has fewer than 1024 elements, they stay as zeros.
            padded_chunk = torch.zeros(1024, dtype=torch.float32)
            padded_chunk[: len(chunk)] = chunk
            chunk_2d = padded_chunk.view(32, 32)

            tt_in = ttnn.from_torch(
                chunk_2d,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            golden_function = ttnn.get_golden_function(ttnn.softmax)
            golden = golden_function(chunk_2d, dim=dim, device=device)

            tt_result = ttnn.softmax(tt_in, dim=dim)
            tt_result = ttnn.from_device(tt_result)
            result = ttnn.to_torch(tt_result)

            # Appending results for ULP computation at the end of processing each chunk
            golden_flat = golden.flatten()
            result_flat = result.flatten()
            input_flat = chunk_2d.flatten()
            all_golden_results.append(golden_flat)
            all_ttnn_results.append(result_flat)
            all_input_values.append(input_flat)

            # print(f"Processed chunk {i+1}/{total_chunks} for dim={dim}")

        # print(f"Concatenating results for dim={dim}...")

        # Concatenate all results into single tensors for this dimension
        final_golden = torch.cat(all_golden_results)
        final_ttnn = torch.cat(all_ttnn_results)
        final_inputs = torch.cat(all_input_values)

        # Store results for this dimension
        results_by_dim[dim] = {"golden": final_golden, "ttnn": final_ttnn, "inputs": final_inputs}

        # ULP check for each dimension
        try:
            assert_with_ulp(final_golden, final_ttnn, 1, allow_nonfinite=True)
            status = f"ULP test PASSED for dim={dim}"
        except AssertionError as e:
            status = f"ULP assertion error for dim={dim}: {str(e)}"

        # print(f"Status for dim={dim}: {status}")
        results_by_dim[dim]["status"] = status

    print(f"\nTest Summary")
    for dim in test_dims:
        print(f"Dim {dim}: {results_by_dim[dim]['status']}")


# Testing softmax with dim0
# softmax_kernel_path: ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_h.cpp

# Testing softmax with dim1
# softmax_defines[EXP_APPROX]: 1
# softmax_kernel_path: ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp
