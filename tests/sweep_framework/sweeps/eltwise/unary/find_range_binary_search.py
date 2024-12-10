# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


device_id = 0
device = ttnn.open_device(device_id=device_id)


def difference(x):
    torch_input_tensor = ttnn.from_torch(
        torch.tensor([[x]], dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=None,
        memory_config=None,
    )
    torch_input_tensor = ttnn.to_torch(torch_input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.cos)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output_tensor = ttnn.cos(input_tensor)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
    abs_diff = torch.abs(torch_output_tensor - ttnn_output_tensor).item()
    return abs_diff


# Find supported range using binary search method
def find_max_range(tolerance, start_value, end_value, precision=1e-5):
    """
    Note:
    For tolerances ranging from 1e-3 to 1e-7, this function returns random supported ranges instead of "No valid range found" due to discontinuities in the TT cosine output.
    This occurs because a few random outputs within the input range (0 to 1e7) give exactly the same result as the torch output, resulting in an absolute difference of 0.0.
    The binary search algorithm identifies these output values as roots and therefore prints the corresponding input values as the positive/negative maximum supported range.
    A similar issue occurs for bisect method. This discontinuity needs to be handled.
    """
    lower_pos = start_value
    upper_pos = end_value
    lower_neg = -start_value
    upper_neg = -end_value
    best_x_pos = None
    best_x_neg = None

    # binary search in positive direction
    while abs(upper_pos - lower_pos) > precision:
        mid_pos = (lower_pos + upper_pos) / 2
        error_at_mid_pos = difference(mid_pos)

        if error_at_mid_pos <= tolerance:
            best_x_pos = mid_pos
            lower_pos = mid_pos  # Move the lower bound up to explore larger values
        else:
            upper_pos = mid_pos  # Move the upper bound down to explore smaller values

    # binary search in negative direction
    while abs(upper_neg - lower_neg) > precision:
        mid_neg = (lower_neg + upper_neg) / 2
        error_at_mid_neg = difference(mid_neg)

        if error_at_mid_neg <= tolerance:
            best_x_neg = mid_neg
            lower_neg = mid_neg

        else:
            upper_neg = mid_neg

    if best_x_pos is None:
        best_x_pos = "No valid positive range found"
    if best_x_neg is None:
        best_x_neg = "No valid negative range found"

    return best_x_neg, best_x_pos


# Max allowed tolerances
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

start_value = 0
end_value = 1e7

for tolerance in tolerances:
    max_support_range_neg, max_support_range_pos = find_max_range(tolerance, start_value, end_value)

    print(f"Maximum support range for cosine function (tolerance {tolerance}):")
    print(f"Negative direction: {max_support_range_neg}")
    print(f"Positive direction: {max_support_range_pos}")
