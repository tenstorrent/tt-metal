# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ttnn_out_value = ttnn_output_tensor.item()
    torch_out_value = torch_output_tensor.item()
    return abs_diff, torch_out_value, ttnn_out_value


# Max allowed tolerances
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

# Function to find the maximum supported range
start_value = -103167
end_value = 103167
num_inputs = 1000
x_vals = np.linspace(start_value, end_value, num_inputs)
abs_diffs = []
torch_out_values = []
ttnn_out_values = []
for x in x_vals:
    abs_diff, torch_out_value, ttnn_out_value = difference(x)
    abs_diffs.append(abs_diff)
    torch_out_values.append(torch_out_value)
    ttnn_out_values.append(ttnn_out_value)
abs_diffs = np.array(abs_diffs)
torch_out_values = np.array(torch_out_values)
ttnn_out_values = np.array(ttnn_out_values)

# Find the range where the difference is within the tolerances
for tolerance in tolerances:
    max_support_range_start = None
    max_support_range_end = None

    for i in range(len(abs_diffs)):
        if abs_diffs[i] <= tolerance:
            if max_support_range_start is None:
                max_support_range_start = x_vals[i]
            max_support_range_end = x_vals[i]
        else:
            if max_support_range_start is not None:
                break

    if max_support_range_start is not None and max_support_range_end is not None:
        print(f"Maximum support range for cosine function (tolerance {tolerance}):")
        print(f"Start: {max_support_range_start} , End: {max_support_range_end}")
    else:
        print(f"No support range found within the tolerance of {tolerance}")

# Plot for comparison
plt.plot(x_vals, torch_out_values, label="torch.cos", linestyle="--")
plt.plot(x_vals, ttnn_out_values, label="ttnn.cos", linestyle="-")
plt.fill_between(
    x_vals,
    torch_out_values,
    ttnn_out_values,
    where=(abs_diffs <= tolerance),
    color="gray",
    alpha=0.5,
    label="Support range",
)
plt.xlabel("x")
plt.ylabel("cos(x)")
plt.legend()
plt.title(f"Comparison of torch.cos and ttnn.cos (tolerance = {tolerance})")
plt.grid(True)
plt.savefig("cosine_comparison_plot.png")
print("Plot saved as 'cosine_comparison_plot.png'")
plt.show()
