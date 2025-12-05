# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import math


def main():
    """
    Main function to test ttnn.log and display results
    """
    print("Testing ttnn.log vs torch.log comparison")
    print("=" * 50)

    # Initialize device
    device = ttnn.open_device(device_id=0)

    try:
        # Create input tensor with specified values
        inputs = [1.0, math.inf, -math.inf, math.nan]
        torch_x_bf16 = torch.tensor(inputs, dtype=torch.bfloat16)
        torch_x_f32 = torch.tensor(inputs, dtype=torch.float32)

        ttnn_x_bf16 = ttnn.from_torch(torch_x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_x_f32 = ttnn.from_torch(torch_x_f32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

        golden_output_f32 = torch.eq(torch_x_f32, torch_x_f32)
        golden_output_bf16 = torch.eq(torch_x_bf16, torch_x_bf16)
        calculated_output_f32 = ttnn.eq(ttnn_x_f32, ttnn_x_f32)
        calculated_output_bf16 = ttnn.eq(ttnn_x_bf16, ttnn_x_bf16)

        # Display results
        print(f"Input values: {inputs}")
        print(f"")
        print(f"torch_x_f32 = {torch_x_f32}")
        print(f"torch_x_bf16 = {torch_x_bf16}")
        print(f"\n----- [bfloat16] -----")
        print(f"Torch (golden) result:  {golden_output_bf16}")
        print(f"TTNN result:            {calculated_output_bf16}")
        print(f"\n----- [float32] -----")
        print(f"Torch (golden) result:  {golden_output_f32}")
        print(f"TTNN result:            {calculated_output_f32}")

    finally:
        # Clean up device
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
