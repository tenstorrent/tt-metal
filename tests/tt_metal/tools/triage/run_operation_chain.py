#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os, glob
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def main():
    """Run the operation chain that may hang if infinite loop is present"""

    # Set up environment
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["PYTHONPATH"] = os.environ["TT_METAL_HOME"]

    # Enable Inspector with RPC server for operation tracking
    os.environ["TT_METAL_INSPECTOR_LOG_PATH"] = "generated/inspector"

    torch.manual_seed(42)

    # Clean up any existing ops.yaml files before test
    ops_files = glob.glob("./generated/inspector/ops/ops.yaml")
    for f in ops_files:
        if os.path.exists(f):
            os.remove(f)

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Create input tensors
        shape = [32, 64]
        torch_a = torch.rand(shape, dtype=torch.bfloat16)
        torch_b = torch.rand(shape, dtype=torch.bfloat16)
        torch_c = torch.rand(shape, dtype=torch.bfloat16)

        # Convert to ttnn tensors
        tensor_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        tensor_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
        tensor_c = ttnn.from_torch(torch_c, layout=ttnn.TILE_LAYOUT, device=device)

        print("=== Starting operation chain ===")

        # Operation 1: multiply tensor_a by scalar
        print("Step 1: tensor_a * 2.5")
        result1 = ttnn.mul(tensor_a, 2.5)

        # Operation 2: multiply result1 and tensor_b
        print("Step 2: result1 * tensor_b")
        result2 = ttnn.mul(result1, tensor_b)

        # Operation 3: subtract tensor_c from result2
        print("Step 3: result2 - tensor_c")
        result3 = ttnn.subtract(result2, tensor_c)

        # Operation 4: add scalar to result3 (this will hang if infinite loop is present)
        print("Step 4: result3 + 1.0 (may hang if infinite loop is active)")
        final_result = ttnn.add(result3, 1.0)

        print("=== Operation chain complete ===")

        # Convert final result back to torch for verification
        torch_final = ttnn.to_torch(final_result)

        # Compute expected result using torch
        torch_expected = (torch_a * 2.5) * torch_b - torch_c + 1.0

        # Verify the computation is correct
        assert_with_pcc(torch_expected, torch_final, pcc=0.99)

        print("=== Test completed successfully ===")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
