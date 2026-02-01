# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def test_simple_add(device):  # ← You just ask for "device" as parameter
    # pytest automatically:
    # 1. Finds the fixture named "device"
    # 2. Calls it to get a device object
    # 3. Passes it to our test
    
    torch.manual_seed(0)
    
    # Create input tensors
    shape = (1, 1, 32, 32)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    
    # Convert to ttnn tensors
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Perform addition
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    
