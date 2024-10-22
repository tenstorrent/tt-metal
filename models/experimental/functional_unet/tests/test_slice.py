# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.functional_unet.tt.slice import chunk_batch_dim
from models.experimental.functional_unet.tests.common import verify_with_pcc


@pytest.mark.parametrize(
    "B, C, H, W, slice_length",
    [
        (2, 2, 32, 32, 2),
        (16, 4, 1024, 512, 2),
        (16, 4, 1024, 512, 1),
    ],
)
def test_unet_slice(B, C, H, W, slice_length, device, use_program_cache, reset_seeds):
    torch_input_tensor = torch.rand(B, C, H, W)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    x = ttnn.to_device(input_tensor, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output = []
    for slice in chunk_batch_dim(x, memory_config=ttnn.L1_MEMORY_CONFIG, slice_length=slice_length):
        output.append(ttnn.to_memory_config(slice, ttnn.DRAM_MEMORY_CONFIG))
    output = [ttnn.to_torch(x) for x in output]
    output = torch.concat(output, dim=0)

    verify_with_pcc(torch_input_tensor, output, 1.0)
