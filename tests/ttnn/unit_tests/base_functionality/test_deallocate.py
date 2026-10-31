# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_deallocate(device, h, w, expect_error):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = input_tensor + input_tensor

    # Create input_tensor_a reference to the same storage by using reshape which will create a new flyweight
    # (If reshape operation changes, then this test might need to be updated)
    output_tensor_reference = ttnn.reshape(output_tensor, (h, w))

    ttnn.deallocate(output_tensor)
    with expect_error(RuntimeError, r"Input Tensor A is not allocated"):
        output_tensor_reference + output_tensor_reference


def test_deallocate_buffers_then_stale_tensor_does_not_free_live_range(device):
    shape = (1, 1, 32, 32)
    stale = ttnn.from_torch(torch.rand(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    stale_address = stale.buffer_address()
    ttnn.device.DeallocateBuffers(device)

    torch_live = torch.rand(shape, dtype=torch.bfloat16)
    live = ttnn.from_torch(torch_live, layout=ttnn.TILE_LAYOUT, device=device)
    assert live.buffer_address() == stale_address
    ttnn.deallocate(stale)
    other = ttnn.from_torch(torch.rand(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    assert other.buffer_address() != live.buffer_address()
    assert torch.equal(ttnn.to_torch(live), torch_live)
