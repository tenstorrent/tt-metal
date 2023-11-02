# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [2 * 32])
def test_free(device, h, w):
    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor)

    with pytest.raises(RuntimeError) as exception:
        ttnn.free(input_tensor)
    assert "Cannot deallocate tensor with borrowed storage!" in str(exception.value)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = input_tensor + input_tensor

    # Create a reference to the same storage by using reshape which will create a new flyweight
    # (If reshape operation changes, then this test might need to be updated)
    output_tensor_reference = ttnn.reshape(output_tensor, (1, 1, h, w))

    ttnn.free(output_tensor)
    with pytest.raises(RuntimeError) as exception:
        print(output_tensor_reference)
    assert "Buffer must be allocated on device!" in str(exception.value)
