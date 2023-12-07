# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn


@pytest.mark.parametrize("num_tensors", [1, 2])
@pytest.mark.parametrize("h", [1024])
@pytest.mark.parametrize("w", [1024])
def test_reallocate(device, num_tensors, h, w):
    torch_tensors = [torch.rand((h, w), dtype=torch.bfloat16) for _ in range(num_tensors)]

    tt_tensors = []
    for torch_tensor in torch_tensors:
        tt_tensor = ttnn.from_torch(torch_tensor)
        tt_tensor = ttnn.to_device(tt_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        tt_tensors.append(tt_tensor)

    reallocated_tt_tensors = []
    for tt_tensor in tt_tensors:
        tt_tensor = ttnn.reallocate(tt_tensor)
        reallocated_tt_tensors.append(tt_tensor)

    for torch_tensor, tt_tensor in zip(torch_tensors, reallocated_tt_tensors):
        tt_tensor = ttnn.from_device(tt_tensor)
        tt_tensor = ttnn.to_torch(tt_tensor)

        assert torch.allclose(torch_tensor, tt_tensor)
