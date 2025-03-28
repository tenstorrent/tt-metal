# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def to_channel_last_ttnn(torch_tensor, dtype, device, memory_config):
    torch_tensor = torch.permute(torch_tensor, (0, 2, 3, 1))
    ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype,
        device=device,
        memory_config=memory_config,
    )
    return ttnn_tensor


def from_channel_last_ttnn(ttnn_tensor, output_shape):
    torch_tensor = ttnn.to_torch(ttnn_tensor)
    torch_tensor = torch_tensor.reshape(output_shape)
    torch_tensor = torch.permute(torch_tensor, (0, 3, 1, 2))
    return torch_tensor
