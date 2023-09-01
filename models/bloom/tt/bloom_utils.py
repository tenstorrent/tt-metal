"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import torch


def pad_input_tensor(tensor: torch.Tensor, value: int, multiple: int) -> torch.Tensor:
    len = tensor.shape[1]

    if len % multiple == 0:
        return tensor

    padded_len = ((len // multiple) + 1) * multiple

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor
