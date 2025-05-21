# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from .format_arg_mapping import format_dict
from .format_config import DataFormat


def tilize(original_tensor, stimuli_format=DataFormat.Float16_b):

    if original_tensor.size(0) != 1024:
        raise ValueError("Input tensor must have 1024 elements.")

    matrix = original_tensor.view(32, 32)

    f0 = matrix[:16, :16]
    f1 = matrix[:16, 16:32]
    f2 = matrix[16:32, :16]
    f3 = matrix[16:32, 16:32]

    result = torch.cat((f0.reshape(-1), f1.reshape(-1), f2.reshape(-1), f3.reshape(-1)))

    return result.to(
        dtype=(
            format_dict[stimuli_format]
            if stimuli_format in [DataFormat.Float16_b, DataFormat.Float16]
            else torch.float32
        )
    )


def untilize(tilized_tensor, stimuli_format=DataFormat.Float16_b):

    tilized_tensor = tilized_tensor.view(-1)

    f0 = tilized_tensor[:256].view(16, 16)
    f1 = tilized_tensor[256:512].view(16, 16)
    f2 = tilized_tensor[512:768].view(16, 16)
    f3 = tilized_tensor[768:].view(16, 16)

    top = torch.cat((f0, f1), dim=1)
    bottom = torch.cat((f2, f3), dim=1)

    original_tensor = torch.cat((top, bottom), dim=0).view(1024)

    return original_tensor.to(
        dtype=(
            format_dict[stimuli_format]
            if stimuli_format in [DataFormat.Float16_b, DataFormat.Float16]
            else torch.float32
        )
    )
