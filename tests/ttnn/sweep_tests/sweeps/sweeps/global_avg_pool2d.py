# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from tests.ttnn.utils_for_testing import check_with_pcc
import ttnn


parameters = {
    "input_shape": [[1, 2048, 7, 7], [1, 64, 1, 32]],
    "dtype": [ttnn.bfloat16],
}


def run(
    input_shape,
    dtype,
    device,
) -> Tuple[bool, Optional[str]]:
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape)
    torch_output_tensor = torch.nn.functional.global_avg_pool2d(torch_input_tensor, (1, 1))

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # ttnn operates on channels-last tensors
    input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    return check_with_pcc(torch_output_tensor, output_tensor)
