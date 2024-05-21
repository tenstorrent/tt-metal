# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from tests.ttnn.utils_for_testing import check_with_pcc
import ttnn


parameters = {
    "batch_size": [1, 2, 4],
    "c": [320, 64],
    "h": [64, 32],
    "w": [64, 16],
    "scale_h": [2, 4],
    "scale_w": [2, 4],
}


def run(
    batch_size,
    c,
    h,
    w,
    scale_h,
    scale_w,
    device,
) -> Tuple[bool, Optional[str]]:
    torch.manual_seed(0)
    input = torch.rand(batch_size, h, w, c, dtype=torch.bfloat16)
    tt_input = input.permute(0, 3, 1, 2)

    scale_factor = (scale_h, scale_w)
    m = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = m(tt_input)
    torch_result = torch_result.permute(0, 2, 3, 1)

    ## ttnn uses NHWC, so need to set scale_factor_c = 1
    scale_factor = (scale_h, scale_w, 1)
    input_tensor = ttnn.from_torch(input, device=device)
    output_tensor = ttnn.upsample(input_tensor, scale_factor)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_result, output_tensor)
