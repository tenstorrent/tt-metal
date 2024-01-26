# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from tests.ttnn.utils_for_testing import check_with_pcc
import ttnn


parameters = {
    "act_shape": [[1, 7, 7, 2048], [1, 1, 32, 64]],
    "dtype": [ttnn.bfloat16],
}


def skip(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def is_expected_to_fail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


def run(
    act_shape,
    dtype,
    device,
) -> Tuple[bool, Optional[str]]:
    torch.manual_seed(0)

    act = torch.randn(act_shape, dtype=torch.bfloat16)
    ttact = ttnn.from_torch(act, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = ttnn.average_pool2d(ttact)

    out_pytorch = ttnn.to_torch(ttnn.from_device(out))

    ## reference
    act_channels_first = torch.permute(act, (0, 3, 1, 2))  # Torch operates on channels-first tensors
    golden_pytorch = torch.nn.AdaptiveAvgPool2d((1, 1))(act_channels_first)
    golden_pytorch = torch.permute(golden_pytorch, (0, 2, 3, 1))

    ## test for equivalance
    return check_with_pcc(golden_pytorch, out_pytorch)
