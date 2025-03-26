# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# @pytest.mark.parametrize(
#     "dim",
#     [
#         (1,)
#     ]
# )
def test_cumprod(device):
    torch.manual_seed(2005)

    input_tensor = torch.randn(2, 2, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    print("INPUT:::::", ttnn_tensor)
    result = ttnn.experimental.cumprod(ttnn_tensor, 0)
    print("RESULT::::", result)
