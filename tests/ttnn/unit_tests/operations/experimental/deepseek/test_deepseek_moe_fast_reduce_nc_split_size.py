# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


# split_size is the divisor of the last dim, and the resulting tensor count divides the width in
# turn, so both 0 and a split larger than the width used to kill the process with SIGFPE.
@pytest.mark.parametrize("split_size", [0, 512], ids=["zero", "wider_than_input"])
def test_deepseek_moe_fast_reduce_nc_invalid_split_size(device, expect_error, split_size):
    x = ttnn.from_torch(torch.randn(2, 1, 32, 256).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must be greater than 0 and at most the last dim"):
        ttnn.experimental.deepseek_moe_fast_reduce_nc(x, 0, split_size=split_size)
