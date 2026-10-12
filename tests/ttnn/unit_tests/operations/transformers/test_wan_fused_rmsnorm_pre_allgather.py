# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def test_wan_fused_rmsnorm_pre_allgather_rejects_non_tile_aligned_width(device, expect_error):
    # The pre op reduces over the padded tile width, so a partial last tile would add its padding to sum(x**2).
    tt_input = ttnn.from_torch(torch.randn(1, 1, 32, 42), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Input last dimension must be the same as padded last dimension"):
        ttnn.experimental.wan_fused_rmsnorm_pre_allgather(tt_input, dtype=ttnn.bfloat16)
