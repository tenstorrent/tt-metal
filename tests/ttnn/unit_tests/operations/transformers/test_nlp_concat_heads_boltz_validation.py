# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def _concat_heads_boltz_ref(x):
    num_heads, s0, s1, head_dim = x.shape
    return x.permute(1, 2, 0, 3).reshape(1, s0, s1, num_heads * head_dim)


def test_nlp_concat_heads_boltz_rejects_unaligned_head_dim(device, expect_error):
    # The factory strides the copy by padded head tiles while the output hidden dim is sized from the
    # logical head dim, so a logical head dim of 48 (padded 64) wrote every head at a 64-column stride.
    x = torch.randn((2, 32, 32, 48), dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "logical head dim"):
        ttnn.experimental.nlp_concat_heads_boltz(xt)


def test_nlp_concat_heads_boltz_aligned_head_dim_control(device):
    # Control: a tile-aligned head dim still concatenates exactly.
    x = torch.randn((2, 32, 32, 64), dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.experimental.nlp_concat_heads_boltz(xt)

    assert torch.equal(_concat_heads_boltz_ref(x), ttnn.to_torch(output))
