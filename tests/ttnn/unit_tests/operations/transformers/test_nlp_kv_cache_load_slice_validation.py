# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def test_nlp_kv_cache_load_slice_rejects_unaligned_head_dim(device, expect_error):
    # A logical head dim of 80 is padded to 96. The output spec took the last dim from the padded
    # shape, so the op returned logical width 96 instead of 80, adding columns the cache never held.
    torch_input = torch.randn(1, 1, 64, 80, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "logical head dim"):
        ttnn.experimental.nlp_kv_cache_load_slice(input_tensor, seq_len_start=0, seq_len_end=64)


def test_nlp_kv_cache_load_slice_aligned_head_dim_control(device):
    # Control: a tile-aligned head dim still produces the requested slice.
    torch.manual_seed(0)
    torch_input = torch.randn(1, 1, 64, 96, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.experimental.nlp_kv_cache_load_slice(input_tensor, seq_len_start=0, seq_len_end=64)

    assert list(output.shape) == [1, 1, 64, 96]
