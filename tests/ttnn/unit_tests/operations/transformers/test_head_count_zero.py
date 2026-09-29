# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


# A head count of 0 used to reach an integer division on the host, which killed the process with
# SIGFPE instead of raising.


def test_split_query_key_value_and_split_heads_num_heads_zero(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 32, 192).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "num_heads must be greater than 0"):
        ttnn.transformer.split_query_key_value_and_split_heads(x, num_heads=0)


def test_nlp_create_qkv_heads_num_heads_zero(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 32, 192).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "num_q_heads must be greater than 0"):
        ttnn.experimental.nlp_create_qkv_heads(x, num_heads=0)
