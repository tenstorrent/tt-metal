# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""wan_fused_rmsnorm_pre_allgather must not accept a memory_config it ignores.

The op exposes `memory_config`, but the prim builds the stats output spec from the input tensor's
memory config and never reads the requested one, so a caller that asked for another placement got
the input's with no error.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _input(device):
    torch_input = torch.randn(1, 1, 32, 64)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input


def test_wan_fused_rmsnorm_pre_allgather_rejects_memory_config(device, expect_error):
    _, tt_input = _input(device)
    requested = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    with expect_error(RuntimeError, "does not support memory_config"):
        ttnn.experimental.wan_fused_rmsnorm_pre_allgather(tt_input, memory_config=requested)


def test_wan_fused_rmsnorm_pre_allgather_default_config(device):
    # Control that passes with and without the fix: no memory_config, the computed stats are used.
    torch_input, tt_input = _input(device)
    stats = ttnn.experimental.wan_fused_rmsnorm_pre_allgather(tt_input)
    assert list(stats.shape) == [1, 1, 32, 32]
    sum_of_squares = ttnn.to_torch(stats).float()[..., 0].reshape(1, 1, 32, 1)
    expected = (torch_input.float() ** 2).sum(dim=-1, keepdim=True)
    assert_with_pcc(expected, sum_of_squares, 0.999)
