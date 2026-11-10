# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def test_nlp_kv_cache_load_slice_honours_memory_config(device):
    # The prim always builds an L1 HEIGHT_SHARDED output, so a DRAM request used to be dropped and
    # the caller received a sharded tensor anyway.
    torch_input = torch.randn(1, 1, 64, 96, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    default_output = ttnn.experimental.nlp_kv_cache_load_slice(input_tensor, seq_len_start=0, seq_len_end=64)
    dram_output = ttnn.experimental.nlp_kv_cache_load_slice(
        input_tensor, seq_len_start=0, seq_len_end=64, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    assert default_output.memory_config().is_sharded()
    assert dram_output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert dram_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert torch.equal(ttnn.to_torch(default_output), ttnn.to_torch(dram_output))
