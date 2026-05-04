# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PCC sweep over `hidden_embedding_dim` for `ttnn.embedding` to isolate the
miscompile observed at D=10752 on tt-xla (Gemma-4 per-layer embedding).

Sweep originally taken on tt-xla `main`:

    D=1024   tile_count=32   PCC ~ 1.0
    D=10240  tile_count=320  PCC ~ 1.0
    D=10752  tile_count=336  PCC ~ 0.02   <-- bad
    D=16384  tile_count=512  PCC ~ 1.0

This test exercises `ttnn.embedding` directly (no PJRT / no graph compiler
between python and the kernel) so a failure here points at the ttnn
embedding op or its tt-metal kernel.
"""

import pytest
import torch
import ttnn
from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "hidden_embedding_dim",
    [
        1024,  # tile_count = 32   (power-of-two)
        10240,  # tile_count = 320  (non-pow2, known good)
        10752,  # tile_count = 336  (Gemma-4 per-layer embedding) -- expected to fail
        16384,  # tile_count = 512  (power-of-two)
    ],
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sentence_size", [64])
@pytest.mark.parametrize("vocabulary_size", [256])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_embedding_dim_sweep_d10752(
    device,
    hidden_embedding_dim,
    batch_size,
    sentence_size,
    vocabulary_size,
    dtype,
    input_mem_config,
    output_mem_config,
    layout,
):
    torch.manual_seed(0)

    torch_input_tensor = torch.randint(0, vocabulary_size, (batch_size, sentence_size))
    torch_weights = torch_random((vocabulary_size, hidden_embedding_dim), -0.1, 0.1, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.embedding(torch_input_tensor, torch_weights)

    input_tensor = ttnn.to_device(ttnn.from_torch(torch_input_tensor), device, memory_config=input_mem_config)
    weights = ttnn.to_device(ttnn.from_torch(torch_weights, dtype=dtype), device, memory_config=input_mem_config)

    output_tensor = ttnn.embedding(input_tensor, weights, memory_config=output_mem_config, layout=layout)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
