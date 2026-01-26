# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers

import ttnn


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 5632, 64)],
)
@pytest.mark.parametrize(
    "input_dtype",
    [
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ],
)
def test_to_with_different_dtypes_and_trace(device, shape, input_dtype):
    torch.random.manual_seed(1234)
    torch_input_tensor = torch.randn(shape).bfloat16().float()
    interleaved_memory_config = ttnn.DRAM_MEMORY_CONFIG
    with ttnn.tracer.trace():
        tensor = ttnn.from_torch(
            torch_input_tensor,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=interleaved_memory_config,
            dtype=input_dtype,
        )
    assert True
