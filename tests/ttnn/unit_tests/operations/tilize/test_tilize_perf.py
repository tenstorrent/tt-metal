# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf-focus shapes (feature_spec LOOSE_CASES, bf16 interleaved DRAM). Run under --profile."""

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize


@pytest.mark.parametrize(
    "shape", [(1, 1, 16384, 64), (1, 1, 16384, 32), (1, 1, 32768, 64), (1, 1, 128, 64), (1, 1, 3520, 64)]
)
def test_tilize_perf(device, shape):
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(tt_in)

    assert torch.equal(ttnn.to_torch(out), x)
