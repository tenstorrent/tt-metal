# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3: empirical DRAM roofline for the prefill shapes — a profiled ttnn.clone moves the same
bytes as rms_norm (read x once, write out once) with no compute. Run with --profile and compare."""

import pytest
import torch
import ttnn

from .test_rms_norm_prefill_perf import PREFILL


@pytest.mark.parametrize("rows,W", PREFILL, ids=lambda v: str(v))
def test_clone_roofline(device, rows, W):
    x = torch.randn(1, 1, rows, W, dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.clone(tx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(out), x)
