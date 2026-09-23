# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf-measurement shapes for tilize (run with run_safe_pytest.sh --profile).

Correctness is still asserted; the timing comes from the profiler CSV.
"""
import os

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 16384, 64),  # PERF FOCUS (LOOSE_CASES entry 1)
        (1, 1, 16384, 32),
        (1, 1, 32768, 64),
        (1, 1, 128, 64),
        (1, 1, 2048, 64),
    ],
    ids=lambda s: "x".join(map(str, s)),
)
def test_tilize_perf_shape(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = tilize(t)
    result = ttnn.to_torch(out)
    # TILIZE_ABLATION=1: payload-stubbed kernels (ablation profiling) produce wrong
    # values by construction; a failing test would suppress the device profile dump.
    if os.environ.get("TILIZE_ABLATION") != "1":
        assert torch.equal(result, x)
