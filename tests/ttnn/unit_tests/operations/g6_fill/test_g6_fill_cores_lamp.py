# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf lamps of g6_fill: device kernel time against MAX_FILL_CORES and FILL_ROW_SOURCE.

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`; each case makes
CALLS_PER_VARIANT calls, so the profiler CSV rows map back by position and core count.
Correctness is still asserted bit for bit on every case.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_fill import g6_fill
from ttnn.operations.g6_fill import g6_fill_program_descriptor as pd

CALLS_PER_VARIANT = 3
VALUES = [1.0, 1 / 3, -0.0, 1e-40, float("nan"), 0.1, 1.0]


def _bits(t):
    return t.contiguous().view(torch.int32)


def _run(device, n, values):
    x = ttnn.from_torch(
        torch.randn(n),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for value in values:
        g6_fill(x, value, output_tensor=x)
        want = torch.tensor(value, dtype=torch.float64).to(torch.float32).expand(n)
        assert torch.equal(_bits(ttnn.to_torch(x)), _bits(want)), f"fill_value={value!r}"


@pytest.mark.parametrize("n", [512, 207], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("max_fill_cores", [None, 1, 2, 4, 8], ids=lambda c: f"cores{c}")
def test_cores_lamp(device, n, max_fill_cores, monkeypatch):
    monkeypatch.setattr(pd, "MAX_FILL_CORES", max_fill_cores)
    _run(device, n, [1.0] * CALLS_PER_VARIANT)


@pytest.mark.parametrize("n", [512, 256, 207, 196, 185], ids=lambda n: f"N{n}")
@pytest.mark.parametrize("source", pd.FILL_ROW_SOURCES)
def test_fill_row_source_lamp(device, n, source, monkeypatch):
    monkeypatch.setattr(pd, "FILL_ROW_SOURCE", source)
    _run(device, n, VALUES[:1] * CALLS_PER_VARIANT + VALUES[1:])
