# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time probes of g6_slice on production cases (BB-22), run under --profile.

Knobs come from the environment so a sweep needs no edit:
  G6_BLOCK_PAGES_CAP, G6_CB_DEPTH, G6_MAX_CORES  (unset -> the op's defaults)
Every run is checked bit-exact so each profiled number belongs to a right answer.
"""

import importlib
import os

import pytest
import torch
import ttnn

from ttnn.operations.g6_slice import describe_plan, g6_slice

g6_module = importlib.import_module("ttnn.operations.g6_slice.g6_slice")

# (id, shape, starts, ends) -- production case ids of eval/golden_tests/g6_slice/scenarios.py
CASES = [
    ("c703_p1", (16, 4096, 8704), [0, 0, 4352], [16, 4096, 8704]),
    ("c730_p3_12cols", (16, 4096, 4160), [0, 0, 4096], [16, 4096, 4108]),
    ("c587_row4095", (16, 4096, 1280), [0, 4095, 0], [16, 4096, 1280]),
    ("c1175_row15_wide", (1, 1, 16, 62080), [0, 0, 15, 0], [1, 1, 16, 62080]),
    ("c179_6144_pages", (1, 2304, 3072), [0, 0, 0], [1, 2048, 3072]),
    ("c1112_40960_pages", (16, 1024, 4160), [0, 0, 0], [16, 1024, 2560]),
    ("c795_p4", (8, 4096, 8704), [0, 0, 4352], [8, 4096, 8704]),
]


def _apply_env_knobs(monkeypatch):
    for env, attr in (
        ("G6_BLOCK_PAGES_CAP", "BLOCK_PAGES_CAP"),
        ("G6_CB_DEPTH", "CB_DEPTH"),
        ("G6_MAX_CORES", "MAX_CORES"),
    ):
        if os.environ.get(env):
            monkeypatch.setattr(g6_module, attr, int(os.environ[env]))


@pytest.mark.parametrize("shape, starts, ends", [pytest.param(*c[1:], id=c[0]) for c in CASES])
def test_g6_slice_perf_knobs(device, monkeypatch, shape, starts, ends):
    _apply_env_knobs(monkeypatch)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    x_dev = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    plan = describe_plan(x_dev, starts, ends)
    print(
        f"PLAN regime={plan['regime']} cores={plan['num_cores']} block_pages={plan['block_pages']} "
        f"blocks={plan['num_blocks_max']} depth={plan['cb_depth']} l1={plan['l1_bytes_per_core']}"
    )
    out = g6_slice(x_dev, starts, ends)
    expected = x[tuple(slice(b, e) for b, e in zip(starts, ends))]
    actual = ttnn.to_torch(out)
    assert torch.equal(actual.view(torch.int16), expected.contiguous().view(torch.int16))
