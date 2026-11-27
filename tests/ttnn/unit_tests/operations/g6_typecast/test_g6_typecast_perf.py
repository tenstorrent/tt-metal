# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 — perf-flagged R2 profile (F322 / F317, bf16 -> uint8 ROW_MAJOR DRAM).

Bit-exact check plus a single device dispatch, so `run_safe_pytest.sh --profile` yields one device-kernel row
per case. Also used as the correctness pin for the vectorized compaction / expansion paths (odd row counts per
core, rows < bank count).
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast

op_mod = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")

RM = ttnn.ROW_MAJOR_LAYOUT


def _run(device, shape, mem=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(3)
    x = (torch.rand(shape) * 300 - 20).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=mem)
    tt_out = g6_typecast(tt_x, ttnn.uint8)
    got = ttnn.to_torch(tt_out).to(torch.int64) & 0xFF
    exp = x.to(torch.uint8).to(torch.int64) & 0xFF
    n_bad = int((got != exp).sum())
    assert n_bad == 0, f"{n_bad}/{got.numel()} differ"
    return op_mod.last_plan


@pytest.mark.parametrize(
    "shape",
    [(1, 2048, 2048, 3), (1, 1024, 1024, 3), (1, 7, 13, 3), (5, 3), (1, 333, 77, 3), (2, 50, 6)],
    ids=["F322", "F317", "odd_rows", "tiny", "ragged_core", "w6"],
)
def test_g6_typecast_rm_perf_profile(device, shape):
    _run(device, shape)
