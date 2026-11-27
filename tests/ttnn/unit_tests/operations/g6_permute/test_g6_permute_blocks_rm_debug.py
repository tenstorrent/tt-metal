# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Perf experiments for g6_permute blocks_rm_gather on FLUX_74 [1, 64, 128, 32, 2, 2].

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`: each case is one dispatch, read the
device kernel duration from the ops CSV. Every case also checks the output bit for bit.
- grid: the same work over fewer cores separates a per-core read-issue limit (time grows as cores
  shrink) from a DRAM-side request limit (time stays flat); alt1_2 repeats it with half the page
  reads on the second NoC, which reaches every DRAM bank through a second port. Blocks of 2 units, depth 4.
DO NOT DELETE.
"""

import importlib
from fractions import Fraction

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute

op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
regime_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_blocks_rm_gather")

BLOCKS = (0, 1, 4, 2, 5, 3)
FLUX_74 = (1, 64, 128, 32, 2, 2)


@pytest.fixture
def fresh_cache():
    op_module._PROGRAM_CACHE.clear()
    yield
    op_module._PROGRAM_CACHE.clear()


def _run(shape, device):
    torch.manual_seed(5)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(g6_permute(ttnn_input, BLOCKS))
    expected = torch.permute(torch_input, BLOCKS).contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("alt_share", [Fraction(0), Fraction(1, 2)], ids=lambda f: f"alt{f.numerator}_{f.denominator}")
@pytest.mark.parametrize("rows", [10, 5, 2, 1], ids=lambda r: f"grid_rows{r}")
def test_grid_scaling(rows, alt_share, device, fresh_cache, monkeypatch):
    full = device.compute_with_storage_grid_size()
    monkeypatch.setattr(regime_module, "worker_grid", lambda _device: ttnn.CoreCoord(full.x, rows))
    monkeypatch.setattr(regime_module, "READ_ALT_NOC_SHARE", alt_share)
    monkeypatch.setattr(regime_module, "BLOCK_H_LIMIT", 2)
    monkeypatch.setattr(regime_module, "INPUT_DEPTH", 4)
    _run(FLUX_74, device)
