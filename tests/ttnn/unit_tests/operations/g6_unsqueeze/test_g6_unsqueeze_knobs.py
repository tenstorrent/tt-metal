# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Knob coverage for g6_unsqueeze (implementer's own tests; the acceptance test is the spec).

Every Blocking-Model knob is turned off its production value and the result graded bit for bit:
core count (grid-sync lamp), halves per RISC (RISC-split lamp), block cap (streaming, ragged last
block), ragged last tile (W % 32 != 0, incl. a half-empty last tile), ranks 1 and 3.
Also used for the perf lamps under `--profile`.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_unsqueeze import g6_unsqueeze
import ttnn.operations.g6_unsqueeze.g6_unsqueeze_program_descriptor as g6u_pd


def _run(device, shape, dim=-1):
    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.float32)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = g6_unsqueeze(tt_x, dim)
    got = ttnn.to_torch(out)
    want = torch.unsqueeze(x, dim)
    assert list(got.shape) == list(want.shape)
    assert torch.equal(got.contiguous().view(torch.int32), want.contiguous().view(torch.int32))
    return g6u_pd.select_plan(tt_x)


@pytest.mark.parametrize("halves", [1, 2])
@pytest.mark.parametrize("max_cores", [None, 8, 4, 2, 1])
def test_lamp_cores_halves_p1(device, monkeypatch, max_cores, halves):
    monkeypatch.setattr(g6u_pd, "MAX_CORES", max_cores)
    monkeypatch.setattr(g6u_pd, "HALVES_PER_RISC", halves)
    plan = _run(device, (1, 512))
    assert plan.regime == "resident"


@pytest.mark.parametrize("halves", [1, 2])
@pytest.mark.parametrize("shape", [(1, 500), (1, 17), (1, 16), (1, 1), (40,), (1, 1, 3000), (1, 70000)])
def test_ragged_and_ranks(device, monkeypatch, shape, halves):
    monkeypatch.setattr(g6u_pd, "HALVES_PER_RISC", halves)
    _run(device, shape)


@pytest.mark.parametrize("halves", [1, 2])
@pytest.mark.parametrize("max_cores,max_block", [(3, 2), (1, 5), (5, 4)])
def test_streaming(device, monkeypatch, max_cores, max_block, halves):
    monkeypatch.setattr(g6u_pd, "MAX_CORES", max_cores)
    monkeypatch.setattr(g6u_pd, "MAX_BLOCK_W_TILES", max_block)
    monkeypatch.setattr(g6u_pd, "HALVES_PER_RISC", halves)
    plan = _run(device, (1, 1000), dim=2)
    assert plan.regime == "streaming" and plan.buffer_depth == 2
