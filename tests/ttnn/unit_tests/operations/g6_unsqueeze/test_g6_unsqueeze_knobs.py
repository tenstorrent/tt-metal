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


def test_descriptor_cache_patches_addresses(device):
    """BB-07: a repeat call reuses the cached descriptor and only patches buffer addresses."""
    g6u_pd._DESCRIPTOR_CACHE.clear()
    results = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        x = torch.randn((1, 512), dtype=torch.float32)
        tt_x = ttnn.from_torch(
            x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        got = ttnn.to_torch(g6_unsqueeze(tt_x, 2))
        assert torch.equal(got.contiguous().view(torch.int32), x.unsqueeze(2).contiguous().view(torch.int32))
        results.append(tt_x)
    assert len(g6u_pd._DESCRIPTOR_CACHE) == 1


@pytest.mark.parametrize(
    "shape,want_ranges",
    [((1, 512), 1), ((1, 256), 1), ((1, 1, 4100), 1), ((1, 416), None)],  # 13 cores: prime -> row-wise fallback
)
def test_core_placement_rectangle(device, shape, want_ranges):
    """Refinement 1: the core set is one most-square rectangle when the core count factors into the
    grid (p1 4x4, p2 4x2, full grid 11x10); otherwise the row-wise fallback, still bit-exact."""
    plan = _run(device, shape)
    n_ranges = len(plan.core_ranges.ranges())
    if want_ranges is not None:
        assert n_ranges == want_ranges, g6u_pd.describe_plan(plan)
        assert plan.placement.startswith("rect")
    assert plan.core_ranges.num_cores() == plan.num_cores


@pytest.mark.parametrize("halves", [1, 2])
@pytest.mark.parametrize("write_shape", ["segment", "face", "tile"])
@pytest.mark.parametrize(
    "shape,max_cores,max_block",
    [
        ((1, 512), None, None),
        ((1, 256), None, None),
        ((1, 500), None, None),
        ((1, 17), None, None),
        ((1, 1000), 3, 2),
        ((1, 112640), None, None),
    ],
)
def test_write_shape(device, monkeypatch, write_shape, halves, shape, max_cores, max_block):
    """Refinement 2: every output write transfer shape is bit-exact on the guard set
    (p1, p2, ragged W, forced streaming with a ragged last block, a 32-tiles-per-core measurement shape)."""
    monkeypatch.setattr(g6u_pd, "WRITE_SHAPE", write_shape)
    monkeypatch.setattr(g6u_pd, "HALVES_PER_RISC", halves)
    monkeypatch.setattr(g6u_pd, "MAX_CORES", max_cores)
    monkeypatch.setattr(g6u_pd, "MAX_BLOCK_W_TILES", max_block)
    plan = _run(device, shape)
    assert plan.write_shape == write_shape
    assert plan.halves_per_risc == (2 if write_shape == "tile" else halves)
    if max_block is not None:
        assert plan.regime == "streaming"
