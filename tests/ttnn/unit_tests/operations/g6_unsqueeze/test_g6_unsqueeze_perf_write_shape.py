# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 perf lamp: output write transfer shape. Run under `run_safe_pytest.sh --profile`;
each case issues exactly one g6_unsqueeze dispatch, in parametrize order, so CSV row i is case i.
Shapes are > 32 tiles (the perf-measurement floor): 440 tiles (4/core) and 3520 tiles (32/core)."""

import pytest
import torch
import ttnn

from ttnn.operations.g6_unsqueeze import g6_unsqueeze
import ttnn.operations.g6_unsqueeze.g6_unsqueeze_program_descriptor as g6u_pd

VARIANTS = [("segment", 1), ("segment", 2), ("face", 1), ("face", 2), ("tile", 2)]


@pytest.mark.parametrize("variant", VARIANTS, ids=[f"{w}-h{h}" for w, h in VARIANTS])
@pytest.mark.parametrize("w", [14080, 112640])
def test_perf_write_shape(device, monkeypatch, w, variant):
    write_shape, halves = variant
    monkeypatch.setattr(g6u_pd, "WRITE_SHAPE", write_shape)
    monkeypatch.setattr(g6u_pd, "HALVES_PER_RISC", halves)
    x = torch.randn((1, w), dtype=torch.float32)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(g6_unsqueeze(tt_x, -1))
    assert torch.equal(got.contiguous().view(torch.int32), x.unsqueeze(-1).contiguous().view(torch.int32))
