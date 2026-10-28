# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2: attribute the small-shape (<=1 tile/core) delta of the bank-affine split.
assign=affine -> the op as graduated; assign=contig -> SAME kernels, host emits one all-bank group
(bank[k]=k), i.e. the pre-Perf-2 contiguous page ranges through the new segment walker."""
import pytest
import torch
import ttnn
from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

_orig = pd._bank_affine_segments


def _contig(device, cores, core_tiles, n_pages):
    nb = device.dram_grid_size().x
    segs, q = [], 0
    for t in core_tiles:
        segs.append([(0, q, t)] if t else [])
        q += t
    return [list(range(nb))], segs


@pytest.mark.parametrize("rep", range(3))
@pytest.mark.parametrize("shape", [(1, 1, 32, 32), (64, 128), (2, 64, 256)], ids=str)
@pytest.mark.parametrize("assign", ["affine", "contig"])
def test_small(device, assign, shape, rep, monkeypatch):
    monkeypatch.setattr(pd, "_bank_affine_segments", _contig if assign == "contig" else _orig)
    t = torch.randn(shape, dtype=torch.bfloat16) * 2
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.generic_op([x, out], pd.create_program_descriptor(x, out, -1.0, 1.0))
    ref = (torch.clamp(t.float(), -1, 1) ** 2).bfloat16()
    assert torch.equal(ttnn.to_torch(out), ref)
