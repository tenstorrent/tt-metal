# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""onorm DM-granularity sweep — picks DM_BLOCK_TILES / depths on evidence.

DO NOT DELETE. Run under `--profile` and read DEVICE KERNEL DURATION [ns].
Each case is also a correctness check, so the sweep cannot "win" by being wrong.

Why this file exists: the per-phase zone in onorm_compute.cpp wraps each helper
call, and a helper's `cb_wait_front` is INSIDE that zone. So a starved pipeline
shows up as a huge "compute phase", which is exactly how the phase-1 defaults
mis-read as compute-bound. This sweep measures the whole kernel per DM setting
across several shapes so the chosen default is not an artifact of one L1 layout.
"""

import pytest
import torch

import ttnn
import ttnn.operations.onorm.onorm_program_descriptor as pd
from ttnn.operations.onorm import default_compute_kernel_config, onorm

from tests.ttnn.utils_for_testing import assert_with_pcc

HV, V = 32, 128
FLAT = HV * V
PCC = 0.995


@pytest.fixture
def restore_knobs():
    saved = {k: getattr(pd, k) for k in ("DM_BLOCK_TILES", "DM_DEPTH", "O_DEPTH")}
    yield
    for k, v in saved.items():
        setattr(pd, k, v)


def _run(device, batch, tokens):
    torch.manual_seed(42)
    t_o = torch.randn(batch, tokens, HV, V, dtype=torch.bfloat16)
    t_g = torch.randn(batch, tokens, FLAT, dtype=torch.bfloat16)
    t_w = torch.randn(1, 1, 1, V, dtype=torch.bfloat16)
    o = ttnn.from_torch(t_o, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(t_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w = ttnn.from_torch(t_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = onorm(o, g, w, compute_kernel_config=default_compute_kernel_config())
    f = t_o.to(torch.float32)
    ref = f * torch.rsqrt(f.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    ref = ref * t_w.to(torch.float32).reshape(1, 1, 1, V)
    ref = ref.reshape(batch, tokens, FLAT) * torch.sigmoid(t_g.to(torch.float32))
    assert_with_pcc(ref, ttnn.to_torch(out).to(torch.float32), PCC)


SHAPES = [(1, 64), (1, 128), (1, 640), (8, 640)]


@pytest.mark.parametrize("dm_block", [4, 8])
@pytest.mark.parametrize("batch, tokens", SHAPES, ids=lambda v: str(v))
def test_dm_block(device, restore_knobs, batch, tokens, dm_block):
    pd.DM_BLOCK_TILES = dm_block
    _run(device, batch, tokens)


@pytest.mark.parametrize("dm_block, dm_depth, o_depth", [(8, 2, 2), (8, 4, 2), (8, 2, 3), (8, 4, 3)])
def test_dm_combo(device, restore_knobs, dm_block, dm_depth, o_depth):
    pd.DM_BLOCK_TILES, pd.DM_DEPTH, pd.O_DEPTH = dm_block, dm_depth, o_depth
    _run(device, 1, 640)
