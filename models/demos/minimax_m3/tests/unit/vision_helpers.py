# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared parts of the vision encoder unit tests (test_vision_*_vs_ref.py): the image grids, the host to
device copies and the PCC check. The reference model and its activations are fixtures in conftest.py."""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3.tt.vision.common import mesh_composer, mesh_mapper

# Patch grid (t, h, w) of one image: its size in pixels divided by the 14 pixel patch. The ids name the grid in
# patches, never as AxB: the Galaxy leg selects -k "8x4", which the pixel size "448x448" would contain.
VISION_GRIDS = {
    "h16w16": (1, 16, 16),  # 224x224: 256 patches, 64 vision tokens
    "h32w32": (1, 32, 32),  # 448x448: 1024 patches
    "h48w48": (1, 48, 48),  # 672x672, the largest the processor makes: 2304 patches, 576 tokens
    "h34w46": (1, 34, 46),  # 640x480 like models/sample_data/huggingface_cat_image.jpg: 1564 patches, not whole tiles
}
# The 1-card CI leg runs h16w16 only, to stay inside its 5 minute timeout. M3_VISION_ALL_GRIDS=1 adds the others.
# They are left out rather than skipped, so the leg shows no skipped vision tests.
CI_GRIDS = ("h16w16",)


def parametrize_vision_grids():
    """Parametrize grid_thw over CI_GRIDS, or over all of VISION_GRIDS when M3_VISION_ALL_GRIDS=1."""
    run_all = os.getenv("M3_VISION_ALL_GRIDS") == "1"
    params = [
        pytest.param(grid, id=grid_id) for grid_id, grid in VISION_GRIDS.items() if run_all or grid_id in CI_GRIDS
    ]
    return pytest.mark.parametrize("grid_thw", params)


def grid_name(grid_thw) -> str:
    t, h, w = grid_thw
    return f"h{h}w{w}" if t == 1 else f"t{t}h{h}w{w}"


def to_device(x: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """(L, D) host tensor -> (1, 1, L, D) bf16 tile tensor in DRAM, replicated over the mesh."""
    return ttnn.from_torch(
        x.to(torch.bfloat16).reshape(1, 1, *x.shape).contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper(mesh_device),
    )


def to_host(x: ttnn.Tensor, mesh_device, shape) -> torch.Tensor:
    """The first device's copy of x, as an fp32 host tensor of the given shape."""
    out = ttnn.to_torch(x, mesh_composer=mesh_composer(mesh_device, dim=0))
    return out[:1].reshape(shape).float()


def assert_pcc(name: str, ref: torch.Tensor, out: torch.Tensor, threshold: float):
    passing, pcc_msg = comp_pcc(ref, out, threshold)
    rel_l2 = (torch.linalg.norm(out - ref) / torch.linalg.norm(ref)).item()
    logger.info(f"[{name}] {comp_allclose(ref, out)} {pcc_msg} relL2={rel_l2:.4f}")
    assert passing, f"{name}: PCC below {threshold}: {pcc_msg}"
