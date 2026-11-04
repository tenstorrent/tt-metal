# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `RMSNorm` against each `nn.RMSNorm` of the reference, at the row count it sees in the model.

The encoder norms have `eps=None`, which torch resolves to `finfo(bf16).eps` for bf16 input, so the
golden runs in bf16 and the TT norm uses `config.norm_eps`.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    random_tensor,
    assert_quality,
    row_grid,
    ROW_GRIDS,
    MeshRows,
    P150_AND_RING,
    REFERENCE,
    REFERENCE_LAYERS,
    SIMULATOR_GRID,
    SIMULATOR,
    simulator_config,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.rms_norm import RMSNorm

# Per-layer norms, under `vision_tower.encoder.blocks.<layer>`.
ENCODER_NORMS = pytest.mark.parametrize("name", ["norm0", "norm1"])


def check_rms_norm(mesh_device, torch_norm, eps: float, rows: MeshRows) -> None:
    (dim,) = torch_norm.normalized_shape
    tt = RMSNorm(dim, eps=eps, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config(mesh_device))
    tt.load_torch_state_dict(torch_norm.state_dict())

    x = random_tensor(rows.n, dim)
    with torch.no_grad():
        golden = torch_norm(rows.take(x))

    out = tt(rows.shard(x, mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.01)


def check_encoder_rms_norm(mesh_device, sp_axis, config, reference, layer, name, grid) -> None:
    """`grid` padded to `config.max_patches`, the ring's length; one chip runs its quarter (`MeshRows`)."""
    torch_norm = reference.vision_tower.encoder.blocks[layer].get_submodule(name)
    assert torch_norm.eps is None
    ((_, h, w),) = grid
    check_rms_norm(mesh_device, torch_norm, config.norm_eps, MeshRows(h * w, config.max_patches, sp_axis))


def check_final_rms_norm(mesh_device, sp_axis, config, reference, grid) -> None:
    """`grid` padded to `config.max_patches`, the ring's length; one chip runs its quarter (`MeshRows`)."""
    torch_norm = reference.vision_tower.encoder.final_layernorm
    assert torch_norm.eps is None
    ((_, h, w),) = grid
    check_rms_norm(mesh_device, torch_norm, config.norm_eps, MeshRows(h * w, config.max_patches, sp_axis))


def check_projector_rms_norm(mesh_device, sp_axis, config, reference, grid) -> None:
    """The merged tokens of `grid`, padded to `config.max_patches / 4` (the ring's length after merging);
    one chip runs its quarter (`MeshRows`)."""
    torch_norm = reference.mm_projector.post_norm
    assert torch_norm.eps == config.projector_norm_eps
    ((_, h, w),) = grid
    rows = MeshRows(h * w // config.merge_area, config.max_patches // config.merge_area, sp_axis)
    check_rms_norm(mesh_device, torch_norm, config.projector_norm_eps, rows)


@P150_AND_RING
@REFERENCE_LAYERS
@ROW_GRIDS
@ENCODER_NORMS
def test_encoder_rms_norm(mesh_device, sp_axis, num_links, config, reference, size, layer, name):
    check_encoder_rms_norm(mesh_device, sp_axis, config, reference, layer, name, row_grid(size))


@SIMULATOR
@ENCODER_NORMS
def test_encoder_rms_norm_small(mesh_device, sp_axis, num_links, config, reference, name):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS."""
    check_encoder_rms_norm(mesh_device, sp_axis, simulator_config(config), reference, 0, name, SIMULATOR_GRID)


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_final_rms_norm(mesh_device, sp_axis, num_links, config, reference, size):
    check_final_rms_norm(mesh_device, sp_axis, config, reference, row_grid(size))


@SIMULATOR
def test_final_rms_norm_small(mesh_device, sp_axis, num_links, config, reference):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS."""
    check_final_rms_norm(mesh_device, sp_axis, simulator_config(config), reference, SIMULATOR_GRID)


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_projector_rms_norm(mesh_device, sp_axis, num_links, config, reference, size):
    check_projector_rms_norm(mesh_device, sp_axis, config, reference, row_grid(size))


@SIMULATOR
def test_projector_rms_norm_small(mesh_device, sp_axis, num_links, config, reference):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS patches."""
    check_projector_rms_norm(mesh_device, sp_axis, simulator_config(config), reference, SIMULATOR_GRID)
