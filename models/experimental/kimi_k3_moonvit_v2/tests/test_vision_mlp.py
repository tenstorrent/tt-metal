# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Vision_MLP` against the reference `blocks[layer].mlp`, which runs on the CPU."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    random_tensor,
    assert_quality,
    device_config,
    row_grid,
    ROW_GRIDS,
    REFERENCE_LAYERS,
    P150_AND_RING,
    SIMULATOR_GRID,
    SIMULATOR,
    MeshRows,
    simulator_config,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.vision_mlp import TT_Vision_MLP


def check_vision_mlp(mesh_device, sp_axis, config, reference, layer, grid) -> None:
    """`grid` padded to `config.max_patches`, the ring's length; one chip runs its quarter (`MeshRows`)."""
    ckc = compute_kernel_config(mesh_device)
    tt = TT_Vision_MLP(device_config(config, sp_axis), mesh_device=mesh_device, compute_kernel_config=ckc)
    tt.load_torch_state_dict(reference.vision_tower.encoder.blocks[layer].mlp.state_dict())

    _, h, w = grid[0]
    rows = MeshRows(h * w, config.max_patches, sp_axis)
    x = random_tensor(h * w, config.hidden_size)
    with torch.no_grad():
        golden = reference.vision_tower.encoder.blocks[layer].mlp(rows.take(x))

    out = tt(rows.shard(x, mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.005)


@P150_AND_RING
@REFERENCE_LAYERS
@ROW_GRIDS
def test_vision_mlp(mesh_device, sp_axis, num_links, config, reference, size, layer):
    check_vision_mlp(mesh_device, sp_axis, config, reference, layer, row_grid(size))


@SIMULATOR
def test_vision_mlp_small(mesh_device, sp_axis, num_links, config, reference):
    """Sized for ttsim: one chip's quarter of a ring padded to SIMULATOR_ROWS."""
    check_vision_mlp(mesh_device, sp_axis, simulator_config(config), reference, 0, SIMULATOR_GRID)
