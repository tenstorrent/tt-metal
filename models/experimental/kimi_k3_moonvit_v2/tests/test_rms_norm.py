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
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.rms_norm import RMSNorm

# Per-layer norms, under `vision_tower.encoder.blocks.<layer>`.
ENCODER_NORMS = ["norm0", "norm1"]


def check_rms_norm(mesh_device, torch_norm, eps: float, rows: MeshRows) -> None:
    (dim,) = torch_norm.normalized_shape
    tt = RMSNorm(dim, eps=eps, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config(mesh_device))
    tt.load_torch_state_dict(torch_norm.state_dict())

    x = random_tensor(rows.n, dim)
    with torch.no_grad():
        golden = torch_norm(rows.take(x))

    out = tt(rows.shard(x, mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.01)


@P150_AND_RING
@REFERENCE_LAYERS
@ROW_GRIDS
@pytest.mark.parametrize("name", ENCODER_NORMS)
def test_encoder_rms_norm(mesh_device, sp_axis, num_links, config, reference, size, layer, name):
    torch_norm = reference.vision_tower.encoder.blocks[layer].get_submodule(name)
    assert torch_norm.eps is None
    ((_, h, w),) = row_grid(size)
    check_rms_norm(mesh_device, torch_norm, config.norm_eps, MeshRows(h * w, config.max_patches, sp_axis))


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_final_rms_norm(mesh_device, sp_axis, num_links, config, reference, size):
    torch_norm = reference.vision_tower.encoder.final_layernorm
    assert torch_norm.eps is None
    ((_, h, w),) = row_grid(size)
    check_rms_norm(mesh_device, torch_norm, config.norm_eps, MeshRows(h * w, config.max_patches, sp_axis))


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_projector_rms_norm(mesh_device, sp_axis, num_links, config, reference, size):
    torch_norm = reference.mm_projector.post_norm
    assert torch_norm.eps == config.projector_norm_eps
    ((_, h, w),) = row_grid(size)
    rows = MeshRows(h * w // config.merge_area, config.max_patches // config.merge_area, sp_axis)
    check_rms_norm(mesh_device, torch_norm, config.projector_norm_eps, rows)
