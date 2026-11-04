# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `Linear` against each `nn.Linear` of the reference, at the row count it sees in the model."""

from __future__ import annotations

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    random_tensor,
    assert_quality,
    device_config,
    row_grid,
    ROW_GRIDS,
    MeshRows,
    P150_AND_RING,
    REFERENCE,
    REFERENCE_LAYERS,
    random_patches,
)
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tt.linear import Linear
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.patch_embed import TT_Patch_Embed

# Per-layer linears, under `vision_tower.encoder.blocks.<layer>`; they see every patch.
ENCODER_LINEARS = {"wqkv": "wqkv", "wo": "wo", "mlp_fc0": "mlp.fc0", "mlp_fc1": "mlp.fc1"}
# Projector linears; they see merged tokens, a quarter of the rows.
PROJECTOR_LINEARS = {"projector_fc0": "mm_projector.proj.0", "projector_fc1": "mm_projector.proj.2"}


def check_linear(mesh_device, torch_linear, rows: MeshRows) -> None:
    tt = Linear(
        torch_linear.in_features,
        torch_linear.out_features,
        bias=torch_linear.bias is not None,
        mesh_device=mesh_device,
        compute_kernel_config=compute_kernel_config(mesh_device),
    )
    tt.load_torch_state_dict(torch_linear.state_dict())

    x = random_tensor(rows.n, torch_linear.in_features)
    with torch.no_grad():
        golden = torch_linear(rows.take(x))

    out = tt(rows.shard(x, mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.003)


@P150_AND_RING
@REFERENCE_LAYERS
@ROW_GRIDS
@pytest.mark.parametrize("path", ENCODER_LINEARS.values(), ids=ENCODER_LINEARS.keys())
def test_encoder_linear(mesh_device, sp_axis, num_links, config, reference, size, layer, path):
    torch_linear = reference.vision_tower.encoder.blocks[layer].get_submodule(path)
    ((_, h, w),) = row_grid(size)
    check_linear(mesh_device, torch_linear, MeshRows(h * w, config.max_patches, sp_axis))


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
@pytest.mark.parametrize("path", PROJECTOR_LINEARS.values(), ids=PROJECTOR_LINEARS.keys())
def test_projector_linear(mesh_device, sp_axis, num_links, config, reference, size, path):
    ((_, h, w),) = row_grid(size)
    rows = MeshRows(h * w // config.merge_area, config.max_patches // config.merge_area, sp_axis)
    check_linear(mesh_device, reference.get_submodule(path), rows)


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_patch_embed_linear(mesh_device, sp_axis, num_links, config, reference, size):
    """`patch_embed.proj`, loaded through `TT_Patch_Embed` (conv weight zero-padded to
    `padded_patch_dim`), on zero-padded patch rows, against the reference Conv2d."""
    ckc = compute_kernel_config(mesh_device)
    tt = TT_Patch_Embed(device_config(config, sp_axis), mesh_device=mesh_device, compute_kernel_config=ckc)
    tt.load_torch_state_dict(reference.vision_tower.patch_embed.state_dict())

    pixels, _ = random_patches(config, row_grid(size))
    pixels = pixels.to(torch.bfloat16)
    rows = MeshRows(pixels.shape[0], config.max_patches, sp_axis)
    with torch.no_grad():
        golden = reference.vision_tower.patch_embed.proj(rows.take(pixels)).flatten(1)

    out = tt.proj(rows.shard(host_prep.patch_rows(config, pixels), mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device), relative_rmse=0.004)
