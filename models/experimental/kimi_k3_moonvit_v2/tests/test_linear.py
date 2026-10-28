# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `Linear` against each `nn.Linear` of the reference, at the row count it sees in the model."""

from __future__ import annotations

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    device_config,
    row_grid,
    ROW_GRIDS,
    P150_AND_RING,
    REFERENCE,
    REFERENCE_LAYERS,
    random_patches,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.tt.linear import Linear
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.patch_embed import TT_Patch_Embed

# Per-layer linears, under `vision_tower.encoder.blocks.<layer>`; they see every patch.
ENCODER_LINEARS = {"wqkv": "wqkv", "wo": "wo", "mlp_fc0": "mlp.fc0", "mlp_fc1": "mlp.fc1"}
# Projector linears; they see merged tokens, a quarter of the rows.
PROJECTOR_LINEARS = {"projector_fc0": "mm_projector.proj.0", "projector_fc1": "mm_projector.proj.2"}


def check_linear(mesh_device, sp_axis, config, torch_linear, n: int, rows: int) -> None:
    tt = Linear(
        torch_linear.in_features,
        torch_linear.out_features,
        bias=torch_linear.bias is not None,
        mesh_device=mesh_device,
        compute_kernel_config=compute_kernel_config(mesh_device),
    )
    tt.load_torch_state_dict(torch_linear.state_dict())

    x = torch.randn(n, torch_linear.in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        golden = torch_linear(x)

    out = tt(sp_shard(x, mesh_device, sp_axis, rows))
    assert_pcc(golden, sp_gather(out, mesh_device, sp_axis, n))


@P150_AND_RING
@REFERENCE_LAYERS
@ROW_GRIDS
@pytest.mark.parametrize("path", ENCODER_LINEARS.values(), ids=ENCODER_LINEARS.keys())
def test_encoder_linear(mesh_device, sp_axis, num_links, config, reference, size, layer, path):
    config = device_config(config, sp_axis)
    grid = row_grid(size, sp_axis)
    torch_linear = reference.vision_tower.encoder.blocks[layer].get_submodule(path)
    _, h, w = grid[0]
    check_linear(mesh_device, sp_axis, config, torch_linear, h * w, config.max_patches)


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
@pytest.mark.parametrize("path", PROJECTOR_LINEARS.values(), ids=PROJECTOR_LINEARS.keys())
def test_projector_linear(mesh_device, sp_axis, num_links, config, reference, size, path):
    config = device_config(config, sp_axis)
    grid = row_grid(size, sp_axis)
    _, h, w = grid[0]
    n, rows = h * w // config.merge_area, config.max_patches // config.merge_area
    check_linear(mesh_device, sp_axis, config, reference.get_submodule(path), n, rows)


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
def test_patch_embed_linear(mesh_device, sp_axis, num_links, config, reference, size):
    """`patch_embed.proj`, loaded through `TT_Patch_Embed` (conv weight zero-padded to
    `padded_patch_dim`), on zero-padded patch rows, against the reference Conv2d."""
    config = device_config(config, sp_axis)
    grid = row_grid(size, sp_axis)
    tt = TT_Patch_Embed(config, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config(mesh_device))
    tt.load_torch_state_dict(reference.vision_tower.patch_embed.state_dict())

    pixels, _ = random_patches(config, grid)
    pixels = pixels.to(torch.bfloat16)
    with torch.no_grad():
        golden = reference.vision_tower.patch_embed.proj(pixels).flatten(1)

    out = tt.proj(sp_shard(host_prep.patch_rows(config, pixels), mesh_device, sp_axis, config.max_patches))
    assert_pcc(golden, sp_gather(out, mesh_device, sp_axis, pixels.shape[0]))
