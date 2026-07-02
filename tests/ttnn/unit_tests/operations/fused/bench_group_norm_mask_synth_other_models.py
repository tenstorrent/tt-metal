# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark: MASK_SYNTHESIZE vs MASK_PARTIAL_READ on representative shapes
# from every model family that calls ttnn.group_norm.
#
# A/B design (same caller, same shape, same compute config, non-Welford):
#   - Baseline ("dram"): caller passes bf16 input_mask -> MASK_PARTIAL_READ
#   - Treatment ("synth"): caller omits input_mask     -> MASK_SYNTHESIZE
#
# Every mask is bf16 in this bench, INCLUDING shapes for models that
# currently pass BFP8 in production (SD1.x wormhole, unet_3d). The point
# of the bench is to measure the writer-side cost delta of synthesis vs
# partial-read at each shape family — that delta doesn't depend on which
# dtype the model happens to use in prod, and running everyone at bf16
# keeps the A/B controlled.
#
# The mapping from shape to model site is documented in `SHAPES` below.

import pytest
import torch

import ttnn

# Representative shapes across all bf16-eligible and BFP8-in-prod sites
# (all measured at bf16 for controlled A/B).
# (label, N, C, H, W, num_groups, core_grid_y, core_grid_x)
_RAW_SHAPES = [
    # ------------------------------------------------------------------
    # SD1.x VAE decoder (models/demos/vision/generative/stable_diffusion/wormhole/tt/vae)
    # block_out_channels=[512, 512, 256, 128]. Latent spatial 64x64 -> 512x512
    # after 8x upsample; benches capped at shapes that fit an 8x8 shard.
    # Bf16 mask in prod (already synth-safe).
    # ------------------------------------------------------------------
    ("sd_vae_C128_H128", 1, 128, 128, 128, 32, 8, 8),
    ("sd_vae_C256_H64", 1, 256, 64, 64, 32, 8, 8),
    ("sd_vae_C512_H64", 1, 512, 64, 64, 32, 8, 8),
    ("sd_vae_C512_H32", 1, 512, 32, 32, 32, 8, 8),
    # ------------------------------------------------------------------
    # SD35 VAE decoder (models/tt_dit/models/vae/vae_sd35.py)
    # block_out_channels=(128, 256, 512, 512), norm_num_groups=32.
    # Same L1 cap as SD1.x VAE above.
    # ------------------------------------------------------------------
    ("sd35_vae_C128_H128", 1, 128, 128, 128, 32, 8, 8),
    ("sd35_vae_C256_H64", 1, 256, 64, 64, 32, 8, 8),
    ("sd35_vae_C512_H32", 1, 512, 32, 32, 32, 8, 8),
    # ------------------------------------------------------------------
    # OFT (models/experimental/oft/tt/common.py)
    # num_channels=256, num_groups=16. Sharded input, BEV-shape featmaps.
    # Approx spatial per bn8/bn16/bn32 stages.
    # ------------------------------------------------------------------
    ("oft_C256_H96", 1, 256, 96, 96, 16, 8, 8),
    ("oft_C256_H48", 1, 256, 48, 48, 16, 8, 8),
    ("oft_C256_H24", 1, 256, 24, 24, 16, 8, 8),
    # ------------------------------------------------------------------
    # retinanet FPN heads (models/experimental/retinanet/tt/tt_{classification,regression}_head.py)
    # in_channels=256, num_groups=32. FPN levels tile-padded to nearest 32.
    # ------------------------------------------------------------------
    ("retinanet_C256_H128", 1, 256, 128, 128, 32, 8, 8),
    ("retinanet_C256_H64", 1, 256, 64, 64, 32, 8, 8),
    ("retinanet_C256_H32", 1, 256, 32, 32, 32, 8, 8),
    # ------------------------------------------------------------------
    # unet_3d (models/demos/unet_3d/ttnn_impl/group_norm3d.py)
    # BFP8 mask in prod, tested here at bf16 to measure the synth path
    # if/when the model switches. Small spatial per 3D volume slice.
    # ------------------------------------------------------------------
    ("unet3d_C128_H128", 1, 128, 128, 128, 32, 8, 8),
    ("unet3d_C256_H64", 1, 256, 64, 64, 32, 8, 8),
]
SHAPES = _RAW_SHAPES


def _run_one(device, shape_row, with_mask: bool, num_iters: int = 8):
    """Run ttnn.group_norm `num_iters` times with the given shape row."""
    label, N, C, H, W, num_groups, gy, gx = shape_row
    core_grid = ttnn.CoreGrid(y=gy, x=gx)
    torch.manual_seed(0)
    torch_in = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    flat = torch_in.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    sharded_mem = ttnn.create_sharded_memory_config(
        shape=flat.shape,
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    tt_in = ttnn.from_torch(
        flat,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=sharded_mem,
        device=device,
    )

    input_mask = None
    if with_mask:
        input_mask = ttnn.create_group_norm_input_mask(C, num_groups, core_grid.x, ttnn.DataType.BFLOAT16)
        input_mask = ttnn.to_device(input_mask, device)

    for _ in range(num_iters):
        out = ttnn.group_norm(
            tt_in,
            num_groups=num_groups,
            input_mask=input_mask,
            memory_config=tt_in.memory_config(),
            core_grid=core_grid,
            inplace=False,
            use_welford=False,
        )
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    ttnn.deallocate(tt_in)
    if input_mask is not None:
        ttnn.deallocate(input_mask)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("shape_row", SHAPES, ids=lambda s: s[0])
@pytest.mark.parametrize("mode", ["dram", "synth"])
def test_bench_other_models_group_norm_mask(device, shape_row, mode):
    """One pytest case per (shape, mode). Covers every non-SDXL bf16-eligible
    site plus BFP8-in-prod shapes exercised at bf16."""
    _run_one(device, shape_row, with_mask=(mode == "dram"))
