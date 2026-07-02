# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark: NEGATIVE mask synthesis vs DRAM-passed bf16 negative mask for
# ttnn.group_norm across SDXL UNet block-sharded shapes.
#
# A/B design (same caller, same shape, same compute config, non-Welford):
#   - Baseline ("dram"): caller passes bf16 negative_mask tensor -> writer
#     kernel uses NEGATIVE_MASK_PARTIAL_READ.
#   - Treatment ("synth"): caller passes synthesize_negative_mask=True ->
#     writer kernel takes the NEGATIVE_MASK_SYNTHESIZE path.
#
# Positive mask is synthesized in both variants (no input_mask passed) so
# only the negative-mask code path differs between rows.
#
# Both paths produce numerically identical output (same {1.0, 0.0} recurrence).
# The device kernel duration difference is the per-call cost of fetching the
# negative mask from DRAM versus synthesizing it inline.

import pytest
import torch

import ttnn

# SDXL UNet block-sharded shapes the production model actually hits.
# Same set as bench_group_norm_mask_synth.py — SDXL is the only production
# consumer of negative_mask today (models/demos/stable_diffusion_xl_base).
# (N, C, H, W, num_groups, core_grid_y, core_grid_x)
_RAW_SHAPES = [
    # 1024x1024 resolution — UNet
    (1, 320, 128, 128, 32, 8, 8),
    (1, 320, 64, 64, 32, 8, 8),
    (1, 640, 64, 64, 32, 8, 8),
    (1, 640, 32, 32, 32, 8, 8),
    (1, 960, 64, 64, 32, 8, 8),
    (1, 1280, 64, 64, 32, 8, 8),
    (1, 1280, 32, 32, 32, 8, 8),
    (1, 1920, 64, 64, 32, 8, 8),
    (1, 1920, 32, 32, 32, 8, 8),
    (1, 2560, 32, 32, 32, 8, 8),
    # 512x512 resolution — UNet
    (1, 640, 16, 16, 32, 8, 8),
    (1, 1280, 16, 16, 32, 8, 8),
    (1, 1920, 16, 16, 32, 8, 8),
    (1, 2560, 16, 16, 32, 8, 8),
    (1, 960, 32, 32, 32, 8, 8),
    (1, 320, 32, 32, 32, 8, 8),
]
SDXL_SHAPES = list(dict.fromkeys(_RAW_SHAPES))


def _run_one(device, shape, with_neg_mask_tensor: bool, num_iters: int = 8):
    """Run ttnn.group_norm `num_iters` times with the given shape.

    with_neg_mask_tensor=True  -> pass bf16 negative_mask   -> NEGATIVE_MASK_PARTIAL_READ
    with_neg_mask_tensor=False -> synthesize_negative_mask  -> NEGATIVE_MASK_SYNTHESIZE
    """
    N, C, H, W, num_groups, gy, gx = shape
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

    neg_mask = None
    synth_neg = False
    if with_neg_mask_tensor:
        neg_mask = ttnn.create_group_norm_input_negative_mask(C, num_groups, core_grid.x, ttnn.DataType.BFLOAT16)
        neg_mask = ttnn.to_device(neg_mask, device)
    else:
        synth_neg = True

    for _ in range(num_iters):
        out = ttnn.group_norm(
            tt_in,
            num_groups=num_groups,
            negative_mask=neg_mask,
            synthesize_negative_mask=synth_neg,
            memory_config=tt_in.memory_config(),
            core_grid=core_grid,
            inplace=False,
            use_welford=False,
        )
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

    ttnn.deallocate(tt_in)
    if neg_mask is not None:
        ttnn.deallocate(neg_mask)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("shape", SDXL_SHAPES, ids=lambda s: f"C{s[1]}_H{s[2]}_W{s[3]}")
@pytest.mark.parametrize("mode", ["dram", "synth"])
def test_bench_sdxl_group_norm_neg_mask(device, shape, mode):
    """One pytest case per (shape, mode). The tracy device profiler picks the
    GroupNormDeviceOperation kernel duration out of the resulting CSV."""
    _run_one(device, shape, with_neg_mask_tensor=(mode == "dram"))
