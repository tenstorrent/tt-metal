# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark: in-kernel mask SYNTHESIS vs DRAM-mask PARTIAL READ for ttnn.group_norm
# across the SDXL UNet block-sharded shapes.
#
# A/B design (same caller, same shape, same compute config):
#   - Baseline ("dram"): caller passes a bf16 mask tensor -> writer kernel uses
#     MASK_PARTIAL_READ (gamma-style two-strip NOC read).
#   - Treatment ("synth"): caller passes no mask -> host skips build,
#     writer kernel takes the MASK_SYNTHESIZE path (direct L1 stores from
#     start_stride recurrence; zero DRAM allocation, zero NOC reads for the mask).
#
# Both paths produce numerically identical output (verified by the unit-test
# matrix). The device kernel duration difference is the per-call cost of
# fetching the mask from DRAM versus synthesizing it inline.

import pytest
import torch

import ttnn

# SDXL UNet block-sharded shapes the production model actually hits.
# (N, C, H, W, num_groups, core_grid_y, core_grid_x)
# Always uses num_groups=32 per SDXL conventions.
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
# Dedup while preserving order
SDXL_SHAPES = list(dict.fromkeys(_RAW_SHAPES))


def _run_one(device, shape, with_mask: bool, num_iters: int = 8):
    """Run ttnn.group_norm `num_iters` times with the given shape.

    with_mask=True  -> caller passes bf16 input_mask -> MASK_PARTIAL_READ path
    with_mask=False -> caller passes no input_mask  -> MASK_SYNTHESIZE path
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

    input_mask = None
    if with_mask:
        # bf16 mask exercises the partial-read path (BFP8 would fall back to legacy)
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
@pytest.mark.parametrize("shape", SDXL_SHAPES, ids=lambda s: f"C{s[1]}_H{s[2]}_W{s[3]}")
@pytest.mark.parametrize("mode", ["dram", "synth"])
def test_bench_sdxl_group_norm_mask(device, shape, mode):
    """One pytest case per (shape, mode). The tracy device profiler picks the
    GroupNormDeviceOperation kernel duration out of the resulting CSV."""
    _run_one(device, shape, with_mask=(mode == "dram"))
