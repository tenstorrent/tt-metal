# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""EVO50 stem pixel_unshuffle alone: [1,3,384,1664] bf16 DRAM -> NHWC [1,192,832,16] RM, height-sharded 64 cores."""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.unit_tests.operations.conv.test_evo50_input_preproc import (
    HS_CORE_GRID,
    IN_C,
    IN_H,
    IN_W,
    S2D_C,
    S2D_CP,
    S2D_H,
    S2D_R,
    S2D_W,
    signpost,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_evo50_pixel_unshuffle_micro(device):
    torch.manual_seed(0)
    x = torch.randn(1, IN_C, IN_H, IN_W, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    n_cores = HS_CORE_GRID[0] * HS_CORE_GRID[1]
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HS_CORE_GRID[0] - 1, HS_CORE_GRID[1] - 1))}
            ),
            (S2D_H * S2D_W // n_cores, S2D_CP),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    def run():
        return ttnn.pixel_unshuffle(
            tt_in,
            S2D_R,
            memory_config=mem,
            channel_order=ttnn.PixelUnshuffleChannelOrder.CHANNEL_MAJOR,
            channels_last=True,
            padded_channels=S2D_CP,
        )

    ttnn.deallocate(run())
    ttnn.synchronize_device(device)
    signpost(header="pu-start")
    outs = [run() for _ in range(3)]
    ttnn.synchronize_device(device)
    signpost(header="pu-end")

    ref = F.pixel_unshuffle(x, S2D_R).permute(0, 2, 3, 1)
    ref = F.pad(ref, (0, S2D_CP - S2D_C))
    got = ttnn.to_torch(outs[0]).reshape(ref.shape)
    assert torch.equal(got, ref), "pixel_unshuffle output mismatch"
