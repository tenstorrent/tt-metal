# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Definitive per-channel window-POSITION coverage probe for quasar max_pool2d.

kernel=(4,8) stride=(2,4) pad=0, in_h=4 in_w=8 => out 1x1 (SINGLE window, cols 0..7, single reader -> isolates
the reduce from the split path). C=32 == the 32 positions of the 4x8 window, so map each channel to a UNIQUE
window position: channel c peaks (10.0) at (row = c // 8, col = c % 8); everything else 1.0.

=> the per-channel max over the window is 10.0 for EVERY channel (each channel's unique position holds the 10).
Expected output: all 10.0. Any channel c that comes back 1.0 means window position (c//8, c%8) was NOT
included in the reduce. So the 1.0 channels enumerate EXACTLY which (row,col) positions the reduce skipped.

RUN (debug sim + POOLDBG trace):
  SIM=/localdev/mstaletovic/craq-sim/sim_qsr_dbg DPRINT=1 POOLDBG=1 \
    TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=1 \
    ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_pool_trace_poscov.py
"""
import pytest
import torch

import ttnn

KERNEL = (4, 8)
STRIDE = (2, 4)
PADDING = (0, 0)
DILATION = (1, 1)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pool_trace_poscov(mesh_device):
    device = mesh_device
    batch, C, in_h, in_w = 1, 32, 4, 8
    out_h = (in_h - KERNEL[0]) // STRIDE[0] + 1  # 1
    out_w = (in_w - KERNEL[1]) // STRIDE[1] + 1  # 1
    n_out = out_h * out_w  # 1

    x_nchw = torch.ones((batch, C, in_h, in_w), dtype=torch.bfloat16)  # baseline 1.0
    for c in range(C):  # channel c peaks at unique position (row=c//8, col=c%8)
        x_nchw[0, c, c // 8, c % 8] = 10.0

    golden = torch.nn.functional.max_pool2d(x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE)).permute(
        0, 2, 3, 1
    ).reshape(-1, C)[:n_out]  # all 10.0

    x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * in_h * in_w, C).contiguous()
    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(1, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, batch * in_h * in_w, C),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(device, mem_config)

    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch,
        input_h=in_h,
        input_w=in_w,
        channels=C,
        kernel_size=list(KERNEL),
        stride=list(STRIDE),
        padding=list(PADDING),
        dilation=list(DILATION),
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(out).float().reshape(-1, C)[:n_out]

    covered = [(c // 8, c % 8) for c in range(C) if abs(got[0][c].item() - 10.0) < 0.1]
    missed = [(c // 8, c % 8) for c in range(C) if abs(got[0][c].item() - 10.0) >= 0.1]
    print(f"  covered positions (row,col): {covered}")
    print(f"  MISSED  positions (row,col): {missed}")
    print(f"  got[:32]={[round(v,1) for v in got[0].tolist()]}")

    assert torch.allclose(got[0], golden[0], atol=1e-2), f"missed positions {missed}"
