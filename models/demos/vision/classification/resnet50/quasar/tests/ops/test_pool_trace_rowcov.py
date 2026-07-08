# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Isolate the remaining quasar max_pool2d bug: per-channel max where different channels peak at DIFFERENT
window positions. (pos_inc/chan_inc pass because a single position dominates ALL channels; randn fails.)

kernel=(4,8) stride=(2,4) pad=0, in_h=4 in_w=12 => out 1x2 (stick0=reader0, stick1=reader1). C=32.

INPUT (row-coverage probe): x[0, c, h, w] = 10.0 if (c % 4 == h) else 1.0, for input rows h=0..3, all cols.
=> channel group k = {c : c%4==k} has its 10.0 ONLY in input row h=k. Every output window covers rows 0..3,
so the per-channel max is 10.0 for EVERY channel. Expected output: all 10.0.

If the reduce only maxes over a SUBSET of the 4 window rows, the channel groups whose winning row was
skipped come back 1.0. So the wrong channel groups tell you EXACTLY which window rows the reduce missed.

RUN (debug sim + POOLDBG trace):
  SIM=/localdev/mstaletovic/craq-sim/sim_qsr_dbg DPRINT=1 POOLDBG=1 \
    TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=1 \
    ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_pool_trace_rowcov.py
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
def test_pool_trace_rowcov(mesh_device):
    device = mesh_device
    batch, C, in_h, in_w = 1, 32, 4, 12
    out_h = (in_h - KERNEL[0]) // STRIDE[0] + 1  # 1
    out_w = (in_w - KERNEL[1]) // STRIDE[1] + 1  # 2
    n_out = out_h * out_w

    x_nchw = torch.ones((batch, C, in_h, in_w), dtype=torch.bfloat16)  # baseline 1.0
    for c in range(C):
        x_nchw[0, c, c % 4, :] = 10.0  # channel c peaks in input row (c % 4)

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

    for s in range(n_out):
        # per channel-group (c%4) result: 10.0 = that window row covered, 1.0 = missed
        grp = [round(got[s][k].item(), 1) for k in range(4)]  # channels 0,1,2,3 => groups 0,1,2,3
        covered = [k for k in range(4) if abs(got[s][k].item() - 10.0) < 0.1]
        missed = [k for k in range(4) if k not in covered]
        print(f"  stick {s} (reader{s & 1}): group[0..3]={grp}  covered_rows={covered}  MISSED_rows={missed}")

    for s in range(n_out):
        assert torch.allclose(got[s], golden[s], atol=1e-2), (
            f"stick {s}: got[:4]={got[s][:4].tolist()} expected all 10.0"
        )
