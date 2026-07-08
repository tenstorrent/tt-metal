# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Sign / max-identity probe for quasar max_pool2d. All coverage probes (positive spike vs baseline) pass;
randn (which has NEGATIVES) fails. This tests the reduce with an ALL-NEGATIVE window.

kernel=(4,8) stride=(2,4) pad=0, in_h=4 in_w=8 => out 1x1 (single window, single reader). C=32 mapped 1:1 to
the 32 window positions: channel c peaks at (row=c//8, col=c%8) with value -1.0; baseline -10.0.
=> per-channel max = -1.0 for every channel (the least-negative value at its unique position). Expected all -1.0.

If the reduce mishandles negatives (wrong -inf identity, sign, or unsigned compare), channels come back wrong
(e.g. -10.0, or 0, or a positive identity leak). Compared against test_pool_trace_poscov (same layout, POSITIVE
values) which passes, a failure here isolates the bug to negative-value handling in the max reduce.

RUN:
  SIM=/localdev/mstaletovic/craq-sim/sim_qsr_dbg DPRINT=1 POOLDBG=1 TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=1 \
    ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_pool_trace_neg.py
"""
import pytest
import torch

import ttnn

KERNEL = (4, 8)
STRIDE = (2, 4)
PADDING = (0, 0)
DILATION = (1, 1)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("baseline,peak", [(-10.0, -1.0), (-1.0, 5.0)], ids=["all_neg", "mixed_sign"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pool_trace_neg(mesh_device, baseline, peak):
    device = mesh_device
    batch, C, in_h, in_w = 1, 32, 4, 8
    n_out = 1

    x_nchw = torch.full((batch, C, in_h, in_w), baseline, dtype=torch.bfloat16)
    for c in range(C):
        x_nchw[0, c, c // 8, c % 8] = peak  # channel c peaks at unique position (c//8, c%8)

    golden = torch.nn.functional.max_pool2d(x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE)).permute(
        0, 2, 3, 1
    ).reshape(-1, C)[:n_out]  # all == peak

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

    wrong = [(c, round(got[0][c].item(), 2)) for c in range(C) if abs(got[0][c].item() - peak) >= 0.05]
    print(f"  baseline={baseline} peak={peak}  expected all {peak}")
    print(f"  WRONG channels (c,got): {wrong}")
    print(f"  got[:32]={[round(v,2) for v in got[0].tolist()]}")

    assert torch.allclose(got[0], golden[0], atol=5e-2), f"wrong channels {wrong}"
