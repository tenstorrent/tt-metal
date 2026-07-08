# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
MINIMAL deterministic repro for the two-reader (split) odd-stick-zero bug, for sim tracing / gdb.

Geometry: kernel=(4,8) stride=(2,4) pad=0, in_h=4 in_w=12 => out_h=1 out_w=2 => EXACTLY 2 output sticks:
  stick 0 = output col 0 = reader0 (even column)
  stick 1 = output col 1 = reader1 (odd column)
1 core, C=32 (one tile width) to keep the trace tiny. Deterministic input channel c -> value (c+1), so
BOTH output sticks must equal [1, 2, ..., 32]. The bug: stick 1 (reader1) comes back all zeros.

This is the smallest configuration that exercises the split reader (one even + one odd stick), so a sim
trace shows exactly one good stick and one broken stick side by side.

RUN (debug sim + POOLDBG + DPRINT):
  cd /localdev/mstaletovic/tt-metal
  SIM=/localdev/mstaletovic/craq-sim/sim_qsr_dbg DPRINT=1 \
    TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=1 POOLDBG=1 \
    ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_pool_trace_repro.py
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
def test_pool_trace_repro(mesh_device):
    device = mesh_device
    batch, C, in_h, in_w = 1, 32, 4, 12
    out_h = (in_h - KERNEL[0]) // STRIDE[0] + 1  # 1
    out_w = (in_w - KERNEL[1]) // STRIDE[1] + 1  # 2
    n_out = out_h * out_w  # 2

    x_nchw = torch.zeros((batch, C, in_h, in_w), dtype=torch.bfloat16)
    for c in range(C):
        x_nchw[0, c, :, :] = float(c + 1)
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

    expected = torch.arange(1, C + 1, dtype=torch.float32)
    for s in range(n_out):
        reader = s & 1
        nz = int(torch.count_nonzero(got[s]))
        ok = bool(torch.allclose(got[s], expected, atol=1e-2))
        print(f"  stick {s} (reader{reader}): nonzero={nz}/{C} correct={ok} got[:8]={got[s][:8].tolist()}")

    for s in range(n_out):
        assert torch.allclose(got[s], expected, atol=1e-2), f"stick {s} (reader{s & 1}) wrong: {got[s][:8].tolist()}"
