# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Deterministic repro for the REMAINING quasar max_pool2d bug (reduce window-coverage / under-max on
varying data), for sim tracing. Distinct from the odd-stick-zero bug (fixed by the unpack+pack re-inits).

kernel=(4,8) stride=(2,4) pad=0, in_h=4 in_w=20 => out_h=1 out_w=4 => 4 output sticks (cols 0..3).
C=32. Input is POSITIONAL: every channel at spatial (h,w) holds value (h*in_w + w + 1) -- so the max over
an output window is exactly the bottom-right position in that window, which we can predict per stick:
  window for out col ow covers rows [0:4], cols [ow*stride_w : ow*stride_w+8]
  => max position value = (3)*in_w + (ow*4 + 7) + 1 = 3*20 + 4*ow + 8 = 68 + 4*ow
  => expected per stick (all channels): [68, 72, 76, 80] for ow = 0,1,2,3.
A stick that comes back BELOW its expected value = the reduce missed the true-max window position(s).

RUN (debug sim + POOLDBG trace):
  SIM=/localdev/mstaletovic/craq-sim/sim_qsr_dbg DPRINT=1 POOLDBG=1 \
    TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=1 \
    ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_pool_trace_coverage.py
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
def test_pool_trace_coverage(mesh_device):
    device = mesh_device
    batch, C, in_h, in_w = 1, 32, 4, 20
    out_h = (in_h - KERNEL[0]) // STRIDE[0] + 1  # 1
    out_w = (in_w - KERNEL[1]) // STRIDE[1] + 1  # 4
    n_out = out_h * out_w

    # positional input: x[:, :, h, w] = h*in_w + w + 1 (identical across channels)
    pos = (torch.arange(in_h * in_w, dtype=torch.float32) + 1.0).reshape(in_h, in_w)
    x_nchw = pos.reshape(1, 1, in_h, in_w).expand(batch, C, in_h, in_w).to(torch.bfloat16).contiguous()
    golden = torch.nn.functional.max_pool2d(x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE)).permute(
        0, 2, 3, 1
    ).reshape(-1, C)[:n_out]

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
        exp = golden[s][0].item()
        g0 = got[s][0].item()
        ok = bool(torch.allclose(got[s], golden[s], atol=1e-2))
        print(f"  stick {s} (reader{s & 1}): expected~{exp:.0f} got0={g0:.1f} correct={ok} got[:6]={got[s][:6].tolist()}")

    for s in range(n_out):
        assert torch.allclose(got[s], golden[s], atol=1e-2), (
            f"stick {s} (reader{s & 1}): got {got[s][:4].tolist()} expected {golden[s][:4].tolist()}"
        )
