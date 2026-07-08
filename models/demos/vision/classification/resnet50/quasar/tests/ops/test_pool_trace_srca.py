# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
SrcA trace for the C>=64 + out_col>=2 under-max bug. Minimal failing case: in_w=16 => out_w=3, C=64,
so stick 2 (last output col) is the failing one and its reduce runs at the TAIL of the sim log.

Prints channel-0's true stick-2 window as bf16 hex (matching the sim's `srcA_col0[..]` POOLDBG dump, which
prints src_a >> 16) so the on-device SrcA can be compared directly against the expected window.

RUN:
  SIM=/localdev/mstaletovic/craq-sim/sim_qsr_dbg POOLDBG=1 TT_METAL_SIMULATOR_PARALLEL_CLOCK_THREADS=1 \
    ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_pool_trace_srca.py
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
def test_pool_trace_srca(mesh_device):
    device = mesh_device
    torch.manual_seed(0)
    batch, C, in_h, in_w = 1, 64, 4, 16
    out_w = (in_w - KERNEL[1]) // STRIDE[1] + 1  # 3
    n_out = out_w

    x_nchw = torch.randn((batch, C, in_h, in_w), dtype=torch.bfloat16)
    golden = torch.nn.functional.max_pool2d(x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE)).permute(
        0, 2, 3, 1
    ).reshape(-1, C)[:n_out]

    def bf16hex(t):
        return [f"0x{v:04x}" for v in t.to(torch.bfloat16).view(torch.uint16).tolist()]

    # channel 0's stick-2 window (rows 0..3, input cols 8..15) — what SrcA col0 SHOULD contain
    for s in range(n_out):
        col0 = s * STRIDE[1]
        win = x_nchw[0, 0, 0:4, col0 : col0 + 8]  # (4,8)
        print(
            f"  stick {s}: ch0 window cols[{col0}:{col0+8}] max={win.max().item():.3f} "
            f"golden={golden[s][0].item():.3f}"
        )
        if s == n_out - 1:
            print(f"    ch0 win rows(h0..h3) bf16hex:")
            for h in range(4):
                print(f"      h{h}: {bf16hex(win[h])}")

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
        print(f"  RESULT stick {s}: ch0 got={got[s][0].item():.3f} golden={golden[s][0].item():.3f}")
    assert torch.allclose(got[n_out - 1], golden[n_out - 1], atol=5e-2), "stick 2 mismatch (expected)"
