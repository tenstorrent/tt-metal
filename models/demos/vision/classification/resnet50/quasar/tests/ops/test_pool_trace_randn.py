# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
randn probe to localize the remaining under-max bug: does it fail on a SINGLE window (out_w=1, no split,
no overlap), or only with overlap/split? All spike/baseline probes pass; randn (many distinct close values)
fails. This isolates the geometry.

kernel=(4,8) stride=(2,4) pad=0, in_h=4, C=32, 1 core. in_w parametrized: 8->out_w=1, 12->2, 20->4.
"""
import pytest
import torch

import ttnn

KERNEL = (4, 8)
STRIDE = (2, 4)
PADDING = (0, 0)
DILATION = (1, 1)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("C", [32, 64, 128], ids=["C32", "C64", "C128"])
@pytest.mark.parametrize("in_w", [8, 20], ids=["in8_ow1", "in20_ow4"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pool_trace_randn(mesh_device, in_w, C):
    device = mesh_device
    torch.manual_seed(0)
    batch, in_h = 1, 4
    out_w = (in_w - KERNEL[1]) // STRIDE[1] + 1
    n_out = out_w

    x_nchw = torch.randn((batch, C, in_h, in_w), dtype=torch.bfloat16)
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
        diff = (got[s] - golden[s]).abs()
        wrong_ch = [c for c in range(C) if diff[c] >= 5e-2]
        print(
            f"  in_w={in_w} C={C} stick {s} (reader{s & 1}): wrong={len(wrong_ch)}/{C} maxerr={diff.max():.3f} "
            f"wrong_ch={wrong_ch[:16]}"
        )

    for s in range(n_out):
        assert torch.allclose(got[s], golden[s], atol=5e-2), f"in_w={in_w} stick {s} mismatch"
