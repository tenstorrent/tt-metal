# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
THOROUGH validation of the quasar max_pool2d 4x8 full-tile window (kernel=(4,8), stride=(2,4), pad=0).
Covers the axes that exercised the fixed bugs: output width (out_w 1..7, incl the col>=2 trigger), output
height (out_h 1..7), channel-tile count (C=32/64/128), input value distribution (chan_inc / randn / pos_inc),
and core count (single + multi). Each case is checked against a torch max_pool2d golden per stick (exact,
since max-pool is value-selection) plus PCC.

RUN:  ./qsr_sim_run models/demos/vision/classification/resnet50/quasar/tests/ops/test_pool_4x8_thorough.py
"""
import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

KERNEL = (4, 8)
STRIDE = (2, 4)
PADDING = (0, 0)
DILATION = (1, 1)

# (in_h, in_w, C, num_cores) — kernel(4,8)/stride(2,4): out_h=(in_h-4)//2+1, out_w=(in_w-8)//4+1.
CASES = [
    # single core, width sweep (out_w = 1,2,3,4,5), C sweep — the out_w>=3 x C>=64 bug trigger
    (4, 8, 32, 1), (4, 8, 64, 1), (4, 8, 128, 1),      # out 1x1
    (4, 12, 32, 1), (4, 12, 64, 1), (4, 12, 128, 1),   # out 1x2
    (4, 16, 32, 1), (4, 16, 64, 1), (4, 16, 128, 1),   # out 1x3
    (4, 20, 64, 1), (4, 20, 128, 1),                    # out 1x4
    (4, 24, 64, 1),                                     # out 1x5
    # height sweep (out_h = 3, 7)
    (8, 16, 64, 1), (8, 20, 64, 1), (8, 20, 128, 1),   # out 3x3, 3x4
    (16, 20, 64, 1),                                    # out 7x4
    (32, 32, 64, 1), (32, 32, 128, 1),                  # out 15x7 (big)
    # multi-core (num_cores divides in_h*in_w/32)
    (8, 16, 64, 2), (8, 16, 128, 2),                    # 4 tiles / 2 cores
    (16, 16, 64, 4),                                    # 8 tiles / 4 cores
    (32, 32, 64, 8),                                    # 32 tiles / 8 cores
]
MODES = ["chan_inc", "randn", "pos_inc"]


def _gen(mode, batch, C, in_h, in_w):
    if mode == "chan_inc":
        x = torch.zeros((batch, C, in_h, in_w), dtype=torch.bfloat16)
        for c in range(C):
            x[:, c, :, :] = float(c + 1)
        return x
    if mode == "randn":
        torch.manual_seed(0)
        return torch.randn((batch, C, in_h, in_w), dtype=torch.bfloat16)
    # pos_inc
    pos = (torch.arange(in_h * in_w, dtype=torch.float32) + 1.0).reshape(in_h, in_w)
    return pos.reshape(1, 1, in_h, in_w).expand(batch, C, in_h, in_w).to(torch.bfloat16).contiguous()


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "in_h,in_w,C,num_cores", CASES, ids=[f"in{h}x{w}_C{c}_k{k}" for (h, w, c, k) in CASES]
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_pool_4x8_thorough(mesh_device, in_h, in_w, C, num_cores, mode):
    device = mesh_device
    batch = 1
    out_h = (in_h - KERNEL[0]) // STRIDE[0] + 1
    out_w = (in_w - KERNEL[1]) // STRIDE[1] + 1
    n_out = out_h * out_w

    tensor_height = in_h * in_w
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    if num_cores > max_cores:
        pytest.skip(f"need {num_cores} cores, device has {max_cores}")
    if (tensor_height // 32) % num_cores != 0:
        pytest.skip(f"height_tiles {tensor_height // 32} not divisible by {num_cores}")

    x_nchw = _gen(mode, batch, C, in_h, in_w)
    golden = torch.nn.functional.max_pool2d(
        x_nchw.float(), kernel_size=list(KERNEL), stride=list(STRIDE), padding=list(PADDING)
    ).permute(0, 2, 3, 1).reshape(-1, C)[:n_out]

    x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch * in_h * in_w, C).contiguous()
    shard_h = (tensor_height // num_cores)
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, C),
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

    nbad = sum(1 for s in range(n_out) if not torch.allclose(got[s], golden[s], atol=2e-2, rtol=2e-2))
    maxerr = (got - golden).abs().max().item()
    print(
        f"  [{mode} {in_h}x{in_w} C{C} k{num_cores}] out={n_out} (h{out_h}xw{out_w}) bad={nbad}/{n_out} maxerr={maxerr:.4f}"
    )
    assert nbad == 0, f"{nbad}/{n_out} sticks wrong (maxerr={maxerr:.4f})"
    assert_with_pcc(golden, got, pcc=0.999)
