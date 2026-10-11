# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from ...utils.check import assert_quality

C_IN = 32
C_OUT = 32
KERNEL = (3, 3, 3)
PADDING = (1, 1, 1)
INPUT_THW = (6, 8, 8)


def _run_conv3d(device, t_blk, h_blk, w_blk):
    torch.manual_seed(0)
    x = torch.randn(1, C_IN, *INPUT_THW)
    conv = torch.nn.Conv3d(C_IN, C_OUT, KERNEL, padding=PADDING)

    config = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t_blk,
        H_out_block=h_blk,
        W_out_block=w_blk,
        C_out_block=C_OUT,
        C_in_block=C_IN,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )
    tt_x = ttnn.from_torch(x.permute(0, 2, 3, 4, 1), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_w = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=ttnn.from_torch(conv.weight.data, dtype=ttnn.bfloat16),
        groups=1,
        C_in_block=C_IN,
        alignment=32,
        device=device,
    )
    tt_b = ttnn.from_torch(conv.bias.data.reshape(1, -1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = ttnn.experimental.conv3d(
        input_tensor=tt_x,
        weight_tensor=tt_w,
        bias_tensor=tt_b,
        device=device,
        dtype=ttnn.bfloat16,
        output_channels=C_OUT,
        kernel_size=KERNEL,
        stride=(1, 1, 1),
        padding=PADDING,
        padding_mode="zeros",
        groups=1,
        config=config,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
    with torch.no_grad():
        ref = conv(x).permute(0, 2, 3, 4, 1)
    assert_quality(ref, out.reshape(ref.shape), pcc=0.999)


@pytest.mark.parametrize("mesh_device", [(4, 8)], ids=["4x8"], indirect=["mesh_device"])
def test_vol2col_ring_blockings(mesh_device, expect_error):
    device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))

    # Unaligned with more than 64 patches per block (5*4*4 = 80): refused on the host.
    with expect_error(RuntimeError, "vol2col CB ring"):
        _run_conv3d(device, 5, 4, 4)

    # Unaligned within 64 (48), and aligned above 64 (192), still run.
    _run_conv3d(device, 3, 4, 4)
    _run_conv3d(device, 3, 8, 8)
