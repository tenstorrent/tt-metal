# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""conv3d with residual_tensor must match conv3d followed by ttnn.add (bf16, row-major)."""

import pytest
import torch

import ttnn


def _to_dev(t, mesh, layout):
    return ttnn.from_torch(
        t, device=mesh, dtype=ttnn.bfloat16, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "C_in, C_out, C_in_block, C_out_block",
    [
        (128, 128, 128, 128),  # one C_in block: streaming output
        (256, 128, 128, 64),  # two C_in blocks: reducer path, two C_out blocks
    ],
)
def test_conv3d_residual(mesh_device, device_params, C_in, C_out, C_in_block, C_out_block):
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    torch.manual_seed(0)
    N, T, H, W = 1, 4, 10, 12
    x = torch.randn(N, T, H, W, C_in)
    conv = torch.nn.Conv3d(C_in, C_out, kernel_size=3, padding=(0, 1, 1))
    tt_x = _to_dev(x, mesh, ttnn.ROW_MAJOR_LAYOUT)
    tt_w = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=ttnn.from_torch(conv.weight.data, dtype=ttnn.bfloat16),
        groups=1,
        C_in_block=C_in_block,
        alignment=32,
        device=mesh,
    )
    tt_b = _to_dev(conv.bias.data.reshape(1, -1), mesh, ttnn.TILE_LAYOUT)
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1,
        W_out_block=4,
        H_out_block=2,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
    )
    kcfg = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def run(residual=None):
        return ttnn.experimental.conv3d(
            input_tensor=tt_x,
            weight_tensor=tt_w,
            bias_tensor=tt_b,
            device=mesh,
            dtype=ttnn.bfloat16,
            output_channels=C_out,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            padding_mode="zeros",
            groups=1,
            config=cfg,
            compute_kernel_config=kcfg,
            residual_tensor=residual,
        )

    plain = run()
    res = torch.randn(tuple(plain.shape))
    tt_res = _to_dev(res, mesh, ttnn.ROW_MAJOR_LAYOUT)
    ref = _host(ttnn.add(tt_res, plain))
    fused = _host(run(tt_res))
    diff = (fused - ref).abs()
    print(
        f"residual C_in={C_in} C_in_block={C_in_block}: max_abs={diff.max().item():.3e} exact={diff.eq(0).float().mean():.6f}"
    )
    assert diff.eq(0).float().mean() > 0.999 and diff.max() <= ref.abs().max() * 2**-7
