# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""rgb_to_yuv on a patchified (1, T, H/p, W/p, 3*p*p) input must match reshape + permute to CHWT, then rgb_to_yuv."""

import pytest
import torch

import ttnn


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "T, Hp, Wp, C_pad",
    [
        (37, 6, 10, 0),  # partial T, Y and UV tiles
        (32, 2, 8, 0),  # all tiles full
        (145, 68, 60, 0),  # LTX 1088x1920 conv_out per chip on 4x8
        (37, 6, 10, 64),  # channels padded to a tile, as conv3d emits them
    ],
)
def test_rgb_to_yuv_patch_input(mesh_device, device_params, T, Hp, Wp, C_pad):
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    p = 4
    C = 3 * p * p
    torch.manual_seed(0)
    # Past [-1, 1] so the clip and the uint8 saturation are exercised too.
    x = torch.randn(1, T, Hp, Wp, C) * 0.8
    coeffs = ttnn.experimental.yuv_bt601_coefficients()
    to_mesh = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    clipped = ttnn.clip(ttnn.from_torch(x, device=mesh, **to_mesh), -1.0, 1.0)
    if C_pad:
        # Large values in the padding channels fail the comparison if the reader ever picks them up.
        x_pad = torch.cat([x, torch.full((1, T, Hp, Wp, C_pad - C), 7.0)], dim=-1)
        host = ttnn.reshape(ttnn.from_torch(x_pad, **to_mesh), ttnn.Shape(x.shape), ttnn.Shape(x_pad.shape))
        padded_in = ttnn.clip(ttnn.to_device(host, mesh), -1.0, 1.0)
        assert tuple(padded_in.padded_shape) == tuple(x_pad.shape)
    else:
        padded_in = clipped

    chwt = ttnn.reshape(clipped, (1, T, Hp, Wp, 3, p, p))
    chwt = ttnn.permute(chwt, (0, 4, 2, 6, 3, 5, 1))
    chwt = ttnn.reshape(chwt, (3, Hp * p, Wp * p, T))
    ref = ttnn.experimental.rgb_to_yuv(chwt, coefficients=coeffs)
    out = ttnn.experimental.rgb_to_yuv(padded_in, coefficients=coeffs, input_patch_size=p)

    for name, a, b in zip("YUV", ref, out):
        assert tuple(a.shape) == tuple(b.shape), name
        assert torch.equal(_host(a), _host(b)), f"{name} plane differs"
