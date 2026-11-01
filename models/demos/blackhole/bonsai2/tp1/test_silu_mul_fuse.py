# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GDN out-gate x * silu(z) as one op: is a fused SiLU bit-exact with silu + multiply, and NaN-free for
large |z|? tp.py's _silu_mul keeps two ops because a b-activation fusion overflowed to NaN in the real layer.

    pytest models/demos/blackhole/bonsai2/tp1/test_silu_mul_fuse.py -s
"""
import pytest
import torch

import ttnn

W = 6144


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("zscale", [1.0, 30.0, 300.0])
@pytest.mark.parametrize("xdtype", [ttnn.bfloat16, ttnn.float32])
def test_silu_mul_fuse(mesh_device, zscale, xdtype):
    torch.manual_seed(0)
    x_t, z_t = torch.randn(1, 1, W), torch.randn(1, 1, W) * zscale
    z_t[0, 0, :8] = torch.tensor([-1e4, -500.0, -90.0, -88.0, 88.0, 90.0, 500.0, 1e4])

    def dev(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                               memory_config=ttnn.L1_MEMORY_CONFIG)

    x, z = dev(x_t, xdtype), dev(z_t)
    ref = ttnn.to_torch(ttnn.multiply(x, ttnn.silu(z))).float()
    silu = [ttnn.UnaryOpType.SILU]
    for label, y in [("a-act (z first)", ttnn.multiply(z, x, input_tensor_a_activations=silu)),
                     ("b-act (x first)", ttnn.multiply(x, z, input_tensor_b_activations=silu))]:
        y = ttnn.to_torch(y).float()
        print(f"x {xdtype} zscale {zscale:6.1f} {label}: bit-exact {100 * (y == ref).float().mean().item():6.2f}%  "
              f"NaN {torch.isnan(y).sum().item()}  inf {torch.isinf(y).sum().item()}  max|d| {(y - ref).nan_to_num(1e9).abs().max():.3e}")
