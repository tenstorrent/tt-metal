# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""rms_norm into a column window of a preallocated tensor (tt/fused/common.rms_norm_bf16_cat) vs
concat(rms_norm_bf16(h), rms_norm_bf16(m)): bit-equal for fp32 and bf16 inputs at T = 128, 1024 and 8448."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F


@pytest.mark.parametrize("t", [128, 1024, 8448])
def test_rms_window(device, t):
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    F.probe_rms_bf16(device, ckc)
    if not F.rms_window():
        pytest.skip("build without rms_norm output_tensor")
    g = torch.Generator().manual_seed(t)
    for dt in (ttnn.float32, ttnn.bfloat16):
        xs = [
            ttnn.from_torch(
                torch.randn(1, 1, t, 2304, generator=g) * s, dtype=d, layout=ttnn.TILE_LAYOUT, device=device
            )
            for s, d in ((3.0, ttnn.float32), (0.1, dt))
        ]
        ref = ttnn.concat([F.rms_norm_bf16(x, epsilon=1e-6, compute_kernel_config=ckc) for x in xs], dim=-1)
        new = F.rms_norm_bf16_cat(xs, epsilon=1e-6, compute_kernel_config=ckc)
        a, b = ttnn.to_torch(ref), ttnn.to_torch(new)
        assert a.shape == b.shape and torch.equal(a, b), float((a - b).abs().max())
        if F.rms_fan():  # one norm program, three destinations
            fan = F.rms_norm_bf16_fan(xs[0], 3, epsilon=1e-6, compute_kernel_config=ckc)
            for o in fan:
                ttnn.rms_norm(xs[1], output_tensor=o, output_col_tiles=72, epsilon=1e-6, compute_kernel_config=ckc)
                assert torch.equal(a, ttnn.to_torch(o))
