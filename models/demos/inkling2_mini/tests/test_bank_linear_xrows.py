# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""BankLinear with x_rows (fused/kernels/x_rows.hpp: only the first rows of each activation tile are read, the rest
filled from L1 zeros) vs whole-tile reads: bit-equal output when the activation's other rows are zero, as for the
attention window input. Random BFP8 weight with w_all's K and kb, a few column chunks."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.bank_linear import BankLinear


@pytest.mark.parametrize("x_rows", [9, 16])
def test_bank_linear_x_rows(device, x_rows):
    torch.manual_seed(0)
    k, n = 2304, 7 * 32 * 6
    w = torch.randn(k, n) * 0.02
    x = torch.zeros(1, 1, 32, k)
    x[..., :x_rows, :] = torch.randn(1, 1, x_rows, k)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )  # fmt: skip
    bl = BankLinear(device, w, ttnn.bfloat8_b, n, kb=2, cores_per_bank=2, compute_kernel_config=ckc,
                    out_dtype=ttnn.bfloat16)  # fmt: skip
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ref = ttnn.to_torch(bl(xt))
    bl.x_rows = x_rows
    out = ttnn.to_torch(bl(xt))
    assert torch.equal(out, ref)
    assert out[..., :x_rows, :].abs().sum() > 0
