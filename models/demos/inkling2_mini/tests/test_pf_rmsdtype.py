# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""fused.common.rms_norm_f32 / rms_norm_bf16 (rms_norm with an output dtype, no typecast pass) vs the typecast forms
they replace (bf16 in -> fp32 out; fp32 in -> bf16 out, with and without weight): bit-equal at the prefill compute
configs."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.prefill_model import EPS


@pytest.mark.parametrize("t", [256, 2048])
@pytest.mark.parametrize("l1_acc", [False, True])
def test_pf_rmsdtype(device, t, l1_acc):
    g = torch.Generator().manual_seed(t)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=l1_acc,
    )  # fmt: skip
    x = torch.randn(1, 1, t, 2304, generator=g) * 3
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kw = dict(epsilon=EPS, compute_kernel_config=ckc)
    new = F.rms_norm_f32(x, **kw)
    assert F._RMS_DTYPE, "this build's rms_norm takes no dtype"
    assert new.dtype == ttnn.float32
    ref = ttnn.to_torch(ttnn.rms_norm(ttnn.typecast(x, ttnn.float32), **kw))
    assert torch.equal(ref, ttnn.to_torch(new))


@pytest.mark.parametrize("t", [256, 2048])
@pytest.mark.parametrize("l1_acc", [False, True])
def test_pf_rms_bf16(device, t, l1_acc):
    g = torch.Generator().manual_seed(t + 7)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=l1_acc,
    )  # fmt: skip
    x = ttnn.from_torch(torch.randn(1, 1, t, 2304, generator=g) * 3, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                        device=device)  # fmt: skip
    w = ttnn.from_torch(  # the model's norm weights: bf16 [72, 32] row-major
        torch.randn(72, 32, generator=g), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    assert F.probe_rms_bf16(device, ckc), "this build's rms_norm(dtype=bfloat16) does not round as typecast"
    for extra in ({}, {"weight": w}):
        kw = dict(epsilon=EPS, compute_kernel_config=ckc, **extra)
        new = F.rms_norm_bf16(x, **kw)
        assert F._RMS_BF16 and new.dtype == ttnn.bfloat16
        ref = ttnn.to_torch(ttnn.typecast(ttnn.rms_norm(x, **kw), ttnn.bfloat16))
        assert torch.equal(ref, ttnn.to_torch(new)), extra.keys()
