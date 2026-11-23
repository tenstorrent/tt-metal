# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""fused.common.rms_norm_f32 (rms_norm of a bf16 input with an fp32 result, no typecast pass) vs the typecast form it
replaces in PfAttnRes: bit-equal at the prefill compute configs."""

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
