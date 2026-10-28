# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""R1-4: the ported ``ops.add_rms_norm`` / ``ops.rms_norm`` against torch on the same bf16 inputs, at the shapes the
model issues (paper s_pad 128, the read-out cut's 32 rows), with a positive control (v1.13)."""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.tafsut_forecast import ops
from models.experimental.tafsut_forecast.reference import dataflow as df

D = df.D


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + df.NORM_EPS)


def worst(got, ref):
    """Worst error over each row, relative to that row's largest value (one bf16 rounding is 2^-8)."""
    return float(((got - ref).abs().amax(-1) / ref.abs().amax(-1)).max())


@pytest.mark.parametrize("B, S", [(1024, 128), (512, 256), (1024, 32)])
def test_add_rms_norm(device, B, S):
    g = torch.Generator().manual_seed(B + S)
    a = (torch.randn(B, S, D, generator=g) * 4).bfloat16()
    b = torch.randn(B, S, D, generator=g).bfloat16()
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y, n = ops.add_rms_norm(up(a), up(b), epsilon=df.NORM_EPS)
    y, n = ttnn.to_torch(y).float(), ttnn.to_torch(n).float()
    n1 = ttnn.to_torch(ops.rms_norm(up(a), epsilon=df.NORM_EPS)).float()
    y_ref = a.float() + b.float()
    errs = dict(y=worst(y, y_ref), n=worst(n, rms(y.float())), n_vs_fp32=worst(n, rms(y_ref)), n1=worst(n1, rms(a.float())))
    print(f"\n[add_rms_norm] B {B} S {S}: {errs}")
    assert errs["y"] <= 2**-8 and errs["n"] <= 2**-7 and errs["n1"] <= 2**-7, errs
    # positive control: a norm that drops eps scaling or one row's sum would be off by far more than a rounding
    assert worst(n * 1.02, rms(y.float())) > 2**-7
