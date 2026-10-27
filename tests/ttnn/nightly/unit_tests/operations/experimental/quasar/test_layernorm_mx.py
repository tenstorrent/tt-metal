# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.layer_norm / rms_norm over MX (OCP microscaling) inputs on Quasar (issue #62742).

The interleaved layernorm unpacks the MX input (and residual) to Float16_b, keeps every intermediate ring in
Float16_b / Float32, and packs the output in the input's format; gamma / beta stay bf16.
"""

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.mx_test_utils import (
    check,
    mx_round_trip,
    pcc_floor,
    skip_unless_quasar,
    to_device,
)

SHAPE = (1, 1, 64, 96)
EPS = 1e-5


@pytest.fixture(autouse=True)
def _quasar_only(device):
    skip_unless_quasar(device)


def _rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("residual", [False, True], ids=["plain", "residual"])
def test_layer_norm_mx(device, dtype, residual):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    r = torch.randn(SHAPE)
    gamma = torch.randn(SHAPE[-1]).bfloat16()
    beta = torch.randn(SHAPE[-1]).bfloat16()
    kwargs = {"residual_input_tensor": to_device(device, r, dtype)} if residual else {}
    out = ttnn.layer_norm(
        to_device(device, x, dtype),
        epsilon=EPS,
        weight=to_device(device, gamma.reshape(1, 1, 1, -1), ttnn.bfloat16),
        bias=to_device(device, beta.reshape(1, 1, 1, -1), ttnn.bfloat16),
        **kwargs,
    )
    assert out.dtype == dtype
    xin = mx_round_trip(x, dtype) + (mx_round_trip(r, dtype) if residual else 0)
    golden = torch.nn.functional.layer_norm(xin, (SHAPE[-1],), gamma.float(), beta.float(), EPS)
    check(mx_round_trip(golden, dtype), ttnn.to_torch(out), pcc_floor(dtype))


@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
def test_rms_norm_mx(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(SHAPE)
    gamma = torch.randn(SHAPE[-1]).bfloat16()
    out = ttnn.rms_norm(
        to_device(device, x, dtype), epsilon=EPS, weight=to_device(device, gamma.reshape(1, 1, 1, -1), ttnn.bfloat16)
    )
    assert out.dtype == dtype
    golden = _rms_norm(mx_round_trip(x, dtype), gamma.float(), EPS)
    check(mx_round_trip(golden, dtype), ttnn.to_torch(out), pcc_floor(dtype))
