# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_linear_norm.TTLinearNorm`
vs reference ``LinearNorm``."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.modules import LinearNorm
from models.experimental.kokoro.tt.tt_linear_norm import TTLinearNorm, preprocess_tt_linear_norm


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )


def _check(device, *, in_dim: int, out_dim: int, B: int, T: int, bias: bool, seed: int):
    torch.manual_seed(seed)
    ref = LinearNorm(in_dim, out_dim, bias=bias).eval()
    params = preprocess_tt_linear_norm(ref, device)
    tt_mod = TTLinearNorm(params)

    x = torch.randn(B, T, in_dim)
    with torch.no_grad():
        ref_out = ref(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_mod(x_tt, compute_kernel_config=_compute_cfg(device))
    tt_torch = ttnn.to_torch(tt_out).float()
    while tt_torch.dim() > ref_out.dim():
        tt_torch = tt_torch.squeeze(0)
    ttnn.deallocate(tt_out)
    ttnn.deallocate(x_tt)

    assert tt_torch.shape == ref_out.shape, (tt_torch.shape, ref_out.shape)
    _, pcc = comp_pcc(ref_out, tt_torch, pcc=0.0)
    print(f"TTLinearNorm (in={in_dim}, out={out_dim}, B={B}, T={T}, bias={bias}) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_linear_norm_kokoro_duration_proj_shape(device):
    """Matches Kokoro's ``duration_proj`` = ``LinearNorm(d_hid=512, max_dur=50)``."""
    _check(device, in_dim=512, out_dim=50, B=1, T=64, bias=True, seed=0)


def test_tt_linear_norm_small_bias(device):
    _check(device, in_dim=64, out_dim=32, B=2, T=16, bias=True, seed=1)


def test_tt_linear_norm_no_bias(device):
    _check(device, in_dim=64, out_dim=32, B=2, T=16, bias=False, seed=2)
