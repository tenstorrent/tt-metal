# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_ada_layer_norm.TTAdaLayerNorm` vs reference ``AdaLayerNorm``."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.modules import AdaLayerNorm
from models.experimental.kokoro.tt.tt_ada_layer_norm import TTAdaLayerNorm, preprocess_tt_ada_layer_norm


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )


def test_tt_ada_layer_norm_matches_torch_default_shape(device):
    """Kokoro-like ``d_model=192``, ``style_dim=64``, full time axis."""
    torch.manual_seed(0)
    sty_dim, channels, b, t = 64, 192, 2, 31
    ref = AdaLayerNorm(sty_dim, channels)
    ref.eval()
    params = preprocess_tt_ada_layer_norm(ref, device)
    tt_mod = TTAdaLayerNorm(params)

    x = torch.randn(b, t, channels)
    s = torch.randn(b, sty_dim)
    with torch.no_grad():
        y_ref = ref(x, s)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, compute_kernel_config=_compute_cfg(device))
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    assert y_hat.shape == y_ref.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTAdaLayerNorm PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_ada_layer_norm_matches_torch_small_channels(device):
    """Smaller ``C`` to stress slice / broadcast path."""
    torch.manual_seed(1)
    sty_dim, channels, b, t = 32, 48, 1, 16
    ref = AdaLayerNorm(sty_dim, channels)
    ref.eval()
    params = preprocess_tt_ada_layer_norm(ref, device)
    tt_mod = TTAdaLayerNorm(params)

    x = torch.randn(b, t, channels)
    s = torch.randn(b, sty_dim)
    with torch.no_grad():
        y_ref = ref(x, s)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, compute_kernel_config=_compute_cfg(device))
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTAdaLayerNorm (small C) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
