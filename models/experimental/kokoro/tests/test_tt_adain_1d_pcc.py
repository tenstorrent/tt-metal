# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: ``TTInstanceNorm1d`` / ``TTAdaIN1d`` vs PyTorch (``reference/istftnet.py`` AdaIN1d)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.istftnet import AdaIN1d
from models.experimental.kokoro.tt.tt_adain_1d import (
    TTAdaIN1d,
    preprocess_tt_adain_1d,
    preprocess_tt_instance_norm_1d,
    tt_instance_norm_1d_nlc,
)


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )


def test_tt_instance_norm_1d_matches_torch(device):
    """``tt_instance_norm_1d_nlc`` vs ``nn.InstanceNorm1d`` on ``[B,C,L]``."""
    torch.manual_seed(0)
    b, c, l = 2, 64, 48
    inn = nn.InstanceNorm1d(c, affine=True)
    inn.eval()
    p = preprocess_tt_instance_norm_1d(inn, device)

    x_bcl = torch.randn(b, c, l)
    with torch.no_grad():
        y_ref = inn(x_bcl)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_instance_norm_1d_nlc(x_nlc=x_tt, params=p)
    y_hat = ttnn.to_torch(y_tt).float().transpose(1, 2).contiguous()

    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    assert y_hat.shape == y_ref.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTInstanceNorm1d PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_1d_matches_torch_default(device):
    """Kokoro-like channel count and length."""
    torch.manual_seed(1)
    style_dim, num_features, b, l = 64, 192, 2, 96
    ref = AdaIN1d(style_dim, num_features)
    ref.eval()
    params = preprocess_tt_adain_1d(ref, device)
    tt_mod = TTAdaIN1d(params)

    x_bcl = torch.randn(b, num_features, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, compute_kernel_config=_compute_cfg(device))
    y_hat = ttnn.to_torch(y_tt).float().transpose(1, 2).contiguous()

    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    assert y_hat.shape == y_ref.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTAdaIN1d PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_1d_matches_torch_smaller(device):
    """Smaller ``C`` / odd ``L``."""
    torch.manual_seed(2)
    style_dim, num_features, b, l = 32, 48, 1, 37
    ref = AdaIN1d(style_dim, num_features)
    ref.eval()
    params = preprocess_tt_adain_1d(ref, device)
    tt_mod = TTAdaIN1d(params)

    x_bcl = torch.randn(b, num_features, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, compute_kernel_config=_compute_cfg(device))
    y_hat = ttnn.to_torch(y_tt).float().transpose(1, 2).contiguous()

    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTAdaIN1d (small) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_1d_decoder_norm1_config(device):
    """``Decoder.encode.norm1`` configuration: ``AdaIN1d(style_dim=128, num_features=514)``.

    ``norm1`` in :class:`~models.experimental.kokoro.reference.istftnet.AdainResBlk1d`
    is applied to the *input* channels (``dim_in = 514 = hidden_dim + 2``) inside
    ``Decoder.encode``.  Same configuration also appears in the ``Decoder.decode``
    blocks 0-3 with ``dim_in=1090``.
    """
    torch.manual_seed(3)
    # Decoder.encode.norm1: AdaIN1d(style_dim=128, num_features=dim_in=514)
    style_dim, num_features, b, l = 128, 514, 1, 5
    ref = AdaIN1d(style_dim, num_features)
    ref.eval()
    params = preprocess_tt_adain_1d(ref, device, weights_dtype=ttnn.float32)
    tt_mod = TTAdaIN1d(params)

    x_bcl = torch.randn(b, num_features, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, compute_kernel_config=_compute_cfg(device))
    y_hat = ttnn.to_torch(y_tt).float().transpose(1, 2).contiguous()

    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    assert y_hat.shape == y_ref.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTAdaIN1d (Decoder.encode norm1: style=128, C=514) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_1d_decoder_norm2_config(device):
    """``Decoder.encode.norm2`` configuration: ``AdaIN1d(style_dim=128, num_features=1024)``.

    ``norm2`` in :class:`~models.experimental.kokoro.reference.istftnet.AdainResBlk1d`
    is applied to the *output* channels (``dim_out = 1024``) inside ``Decoder.encode``
    and all ``Decoder.decode`` blocks 0-2.
    """
    torch.manual_seed(4)
    # Decoder.encode.norm2: AdaIN1d(style_dim=128, num_features=dim_out=1024)
    style_dim, num_features, b, l = 128, 1024, 1, 5
    ref = AdaIN1d(style_dim, num_features)
    ref.eval()
    params = preprocess_tt_adain_1d(ref, device, weights_dtype=ttnn.float32)
    tt_mod = TTAdaIN1d(params)

    x_bcl = torch.randn(b, num_features, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, compute_kernel_config=_compute_cfg(device))
    y_hat = ttnn.to_torch(y_tt).float().transpose(1, 2).contiguous()

    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    assert y_hat.shape == y_ref.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTAdaIN1d (Decoder.encode norm2: style=128, C=1024) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
