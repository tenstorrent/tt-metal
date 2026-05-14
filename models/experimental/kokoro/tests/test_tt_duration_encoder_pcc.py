# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_duration_encoder.TTDurationEncoder` vs reference ``DurationEncoder``."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.modules import DurationEncoder
from models.experimental.kokoro.tt.tt_duration_encoder import TTDurationEncoder, preprocess_tt_duration_encoder


def _compute_cfg(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )


def test_tt_duration_encoder_matches_torch_full_length(device):
    """Unpadded sequence: all tokens valid."""
    torch.manual_seed(0)
    sty_dim, d_model, nlayers, b, t = 64, 192, 2, 2, 32
    ref = DurationEncoder(sty_dim, d_model, nlayers)
    ref.eval()
    params = preprocess_tt_duration_encoder(ref, device)
    tt_mod = TTDurationEncoder(params)

    x_bct = torch.randn(b, d_model, t)
    style = torch.randn(b, sty_dim)
    text_lengths = torch.tensor([t] * b, dtype=torch.long)
    text_mask = torch.zeros((b, t), dtype=torch.bool)

    with torch.no_grad():
        y_ref = ref(x_bct, style, text_lengths, text_mask)

    keep = (~text_mask).to(torch.float32).unsqueeze(-1)
    x_tt = ttnn.from_torch(x_bct, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(style, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    keep_tt = ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = tt_mod(
        x_tt,
        s_tt,
        sequence_lengths=text_lengths.tolist(),
        keep_mask_btl=keep_tt,
        compute_kernel_config=_compute_cfg(device),
    )
    y_hat = ttnn.to_torch(y_tt).float()

    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)
    ttnn.deallocate(keep_tt)

    assert y_hat.shape == y_ref.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTDurationEncoder PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_duration_encoder_matches_torch_variable_length(device):
    """Padded batch: mask zeros past ``text_lengths``."""
    torch.manual_seed(2)
    sty_dim, d_model, nlayers, b, t = 48, 128, 3, 2, 48
    ref = DurationEncoder(sty_dim, d_model, nlayers)
    ref.eval()
    params = preprocess_tt_duration_encoder(ref, device)
    tt_mod = TTDurationEncoder(params)

    lens = [28, 35]
    x_bct = torch.randn(b, d_model, t)
    style = torch.randn(b, sty_dim)
    text_lengths = torch.tensor(lens, dtype=torch.long)
    text_mask = torch.zeros((b, t), dtype=torch.bool)
    for bi, le in enumerate(lens):
        text_mask[bi, le:] = True

    with torch.no_grad():
        y_ref = ref(x_bct, style, text_lengths, text_mask)

    keep = (~text_mask).to(torch.float32).unsqueeze(-1)
    x_tt = ttnn.from_torch(x_bct, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(style, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    keep_tt = ttnn.from_torch(keep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = tt_mod(
        x_tt,
        s_tt,
        sequence_lengths=lens,
        keep_mask_btl=keep_tt,
        compute_kernel_config=_compute_cfg(device),
    )
    y_hat = ttnn.to_torch(y_tt).float()

    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)
    ttnn.deallocate(keep_tt)

    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTDurationEncoder (masked) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
