# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_adain_resblock1.TTAdaINResBlock1`
vs reference :class:`~models.experimental.kokoro.reference.istftnet.AdaINResBlock1`."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.istftnet import AdaINResBlock1
from models.experimental.kokoro.tt.tt_adain_resblock1 import (
    TTAdaINResBlock1,
    preprocess_tt_adain_resblock1,
)


def _pcc_nlc(y_ref_bcl: torch.Tensor, y_tt: ttnn.Tensor) -> float:
    """Compare reference BCT vs TT NLC outputs after a transpose."""
    y_ref_nlc = y_ref_bcl.transpose(1, 2).contiguous()
    y_hat = ttnn.to_torch(y_tt).float()
    while y_hat.dim() > y_ref_nlc.dim():
        y_hat = y_hat.squeeze(0)
    _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
    return pcc


def _run_pcc(device, *, channels: int, kernel_size: int, dilation, style_dim: int, B: int, L: int, seed: int):
    torch.manual_seed(seed)

    ref = AdaINResBlock1(channels=channels, kernel_size=kernel_size, dilation=dilation, style_dim=style_dim).eval()

    # Perturb ``alpha`` away from the default ones so the Snake1D math is actually exercised.
    with torch.no_grad():
        for p in list(ref.alpha1) + list(ref.alpha2):
            p.copy_(0.5 + torch.rand_like(p))

    params = preprocess_tt_adain_resblock1(ref, device)
    tt_blk = TTAdaINResBlock1(device, params)

    x_bcl = torch.randn(B, channels, L)
    s = torch.randn(B, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_blk(x_tt, s_tt)
    pcc = _pcc_nlc(y_ref, y_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    print(
        f"TTAdaINResBlock1 (C={channels}, k={kernel_size}, dilation={dilation}, "
        f"style_dim={style_dim}, B={B}, L={L}) PCC: {pcc:.6f}"
    )
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_resblock1_default_dilation(device):
    """Default Kokoro istftnet config: ``kernel_size=3``, ``dilation=(1, 3, 5)``."""
    # ``C >= 48`` keeps ``conv1d`` on the PCC-matching Wormhole code path.
    _run_pcc(device, channels=64, kernel_size=3, dilation=(1, 3, 5), style_dim=64, B=1, L=64, seed=0)


def test_tt_adain_resblock1_kernel7(device):
    """``kernel_size=7``, ``dilation=(1, 3, 5)`` — one of the three resblock kernel sizes in config."""
    _run_pcc(device, channels=64, kernel_size=7, dilation=(1, 3, 5), style_dim=64, B=1, L=64, seed=1)


def test_tt_adain_resblock1_batch2(device):
    """Two-row batch with default kernel_size."""
    _run_pcc(device, channels=64, kernel_size=3, dilation=(1, 3, 5), style_dim=48, B=2, L=64, seed=2)
