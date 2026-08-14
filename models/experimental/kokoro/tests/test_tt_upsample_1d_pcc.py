# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_upsample_1d.TTUpSample1d` vs nearest ``F.interpolate``."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.tt.tt_upsample_1d import TTUpSample1d


class RefUpSample1d(nn.Module):
    """Mirrors ``reference/istftnet.py`` ``UpSample1d``."""

    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "none":
            return x
        return F.interpolate(x, scale_factor=2, mode="nearest")


def test_tt_upsample_1d_none_matches_reference(device):
    torch.manual_seed(0)
    b, c, l = 2, 64, 48
    ref = RefUpSample1d("none")
    tt_up = TTUpSample1d("none")
    x_bcl = torch.randn(b, c, l)
    x_nlc = x_bcl.transpose(1, 2).contiguous()
    with torch.no_grad():
        y_ref = ref(x_bcl)
    y_ref_nlc = y_ref.transpose(1, 2).contiguous()

    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_up(x_tt)
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
    print(f"TTUpSample1d (none) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_upsample_1d_scale2_matches_reference(device):
    torch.manual_seed(1)
    b, c, l = 1, 32, 31
    ref = RefUpSample1d("nearest")
    tt_up = TTUpSample1d("nearest")
    x_bcl = torch.randn(b, c, l)
    x_nlc = x_bcl.transpose(1, 2).contiguous()
    with torch.no_grad():
        y_ref = ref(x_bcl)
    y_ref_nlc = y_ref.transpose(1, 2).contiguous()

    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_up(x_tt)
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    assert y_hat.shape == y_ref_nlc.shape
    _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
    print(f"TTUpSample1d (2x) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
