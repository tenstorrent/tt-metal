# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.audiox.reference.oobleck import ResidualUnit, SnakeBeta
from models.experimental.audiox.tt.oobleck import TtResidualUnit, TtSnakeBeta
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.99


def _to_nhwc(x: torch.Tensor) -> torch.Tensor:
    # [B, C, T] -> [B, T, 1, C]
    return x.transpose(1, 2).unsqueeze(2).contiguous()


def _from_nhwc(t: ttnn.Tensor) -> torch.Tensor:
    # [B, T, 1, C] -> [B, C, T]
    return ttnn.to_torch(t).squeeze(2).transpose(1, 2)


@pytest.mark.parametrize("batch, channels, t", [(1, 8, 32), (2, 16, 64)])
def test_tt_snake_beta_pcc(device, batch, channels, t):
    torch.manual_seed(0)

    ref = SnakeBeta(in_features=channels).eval()
    # Init at zero collapses Snake to identity-ish — randomize alpha/beta to
    # exercise the periodic term.
    with torch.no_grad():
        ref.alpha.copy_(torch.randn(channels) * 0.1)
        ref.beta.copy_(torch.randn(channels) * 0.1)

    x = torch.randn(batch, channels, t)
    with torch.no_grad():
        ref_out = ref(x)

    tt_act = TtSnakeBeta(mesh_device=device, state_dict=ref.state_dict())
    tt_x = ttnn.from_torch(_to_nhwc(x), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_act(tt_x)

    assert_with_pcc(ref_out, _from_nhwc(tt_out), pcc=PCC_THRESHOLD)


@pytest.mark.parametrize("batch, channels, t, dilation", [(1, 16, 32, 1), (1, 16, 32, 3)])
def test_tt_residual_unit_pcc(device, batch, channels, t, dilation):
    torch.manual_seed(0)

    ref = ResidualUnit(channels=channels, dilation=dilation).eval()
    x = torch.randn(batch, channels, t)
    with torch.no_grad():
        ref_out = ref(x)

    tt_block = TtResidualUnit(
        mesh_device=device,
        state_dict=ref.state_dict(),
        channels=channels,
        dilation=dilation,
    )
    tt_x = ttnn.from_torch(_to_nhwc(x), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_block(tt_x)

    assert_with_pcc(ref_out, _from_nhwc(tt_out), pcc=PCC_THRESHOLD)
