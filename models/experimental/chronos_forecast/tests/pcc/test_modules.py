# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""PCC tests comparing TTNN Chronos against the PyTorch reference (single-chip)."""

from __future__ import annotations

import pytest
import torch

from models.experimental.chronos_forecast.reference.chronos2.layers import ResidualBlock as RefRB
from models.experimental.chronos_forecast.tests.golden_helpers import log_golden
from models.experimental.chronos_forecast.tt.residual_block import (
    TtResidualBlock,
    TtResidualBlockWeights,
)


@pytest.mark.parametrize(
    "shape",
    [pytest.param((2, 4, 48), id="tiny_2x4x48"), pytest.param((2, 2, 48), id="real_2x2x48")],
)
def test_residual_block_pcc(request, shape):
    """TT ResidualBlock (48d patched input) vs reference oracle.

    tiny covers the golden stub geometry (tests/test_residual.py); real covers
    the T=32 patched context (B=2, P=2, d_model=6). Single-chip only.
    """
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    mesh_device = request.getfixturevalue("mesh_device")
    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    torch.manual_seed(0)
    block = RefRB(in_dim=48, h_dim=8, out_dim=6, act_fn_name="relu", dropout_p=0.0).eval()
    weights = TtResidualBlockWeights.from_torch_block(block)
    tt = TtResidualBlock(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(*shape)
    expected = block(x).float()
    got = tt.forward(x)
    assert got.shape == (*shape[:2], 6)
    log_golden(f"tt_residual/device_{shape[1]}p", got)
    assert_with_pcc(expected, got, pcc=0.99)


def test_time_attention_pcc(request):
    """TT TimeSelfAttention (tiny golden dims) vs reference oracle. Single-chip only."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.layers import (
        TimeSelfAttention as RefTSA,
    )
    from models.experimental.chronos_forecast.tests.golden_helpers import tiny_config
    from models.experimental.chronos_forecast.tt.time_attention import (
        TtTimeAttention,
        TtTimeAttentionWeights,
        build_rope_cache,
    )

    mesh_device = request.getfixturevalue("mesh_device")
    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    cfg = tiny_config()
    torch.manual_seed(0)
    layer = RefTSA(cfg).eval()
    weights = TtTimeAttentionWeights.from_torch_layer(layer)
    tt = TtTimeAttention(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(2, 8, cfg.d_model)
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    expected = layer(
        x,
        attention_mask=torch.zeros(2, cfg.num_heads, 8, 8),
        position_ids=position_ids,
    ).hidden_states

    cos, sin = build_rope_cache(position_ids, weights.inv_freq)
    got = tt.forward(x, cos, sin, torch.zeros(1, 1, 8, 8))
    assert got.shape == x.shape
    log_golden("tt_time_attn/device_8", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)


def test_group_attention_pcc(request):
    """TT GroupSelfAttention (tiny golden dims) vs reference oracle. Single-chip only."""
    pytest.importorskip("ttnn")
    from tests.ttnn.utils_for_testing import assert_with_pcc

    from models.experimental.chronos_forecast.reference.chronos2.layers import (
        GroupSelfAttention as RefGSA,
    )
    from models.experimental.chronos_forecast.tests.golden_helpers import tiny_config
    from models.experimental.chronos_forecast.tt.group_attention import (
        TtGroupAttention,
        TtGroupAttentionWeights,
    )

    mesh_device = request.getfixturevalue("mesh_device")
    if mesh_device.get_num_devices() != 1:
        pytest.skip("single-chip bring-up only (one chip)")

    cfg = tiny_config()
    torch.manual_seed(0)
    layer = RefGSA(cfg).eval()
    weights = TtGroupAttentionWeights.from_torch_layer(layer)
    tt = TtGroupAttention(device=mesh_device, weights=weights)

    torch.manual_seed(1)
    x = torch.randn(2, 8, cfg.d_model)
    expected = layer(x, attention_mask=torch.zeros(8, 1, 2, 2)).hidden_states

    got = tt.forward(x, torch.zeros(8, 1, 2, 2))
    assert got.shape == x.shape
    log_golden("tt_group_attn/device_8", got)
    assert_with_pcc(expected.float(), got, pcc=0.99)
