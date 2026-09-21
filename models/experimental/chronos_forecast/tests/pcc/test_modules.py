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
