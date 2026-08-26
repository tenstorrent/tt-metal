# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: single-device SwiGLU MLP (layer 0) vs torch reference.

``device`` and ``setup`` come from tests/unit/conftest.py.
"""
import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, get_pcc_threshold

from .conftest import DEVICE_PARAMS

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]


def test_mlp_pcc(device, setup, request):
    args, sd, raw = setup

    gate_w = sd["layers.0.mlp.gate_proj.weight"]
    up_w = sd["layers.0.mlp.up_proj.weight"]
    down_w = sd["layers.0.mlp.down_proj.weight"]

    x = torch.randn(1, 4, 4096, dtype=torch.bfloat16)
    ref = F.linear(
        F.silu(F.linear(x, gate_w.to(torch.bfloat16))) * F.linear(x, up_w.to(torch.bfloat16)),
        down_w.to(torch.bfloat16),
    )

    from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
    from models.demos.blackhole.qwen36.utils.substate import substate

    mlp_state = substate(sd, "layers.0.mlp")
    mlp = Qwen36MLP(device, mlp_state)
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(mlp.forward(x_t))

    pcc = compute_pcc(ref, out)
    logger.info(f"MLP PCC: {pcc:.6f}")
    logger.info(f"Ref range: [{ref.min():.4f}, {ref.max():.4f}]  TTNN range: [{out.min():.4f}, {out.max():.4f}]")
    # MLP uses bf8b weights so a relaxed threshold.
    assert pcc > get_pcc_threshold(request), f"MLP PCC too low: {pcc}"
