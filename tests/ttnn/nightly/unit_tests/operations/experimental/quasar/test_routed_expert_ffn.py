# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Functional test for `ttnn.experimental.quasar.routed_expert_ffn`: one routed expert's FFN without an activation,
y = ((x @ w_gate) * (x @ w_up)) @ w_down, on a single node with one reader, one compute and one writer thread.
All tensors are bfloat16 TILE-layout DRAM-interleaved:
    pytest tests/ttnn/nightly/unit_tests/operations/experimental/quasar/test_routed_expert_ffn.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _to_device(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize(
    "m, k, h",
    [
        (32, 64, 64),  # 1 tile row: 1 Tensix
        (64, 64, 64),  # 2 tile rows: 2 Tensix
        (128, 64, 64),  # 4 tile rows: all 4 Tensix
        (256, 64, 64),  # 8 tile rows: 4 Tensix, 2 rows each
    ],
)
def test_routed_expert_ffn(device, m, k, h):
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.bfloat16)
    # Scale the weights so gate, up and their product stay near unit range in bfloat16.
    w_gate = (torch.randn((k, h)) / k**0.5).to(torch.bfloat16)
    w_up = (torch.randn((k, h)) / k**0.5).to(torch.bfloat16)
    w_down = (torch.randn((h, k)) / h**0.5).to(torch.bfloat16)

    y_tt = ttnn.experimental.quasar.routed_expert_ffn(
        _to_device(x, device), _to_device(w_gate, device), _to_device(w_up, device), _to_device(w_down, device)
    )

    assert tuple(y_tt.shape) == (m, k)
    assert y_tt.dtype == ttnn.bfloat16
    assert y_tt.layout == ttnn.TILE_LAYOUT

    xf = x.float()
    golden = ((xf @ w_gate.float()) * (xf @ w_up.float())) @ w_down.float()
    assert_with_pcc(ttnn.to_torch(y_tt).float(), golden, 0.99)


def test_routed_expert_ffn_rejects_mismatched_shapes(device, expect_error):
    x = _to_device(torch.randn((32, 64), dtype=torch.bfloat16), device)
    w = _to_device(torch.randn((64, 64), dtype=torch.bfloat16), device)
    w_bad = _to_device(torch.randn((32, 64), dtype=torch.bfloat16), device)
    with expect_error(RuntimeError, "w_gate and w_up must be"):
        ttnn.experimental.quasar.routed_expert_ffn(x, w_bad, w, w)
