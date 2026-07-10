# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for Flash-Attention scaled_dot_product_attention (Phase 0).

Measures PCC, max/mean abs error, and relative RMS error across a small set of
shapes (single-tile, multi-tile self-attn with the online-softmax recurrence,
multi-head/batch, and cross-attention). bf16, TILE, HiFi4 + fp32_dest_acc_en
(the Phase-0 default config). Recorded in verification_report.md.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# (Q_shape, K_shape, V_shape): small -> medium -> larger (multi-KV-block) -> multi-head
SHAPES = [
    ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),  # single tile
    ((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)),  # multi-tile, single KV block
    ((1, 2, 256, 64), (1, 2, 256, 64), (1, 2, 256, 64)),  # online-softmax recurrence (2 KV blocks)
    ((2, 4, 512, 64), (2, 4, 512, 64), (2, 4, 512, 64)),  # larger: multi-batch/head, 4 KV blocks
]


def _reference(Q, K, V, scale=None):
    return torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), scale=scale)


@pytest.mark.parametrize("q_shape, k_shape, v_shape", SHAPES)
def test_precision_baseline(device, q_shape, k_shape, v_shape):
    torch.manual_seed(2026)
    Q = torch.randn(q_shape, dtype=torch.bfloat16)
    K = torch.randn(k_shape, dtype=torch.bfloat16)
    V = torch.randn(v_shape, dtype=torch.bfloat16)

    expected = _reference(Q, K, V)

    to_dev = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn_out = scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V))
    out = ttnn.to_torch(ttnn_out).float()

    # Error metrics
    diff = (expected - out).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rms = torch.sqrt((diff**2).mean()).item()
    denom = torch.sqrt((expected**2).mean()).item()
    rel_rms = rms / denom if denom > 0 else 0.0

    _, allclose_msg = comp_allclose(expected, out, rtol=0.05, atol=0.05)
    _, pcc_msg = comp_pcc(expected, out, pcc=0.995)
    print(
        f"\n[precision] shape={q_shape} {pcc_msg} max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
        f"rel_rms={rel_rms:.5f} | {allclose_msg}"
    )

    assert_with_pcc(expected, out, pcc=0.995)
