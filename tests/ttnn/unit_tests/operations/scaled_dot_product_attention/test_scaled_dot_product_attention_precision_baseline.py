# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for scaled_dot_product_attention (Phase-0 supported corner:
bfloat16, TILE, tile-aligned, self, MHA, no mask, auto scale, HiFi2 + fp32 DEST).

Records PCC, max/mean abs error, relative RMS error, and the got/true ratio
spread (median + p5/p95) — the scale-bug detector: a tight cluster around a
non-1.0 constant is a uniform scale/structural bug; a symmetric spread centered
on 1.0 is ordinary bf16 precision noise.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 4, 128, 64),  # medium multi-head
    (1, 8, 256, 64),  # larger multi-head
    (1, 4, 512, 64),  # multi-KV-chunk (deeper sequence)
]


def _reference(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float())


@pytest.mark.parametrize("shape", SHAPES, ids=[f"B{s[0]}_H{s[1]}_S{s[2]}_D{s[3]}" for s in SHAPES])
def test_precision_baseline(device, shape):
    torch.manual_seed(42)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)

    expected = _reference(Q, K, V)

    to_dev = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V))).float()

    diff = (out - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (torch.sqrt((diff**2).mean()) / expected.std()).item()

    mask = expected.abs() > 1e-6
    ratio = out[mask] / expected[mask]
    r_med = ratio.median().item()
    r_p5 = ratio.quantile(0.05).item()
    r_p95 = ratio.quantile(0.95).item()

    _, allclose = comp_allclose(expected, out)
    print(
        f"\n[precision] {shape}: max_abs={max_abs:.5f} mean_abs={mean_abs:.5f} "
        f"rel_rms={rel_rms:.5f} ratio med={r_med:.4f} p5={r_p5:.4f} p95={r_p95:.4f}\n"
        f"           {allclose}"
    )

    # PCC is the gate; the printed metrics are the recorded baseline.
    assert_with_pcc(expected, out, 0.995)
