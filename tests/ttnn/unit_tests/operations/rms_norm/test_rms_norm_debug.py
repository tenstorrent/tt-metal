# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for rms_norm — DO NOT DELETE.

Documents the debugging that shaped the kernels. Deterministic inputs make
every intermediate hand-calculable, so a regression pinpoints the failing
phase instead of a fuzzy PCC miss.

Key regression captured here: RM (ROW_MAJOR) + non-tile-aligned H. A
partial-height tilize (tilize(1, <32)) mis-tilizes REAL rows on device, which
first showed up as a PCC miss on shapes like (1,1,17,64) RM. The fix: the RM
reader emits a FULL 32-row tile-row (real rows read, H-padding rows zeroed) so
the compute always tilizes a full tile (mirroring the TILE regime). The
`test_rms_norm_debug_row_constant_*` cases below are the deterministic probe
that isolated it — each row is a constant c, so RMSNorm(row) == 1.0 for every
element, and any wrong row is immediately visible.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("H", [17, 32, 33, 100])
def test_rms_norm_debug_row_constant(device, layout, H):
    """Row r = constant (r+1). With eps tiny, RMSNorm(row) == 1.0 for every
    element and every row — so the output must be all-ones over the real rows.
    This isolates per-row correctness independent of W-reduction magnitude and
    catches H-padding / tilize row-placement bugs directly."""
    W = 64
    x = torch.zeros(1, 1, H, W, dtype=torch.float32)
    for r in range(H):
        x[0, 0, r, :] = float(r + 1)

    ti = ttnn.from_torch(x, dtype=ttnn.float32, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = rms_norm(ti, epsilon=1e-8, compute_kernel_config=_cfg())
    act = ttnn.to_torch(out).reshape(1, 1, H, W).to(torch.float32)

    expected = torch.ones(1, 1, H, W, dtype=torch.float32)
    max_diff = (act - expected).abs().max().item()
    assert torch.allclose(act, expected, atol=2e-2), (
        f"H={H} layout={layout}: expected all 1.0, max_diff={max_diff}\n"
        f"per-row means: {[round(v, 3) for v in act[0, 0].mean(dim=1).tolist()]}"
    )


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_rms_norm_debug_w_nonaligned_ones(device, layout):
    """All-ones over a non-tile-aligned W. mean(1^2)=1, rsqrt(1+eps)≈1, so the
    output is ≈1.0 over all W real columns. Verifies the masked (partial-scaler)
    reduce divides by the TRUE W (not Wt*32) and the W-tail is zeroed cleanly."""
    W = 50  # W % 32 == 18
    x = torch.ones(1, 1, 32, W, dtype=torch.float32)
    ti = ttnn.from_torch(x, dtype=ttnn.float32, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = rms_norm(ti, epsilon=1e-8, compute_kernel_config=_cfg())
    act = ttnn.to_torch(out).reshape(1, 1, 32, W).to(torch.float32)
    expected = torch.ones(1, 1, 32, W, dtype=torch.float32)
    max_diff = (act - expected).abs().max().item()
    assert torch.allclose(act, expected, atol=2e-2), f"max_diff={max_diff}"
