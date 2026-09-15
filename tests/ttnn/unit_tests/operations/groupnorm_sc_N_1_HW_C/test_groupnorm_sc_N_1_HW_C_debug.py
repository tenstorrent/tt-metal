# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debugging tests for groupnorm_sc_N_1_HW_C — DO NOT DELETE.

Hand-calculable inputs that isolate the pieces of the pipeline. They document the
debugging done while bringing the op up:

* RM-layout gamma/beta rows were read half wrong (PCC ~0.5): the second 16-lane half of each
  channel tile was fetched with a NoC read whose source (+32 B in the stick) and destination
  (+512 B, face 1) had different residues modulo the 64 B Blackhole DRAM alignment. The
  `test_affine_rm_lane_pattern` case makes that failure mode visible per lane: gamma encodes
  its own channel index, so a wrong half shows up as a block of 16 wrong columns.
* `test_all_ones_*`: with x == 1 the statistics are trivially mean = 1, var = 0, so
  y = beta everywhere (or 0 without affine) — any non-zero output without affine means the
  variance / rsqrt path or the per-group aggregation is off.
"""

import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C


def _to_device(t, layout, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_all_ones_no_affine(device):
    """x = 1 everywhere → mean = 1, var = 0 → y = (1 - 1) * rsqrt(eps) = 0 exactly."""
    shape, G = (1, 1, 64, 320), 32  # C/G = 10 (group straddling)
    x = torch.ones(shape, dtype=torch.bfloat16)
    out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(_to_device(x, ttnn.TILE_LAYOUT, device), G)).float()
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e-3, f"max |y| = {out.abs().max()} (expected 0)"


def test_all_ones_affine_gives_beta(device):
    """x = 1 → y = 0 * gamma + beta = beta for every row; beta[c] = c / 64 is exactly representable."""
    shape, G = (1, 1, 64, 128), 4
    C = shape[-1]
    x = torch.ones(shape, dtype=torch.bfloat16)
    gamma = torch.full((1, 1, 1, C), 2.0, dtype=torch.bfloat16)
    beta = (torch.arange(C, dtype=torch.float32) / 64.0).reshape(1, 1, 1, C).to(torch.bfloat16)
    out = ttnn.to_torch(
        groupnorm_sc_N_1_HW_C(
            _to_device(x, ttnn.TILE_LAYOUT, device),
            G,
            gamma=_to_device(gamma, ttnn.ROW_MAJOR_LAYOUT, device),
            beta=_to_device(beta, ttnn.ROW_MAJOR_LAYOUT, device),
        )
    ).float()
    expected = beta.float().expand(shape)
    assert torch.allclose(out, expected, atol=1e-2), f"max diff {(out - expected).abs().max()}"


def test_affine_rm_lane_pattern(device):
    """gamma[c] = c encodes the channel index; with x drawn from a fixed pattern the normalized value
    z is the same for every channel of a row, so y[:, c] / z == gamma[c] exposes any lane permutation
    or half-tile misread of the RM gamma stick (the bug this test was written for)."""
    shape, G = (1, 1, 32, 64), 1  # one group: every channel shares mean/rstd
    C = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    gamma = torch.arange(1, C + 1, dtype=torch.float32).reshape(1, 1, 1, C).to(torch.bfloat16)  # 1..C exact
    out = ttnn.to_torch(
        groupnorm_sc_N_1_HW_C(
            _to_device(x, ttnn.TILE_LAYOUT, device), G, gamma=_to_device(gamma, ttnn.ROW_MAJOR_LAYOUT, device)
        )
    ).float()
    xf = x.float()
    z = (xf - xf.mean()) * torch.rsqrt(xf.var(unbiased=False) + 1e-5)
    expected = z * gamma.float()
    # per-lane check: a misread half-tile shows as 16 consecutive wrong columns
    col_err = (out - expected).abs().amax(dim=(0, 1, 2))
    bad = torch.nonzero(col_err > 0.25).flatten().tolist()
    assert not bad, f"columns with wrong gamma: {bad}"
