# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debugging tests for groupnorm_sc_N_1_HW_C.

DO NOT DELETE — documents the debugging process.

Bisects the two Phase-0 failure classes seen on the acceptance file:
  (A) num_group_tiles Ng = 2 (G > 32): PCC ~0.5 on (1,1,64,2048) G=64 in both
      layouts -> slot-1 handling somewhere in membership / aggregation /
      gather / expansion. Reproduced here on the smallest shape (2 cores).
  (B) forced streaming regime: hang (writer end-of-kernel atomic assert) then
      PCC 0 on the tests that followed on the reset device. Reproduced here on
      1-core and 4-core shapes plus the shape that hung.

Inputs are seeded randn (the PCC metric of the acceptance test) plus a
"channel-index ramp" case whose group means are hand-calculable:
    x[n, 0, h, c] = c  ->  every group's per-channel column mean is c itself,
    so with G = C (Cg = 1) the variance is 0 and y = beta (or 0).
"""

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C, config

from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C import (
    torch_groupnorm_n_1_hw_c,
    compute_pcc,
)


def _run(device, x, num_groups, *, layout, gamma=None, beta=None, eps=1e-5):
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kw = {"eps": eps}
    if gamma is not None:
        kw["gamma"] = ttnn.from_torch(
            gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if beta is not None:
        kw["beta"] = ttnn.from_torch(
            beta,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    tt_y = groupnorm_sc_N_1_HW_C(tt_x, num_groups, **kw)
    actual = ttnn.to_torch(tt_y).float()
    expected = torch_groupnorm_n_1_hw_c(x, num_groups, gamma=gamma, beta=beta, eps=eps).float()
    return actual, expected


def _check(actual, expected, tag):
    pcc = compute_pcc(actual, expected)
    max_diff = (actual - expected).abs().max().item()
    assert torch.isfinite(actual).all(), f"{tag}: non-finite output"
    assert pcc >= 0.995, f"{tag}: PCC {pcc:.6f}, max diff {max_diff:.4f}"


# --------------------------------------------------------------------------- #
# (A) Ng = 2: two group-slot tiles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_ng2_smallest_two_cores(device, layout):
    """(1,1,32,64) G=64 -> Cg=1, Ct=2, HWt=1: split picks 2 cores with K=1, Ng=2."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 64).to(torch.bfloat16)
    actual, expected = _run(device, x, 64, layout=layout)
    _check(actual, expected, "ng2 two cores")


def test_ng2_two_cores_channel_ramp(device):
    """x = channel index (constant along HW) with G = C: every group is one
    channel, variance 0 -> y == beta everywhere. Group g lives in slot g//32,
    lane g%32, so slot-1 groups are channels 32..63 (the second core's tile)."""
    C = 64
    x = torch.arange(C, dtype=torch.float32).view(1, 1, 1, C).expand(1, 1, 32, C).contiguous().to(torch.bfloat16)
    beta = torch.arange(C, dtype=torch.float32).view(1, 1, 1, C).to(torch.bfloat16) * 0.5
    actual, expected = _run(device, x, C, layout=ttnn.TILE_LAYOUT, beta=beta)
    # Report per-half deviation so slot 0 vs slot 1 is visible in the failure.
    # Bound: this construction has ZERO variance, so rstd = rsqrt(eps) ~ 316 and
    # the FPU's tf32 rounding of the expanded mean (~2^-11 * |c|, op_design.md
    # "tf32 operand rounding") is amplified to ~316 * 2^-11 * 63 ~ 10. A broken
    # slot shows up as ~316 * |x| ~ 2e4 instead (the original symptom).
    d0 = (actual[..., :32] - expected[..., :32]).abs().max().item()
    d1 = (actual[..., 32:] - expected[..., 32:]).abs().max().item()
    assert d0 < 20.0 and d1 < 20.0, f"slot0 max diff {d0:.4f}, slot1 max diff {d1:.4f}"


def test_ng2_single_core_column(device):
    """(1,1,32,64) G=64 with c_splits forced to 1 via MAX_CORE_C_TILES is not
    reachable (the split minimizes per-core tiles); instead use HW=32, C=64 and
    G=64 but N=2 so both cores see two images (exercises the ring twice)."""
    torch.manual_seed(1)
    x = torch.randn(2, 1, 32, 64).to(torch.bfloat16)
    actual, expected = _run(device, x, 64, layout=ttnn.TILE_LAYOUT)
    _check(actual, expected, "ng2 N=2")


# --------------------------------------------------------------------------- #
# (B) forced streaming regime
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 32, 32), 1, id="stream_1core"),
        pytest.param((1, 1, 64, 64), 2, id="stream_4cores"),
        pytest.param((1, 1, 128, 64), 2, id="stream_multichunk"),
        pytest.param((1, 1, 1024, 320), 32, id="stream_sd_C320"),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_streaming(device, monkeypatch, shape, num_groups, layout):
    monkeypatch.setattr(config, "FORCE_STREAMING", True)
    torch.manual_seed(2)
    x = torch.randn(shape).to(torch.bfloat16)
    C = shape[-1]
    gamma = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    beta = torch.randn(1, 1, 1, C).to(torch.bfloat16)
    actual, expected = _run(device, x, num_groups, layout=layout, gamma=gamma, beta=beta)
    _check(actual, expected, f"streaming {shape}")


def test_streaming_small_chunk_ragged_tail(device, monkeypatch):
    """CHUNK_TILES_TARGET=2 with K=2 -> Q=1; HWt=3 per core-row gives ragged
    per-core row counts (2,1) and multi-chunk passes with pads."""
    monkeypatch.setattr(config, "FORCE_STREAMING", True)
    monkeypatch.setattr(config, "CHUNK_TILES_TARGET", 2)
    torch.manual_seed(3)
    x = torch.randn(1, 1, 96, 64).to(torch.bfloat16)
    actual, expected = _run(device, x, 2, layout=ttnn.TILE_LAYOUT)
    _check(actual, expected, "streaming ragged")
