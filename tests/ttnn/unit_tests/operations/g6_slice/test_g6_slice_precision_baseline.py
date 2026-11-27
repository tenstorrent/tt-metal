# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of g6_slice (Phase 0: bfloat16, TILE, DRAM interleaved).

The op is a bit copy, so the expected baseline is PCC 1, zero error and zero ULP in every regime.
The relative RMS, ULP and got/true ratio spread are recorded so a later refinement that routes a
value through another format (forbidden by the spec) shows up as a number, not only as a failure.
Shapes are production cases (one per regime) plus the rule case of an output tile fed by two tiles.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_slice import describe_plan, g6_slice

CASES = {
    # (shape, starts, ends, regime)
    "rule_two_tiles_small": ((1, 1, 128, 64), (0, 0, 17, 0), (1, 1, 82, 64), "row_runs"),
    "rule_w_and_h_begin_in_tile": ((1, 1, 96, 128), (0, 0, 17, 20), (1, 1, 90, 120), "col_realign"),
    "c179_medium": ((1, 2304, 3072), (0, 0, 0), (1, 2048, 3072), "page_copy"),
    "c587_row_4095": ((16, 4096, 1280), (0, 4095, 0), (16, 4096, 1280), "row_runs"),
    "c1112_large": ((16, 1024, 4160), (0, 0, 0), (16, 1024, 2560), "page_copy"),
}


def _bf16_bits(t):
    return t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _ordered(bits):
    """bf16 bit pattern -> integer on a monotonic scale, so |a - b| counts ULPs across signs."""
    return torch.where(bits >= 0x8000, 0x8000 - bits, bits)


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_g6_slice_precision_baseline(device, case):
    shape, starts, ends, regime = CASES[case]
    torch.manual_seed(1234)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert describe_plan(x, list(starts), list(ends))["regime"] == regime

    expected = torch_x[tuple(slice(b, e) for b, e in zip(starts, ends))].contiguous()
    actual = ttnn.to_torch(g6_slice(x, list(starts), list(ends)))
    assert actual.shape == expected.shape

    passing_pcc, pcc = comp_pcc(expected, actual, pcc=0.99999)
    _, allclose_msg = comp_allclose(expected, actual)

    exp32, act32 = expected.float(), actual.float()
    err = act32 - exp32
    max_abs = err.abs().max().item()
    mean_abs = err.abs().mean().item()
    rel_rms = (err.pow(2).mean().sqrt() / exp32.std()).item()
    max_ulp = int((_ordered(_bf16_bits(actual)) - _ordered(_bf16_bits(expected))).abs().max())

    nonzero = exp32 != 0
    ratio = act32[nonzero] / exp32[nonzero]
    r_med = ratio.median().item()
    r_p5, r_p95 = torch.quantile(ratio[:1_000_000], torch.tensor([0.05, 0.95])).tolist()

    print(
        f"\nPRECISION {case} regime={regime} out={tuple(expected.shape)} pcc={pcc} "
        f"max_abs={max_abs:.3g} mean_abs={mean_abs:.3g} rel_rms={rel_rms:.3g} max_ulp={max_ulp} "
        f"ratio_median={r_med:.6g} ratio_p5={r_p5:.6g} ratio_p95={r_p95:.6g} [{allclose_msg}]"
    )

    assert passing_pcc
    assert max_ulp == 0, "g6_slice must copy bits: any ULP difference is a format conversion or a misplaced value"
    assert max_abs == 0.0 and rel_rms == 0.0
    assert r_med == 1.0 and r_p5 == 1.0 and r_p95 == 1.0
