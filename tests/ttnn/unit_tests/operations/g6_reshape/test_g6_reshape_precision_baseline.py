# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_reshape (Phase 0: bfloat16 and float32, TILE_LAYOUT, DRAM interleaved).

The op moves values and never computes on them, so the baseline is bit-exactness: PCC 1, every
error metric 0, every got/true ratio exactly 1. The metrics are still measured so a later
refinement that routes values through a converting path shows up as a number, not only as a
failed bit compare. Run with -s to print the table.
"""

import pytest
import torch

import ttnn
from models.common.metrics import compute_mean_abs_error
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_reshape import describe_plan, g6_reshape

# (input shape, target shape, dtype): small, medium and large bfloat16 tile_face_row production cases,
# and the FLUX row-to-column geometry, which runs the element-run regime, at both dtypes.
CASES = [
    ((16, 1, 64), (1, 1, 16, 64), ttnn.bfloat16),
    ((1, 16, 6, 256), (1, 1, 16, 1536), ttnn.bfloat16),
    ((16, 1, 5120), (1, 1, 16, 5120), ttnn.bfloat16),
    ((1, 185), (1, 1, 185, 1), ttnn.bfloat16),
    ((1, 512), (1, 1, 512, 1), ttnn.float32),
]

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _case_id(case):
    src, dst, dtype = case
    return "x".join(map(str, src)) + "-to-" + "x".join(map(str, dst)) + "-" + dtype.name.lower()


def _ordered_bits(t):
    """Float bit patterns mapped to a monotonic integer line, so |a - b| counts ULPs."""
    if t.dtype == torch.bfloat16:
        bits = t.contiguous().view(torch.int16).to(torch.int64)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    bits = t.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def _max_atol(details):
    return float(details.split("Max ATOL Delta:")[1].split(",")[0])


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_precision_baseline(case, device):
    src, dst, dtype = case
    torch.manual_seed(7)
    torch_input = torch.randn(src).to(_TORCH_DTYPE[dtype])
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    regime = describe_plan(ttnn_input, list(dst))["regime"]
    got = ttnn.to_torch(g6_reshape(ttnn_input, list(dst)))
    expected = torch.reshape(torch_input, dst)

    _, pcc = comp_pcc(expected, got, pcc=0.99999)
    _, allclose_details = comp_allclose(expected, got)
    max_abs = _max_atol(allclose_details)
    mean_abs = compute_mean_abs_error(got.float(), expected.float())
    diff = got.float() - expected.float()
    rel_rms = (diff.pow(2).mean().sqrt() / expected.float().std()).item()
    max_ulp = int((_ordered_bits(got) - _ordered_bits(expected)).abs().max())

    nonzero = expected.float() != 0
    ratio = got.float()[nonzero] / expected.float()[nonzero]
    ratio_median = ratio.median().item()
    ratio_p5, ratio_p95 = (torch.quantile(ratio, q).item() for q in (0.05, 0.95))

    print(
        f"\n{_case_id(case)} [{regime}] PCC={pcc:.6f} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rel_rms={rel_rms:.3e} max_ulp={max_ulp} ratio median={ratio_median:.6f} "
        f"p5={ratio_p5:.6f} p95={ratio_p95:.6f}"
    )

    assert pcc >= 0.99999
    assert max_abs == 0.0 and mean_abs == 0.0 and rel_rms == 0.0
    assert max_ulp == 0
    assert ratio_median == 1.0 and ratio_p5 == 1.0 and ratio_p95 == 1.0
    raw = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.contiguous().view(raw), expected.contiguous().view(raw))
