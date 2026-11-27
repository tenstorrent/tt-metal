# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_repeat.

The op copies bits and computes nothing, so the only acceptable baseline is exact:
PCC 1, zero absolute and relative RMS error, zero ULP distance, and a got/true ratio of
exactly 1 everywhere. The metrics are still measured the standard way so a regression
shows up as numbers, not just as a failed bit compare.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_repeat import g6_repeat


def _ordered_bits(t):
    """bfloat16 bit patterns mapped to a monotonic integer line, so |a - b| is the ULP distance."""
    b = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(b < 0, -(b & 0x7FFF), b)


@pytest.mark.parametrize(
    "w, r",
    [(256, 16), (64, 17), (96, 31), (9600, 5)],
    ids=["p1_w256_r16", "w64_r17", "w96_r31", "w9600_r5"],
)
def test_g6_repeat_precision_baseline(device, w, r):
    torch.manual_seed(0)
    torch_x = torch.randn((1, 1, 1, w), dtype=torch.float32).to(torch.bfloat16)
    reps = [1, 1, r, 1]
    expected = torch_x.repeat(*reps)

    ttnn_x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    actual = ttnn.to_torch(g6_repeat(ttnn_x, reps))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype

    _, pcc = comp_pcc(expected, actual, pcc=1.0)
    _, allclose_msg = comp_allclose(expected, actual)
    err = (actual.float() - expected.float()).abs()
    max_abs, mean_abs = float(err.max()), float(err.mean())
    rel_rms = float(err.pow(2).mean().sqrt() / expected.float().std())
    max_ulp = int((_ordered_bits(actual) - _ordered_bits(expected)).abs().max())

    nonzero = expected.float() != 0
    ratio = actual.float()[nonzero] / expected.float()[nonzero]
    q = torch.quantile(ratio, torch.tensor([0.05, 0.5, 0.95]))

    print(
        f"\nPRECISION g6_repeat w={w} r={r}: pcc={pcc} max_abs={max_abs} mean_abs={mean_abs} "
        f"rel_rms={rel_rms} max_ulp={max_ulp} ratio_p5/p50/p95={q.tolist()} | {allclose_msg}"
    )

    assert float(pcc) == 1.0
    assert max_abs == 0.0 and mean_abs == 0.0 and rel_rms == 0.0
    assert max_ulp == 0
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
