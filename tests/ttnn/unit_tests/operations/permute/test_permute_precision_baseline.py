# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for permute (Phase 0: fp32 / TILE / rank 4 / DRAM interleaved).

permute is pure data movement, so the expectation is bit-exactness; the metrics
below exist to make any future regression (or a scale/structural bug) visible.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

SHAPES = [
    (1, 1, 32, 64),
    (2, 4, 64, 128),
    (4, 8, 128, 256),
    (2, 4, 512, 512),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dims", [(1, 0, 2, 3)])
def test_permute_precision_baseline(device, shape, dims):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    expected = torch.permute(torch_input, dims).contiguous()

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(permute_op(tt_in, dims)).to(torch.float32)

    a, e = actual.flatten().double(), expected.flatten().double()
    abs_err = (a - e).abs()
    rms = (abs_err.pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()

    # scale-bug detector: got/true ratio spread over finite non-zero references
    mask = (e != 0) & torch.isfinite(e) & torch.isfinite(a)
    r = a[mask] / e[mask]
    med = r.median().item()
    p5, p95 = torch.quantile(r, torch.tensor([0.05, 0.95], dtype=torch.float64)).tolist()

    print(
        f"shape={shape} dims={dims} max_abs={abs_err.max().item():.3e} "
        f"mean_abs={abs_err.mean().item():.3e} rel_rms={rms:.3e} "
        f"ratio_med={med:.6f} ratio_p5={p5:.6f} ratio_p95={p95:.6f}"
    )
    print(comp_allclose(expected, actual))

    assert_with_pcc(expected, actual, 0.9999)
    assert rms == 0.0, "permute is pure relocation — expected bit-exact output"


def permute_op(t, dims):
    from ttnn.operations.permute import permute

    return permute(t, dims)
