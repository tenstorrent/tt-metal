# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""lgamma regression tests that the category tests do not cover.

1. Boundary coverage for the fp32 arm's two Taylor bridges (test_lgamma_bridge_boundary_is_no_worse_than_the_bridge).
2. The bf16 arm under a 32-bit Dest, i.e. a bf16 input with a preallocated float32 output
   (test_lgamma_bf16_input_fp32_output), checked against absolute/relative bf16-grade bounds.
"""

import pytest
import torch

import ttnn

# (boundary, the neighbour just inside the bridge it belongs to)
_BOUNDARIES = [
    (0.75, 0.7500001),
    (1.25, 1.2499999),
    (1.75, 1.7500001),
    (2.25, 2.2499999),
]


@pytest.mark.parametrize("z, inside", _BOUNDARIES)
def test_lgamma_bridge_boundary_is_no_worse_than_the_bridge(device, z, inside):
    """The fp32 arm evaluates a series around z = 1 while |z-1| is inside 0.25, and one around z = 2
    while |z-2| is inside 0.25. Both series are fitted on the closed interval, so the four inputs
    that sit exactly on |d| = 0.25 belong to a bridge; a strict guard sends them to the Stirling
    arm instead, which is an asymptotic expansion for large z.

    The assertion is scale-free on purpose: a boundary input must be no worse than the input one
    step inside the bridge it belongs to. Absolute bounds would have to be re-tuned every time the
    bridge coefficients move, and the property being tested is the branch selection, not the fit."""
    values = torch.tensor([z, inside], dtype=torch.float32)
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded.view(-1)[: values.numel()] = values

    tt_in = ttnn.from_torch(padded, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.lgamma(tt_in)).view(-1)[: values.numel()].double()
    ref = torch.lgamma(values.double())

    err_boundary = abs(got[0] - ref[0]).item()
    err_inside = abs(got[1] - ref[1]).item()

    assert err_boundary <= 2.0 * err_inside, (
        f"lgamma({z}) is off by {err_boundary:.4e} while lgamma({inside}) is off by "
        f"{err_inside:.4e} ({err_boundary / err_inside:.1f}x). The boundary is being sent to "
        f"the Stirling arm instead of the bridge it belongs to."
    )


def test_lgamma_bf16_input_fp32_output(device):
    """A preallocated float32 output puts Dest in 32-bit mode while the kernel is still the bf16 one.
    Every element of its chain must then address full 32-bit Dest words; the where in the chain used
    to address 16-bit halves, which under a 32-bit Dest land on the Stirling tile instead, so the
    lower half of every tile came back corrupted. The tensor is dense so every row is checked."""
    # bf16 non-integers on a GRID_STEP grid in [GRID_LO, GRID_HI), kept POLE_MARGIN away from every integer
    # (poles for x <= 0, and lgamma is exactly 0 at 1 and 2), repeated to fill the tensor; the first lanes
    # are the non-positive integer poles themselves, where the chain's where selects its true value (0)
    # and lgamma must come back +inf.
    GRID_STEP = 1.0 / 16
    POLE_MARGIN = 2 * GRID_STEP
    GRID_LO, GRID_HI = -4.5, 10.0
    POLES = [0.0, -1.0, -2.0, -3.0]
    grid = torch.arange(GRID_LO, GRID_HI, GRID_STEP, dtype=torch.float32)
    grid = grid[(grid - torch.round(grid)).abs() >= POLE_MARGIN]
    shape = (1, 1, 64, 128)
    numel = torch.Size(shape).numel()
    x = grid.repeat(numel // grid.numel() + 1)[:numel].reshape(shape)
    x.view(-1)[: len(POLES)] = torch.tensor(POLES)
    bf16_in = x.to(torch.bfloat16)
    assert torch.equal(bf16_in.float(), x), "inputs must be exact in bf16"

    ref = torch.lgamma(x)
    tt_in = ttnn.from_torch(bf16_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.from_torch(torch.zeros_like(x), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.lgamma(tt_in, output_tensor=tt_out)
    got = ttnn.to_torch(tt_out)

    assert got.dtype == torch.float32
    is_pole = torch.isposinf(ref)
    assert torch.equal(is_pole.view(-1)[: len(POLES)], torch.ones(len(POLES), dtype=torch.bool))
    assert torch.all(torch.isposinf(got[is_pole])), f"poles: got {got.view(-1)[: len(POLES)].tolist()}"
    got, ref = got[~is_pole], ref[~is_pole]
    x = x[~is_pole]
    # The bf16 kernel evaluates Stirling plus a reflection, so a bf16-grade result is expected: |ref| < 16
    # here, so one bf16 ulp is at most 1/16; allow just under one ulp absolute plus a few ulps relative.
    abs_tol = 0.05
    rel_tol = 0.02
    tol = abs_tol + rel_tol * ref.abs()
    err = (got - ref).abs()
    worst = err.view(-1).argmax()
    assert torch.all(err <= tol), (
        f"{(err > tol).sum().item()} of {err.numel()} lanes off; worst at "
        f"x={x.view(-1)[worst].item()}: got {got.view(-1)[worst].item():.4f}, ref {ref.view(-1)[worst].item():.4f}"
    )
