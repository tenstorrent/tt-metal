# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Boundary coverage for lgamma's two Taylor bridges.

The fp32 arm evaluates a series around z = 1 while |z-1| is inside 0.25, and one around z = 2
while |z-2| is inside 0.25. Both series are fitted on the closed interval, so the four inputs
that sit exactly on |d| = 0.25 belong to a bridge; a strict guard sends them to the Stirling
arm instead, which is an asymptotic expansion for large z.

The assertion is scale-free on purpose: a boundary input must be no worse than the input one
step inside the bridge it belongs to. Absolute bounds would have to be re-tuned every time the
bridge coefficients move, and the property being tested is the branch selection, not the fit.
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


# multigammaln (p = 4) is computed from two lgamma evaluations plus ln((a - 1)(a - 1.5)) via
# lgamma(a - 1) = lgamma(a) - ln(a - 1). Evaluating lgamma(a - 1.5) directly near the a > 1.5 domain
# edge cost up to ~1e-2 relative error in fp32; the recurrence keeps the whole domain near fp32 accuracy.
def test_multigammaln_fp32(device):
    x = torch.linspace(1.5 + 2**-10, 100.0, 32 * 32 * 8, dtype=torch.float32).reshape(1, 1, 256, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.multigammaln(tt_x)).double()
    expected = torch.special.multigammaln(x.double(), 4)
    rel = (result - expected).abs() / expected.abs().clamp(min=1.0)
    assert rel.max().item() < 1e-4, f"max relative error {rel.max().item():.3e} at x={x.flatten()[rel.argmax()].item()}"
