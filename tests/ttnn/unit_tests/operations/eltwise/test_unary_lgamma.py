# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Boundary and range coverage for lgamma's Chebyshev and Taylor bridges.

The fp32 arm evaluates a Chebyshev bridge on [0.5, 2.0], a Taylor series around
z = 2 on [2.0, 2.2], and the Stirling asymptotic expansion for z >= 2.2.
"""

import pytest
import torch

import ttnn

# (boundary, the neighbour just inside the bridge it belongs to)
_BOUNDARIES = [
    (0.5, 0.5000001),
    (1.0, 1.0000001),
    (2.0, 1.9999999),
    (2.2, 2.1999998),
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


def test_lgamma_bridge_accuracy(device):
    values = torch.linspace(0.5, 3.0, 64, dtype=torch.float32)
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded.view(-1)[: values.numel()] = values

    tt_in = ttnn.from_torch(padded, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.lgamma(tt_in)).view(-1)[: values.numel()].double()
    ref = torch.lgamma(values.double())

    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5), (
        f"lgamma accuracy degraded across bridge range [0.5, 3.0]; "
        f"max diff = {(got - ref).abs().max().item():.4e}"
    )
