# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Extended coverage for multigammaln_lanczos beyond the acceptance test.

Focus: small targeted shapes and parameter cases that exercise things the
acceptance test doesn't quite hit — value-domain behaviour at the safe-domain
edges, L1 memory config (vs DRAM), deterministic input patterns, and the
larger-than-grid multi-core split. Intentionally narrow to keep the matrix
small. Broad coverage (extra ranks, dtype widening, non-tile-aligned shapes,
sharded layouts) is deferred to refinements.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos


SAFE_LO = 2.0
SAFE_HI = 10.0

# Same precision floor recorded in the precision baseline.
PCC = 0.999
RTOL = 0.1
ATOL = 0.5


def _torch_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.special.multigammaln(x.double(), 4).float()


def _ttnn_run(device, torch_input, *, memory_config):
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return ttnn.to_torch(multigammaln_lanczos(ttnn_input)).float()


# -----------------------------------------------------------------------------
# Memory config: L1-interleaved input (acceptance test only exercises DRAM)
# -----------------------------------------------------------------------------


def test_l1_interleaved_input(device):
    """L1-interleaved memory config — the kernel should be agnostic to the
    storage location because it reads via TensorAccessor."""
    torch.manual_seed(0)
    shape = (1, 1, 64, 64)
    torch_input = SAFE_LO + (SAFE_HI - SAFE_LO) * torch.rand(shape, dtype=torch.float32)

    actual = _ttnn_run(device, torch_input, memory_config=ttnn.L1_MEMORY_CONFIG)
    expected = _torch_reference(torch_input)
    assert_with_pcc(expected, actual, PCC)
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)


# -----------------------------------------------------------------------------
# Value-domain edges within the documented safe domain
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fill_value, label",
    [
        (2.0, "low_edge_pole"),  # exact pole at a=2 inside lgamma(x)
        (2.5, "near_pole"),  # near the pole; mask should be 1
        (10.0, "high_edge"),  # upper end of safe domain
    ],
)
def test_constant_inputs_inside_safe_domain(device, fill_value, label):
    """Fill the input tile with a single value to exercise the pole-zeroing
    mask and the upper-edge magnitude in isolation."""
    shape = (1, 1, 32, 32)
    torch_input = torch.full(shape, fill_value, dtype=torch.float32)
    expected = _torch_reference(torch_input)

    actual = _ttnn_run(device, torch_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    assert torch.isfinite(actual).all(), f"non-finite output at fill={fill_value}"
    assert_with_pcc(expected, actual, PCC)
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
        f"constant input {fill_value}: actual={actual[0,0,0,0].item():.6f} " f"expected={expected[0,0,0,0].item():.6f}"
    )


# -----------------------------------------------------------------------------
# Deterministic linspace pattern (helps debugging — values are predictable)
# -----------------------------------------------------------------------------


def test_linspace_safe_domain(device):
    """A monotone input over the safe domain — every tile sees a different
    value pattern. Catches per-tile drift if it exists."""
    shape = (1, 1, 64, 128)
    n = int(torch.tensor(shape).prod().item())
    torch_input = torch.linspace(SAFE_LO, SAFE_HI, n, dtype=torch.float32).reshape(shape)
    expected = _torch_reference(torch_input)

    actual = _ttnn_run(device, torch_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert_with_pcc(expected, actual, PCC)
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)


# -----------------------------------------------------------------------------
# Shape that forces split_work_to_cores group_2 to be non-empty
#   On Wormhole (8x8 grid → 64 cores) a 60-tile shape gives group_1=60 cores @1
#   tile and group_2 has 0 cores. To force both groups, pick a count that
#   doesn't divide evenly — e.g. 65 tiles → group_1 cores @2 tiles + group_2 @1.
# -----------------------------------------------------------------------------


def test_uneven_work_split(device):
    """65 tiles on a 64-core grid → group_1 has 1 core with 2 tiles, group_2
    has 63 cores with 1 tile. Validates the per-group RT-arg walk."""
    # 65 tiles total — choose (1, 1, 32, 65*32) = (1, 1, 32, 2080).
    shape = (1, 1, 32, 65 * 32)
    torch.manual_seed(11)
    torch_input = SAFE_LO + (SAFE_HI - SAFE_LO) * torch.rand(shape, dtype=torch.float32)

    actual = _ttnn_run(device, torch_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    expected = _torch_reference(torch_input)
    assert_with_pcc(expected, actual, PCC)
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)
