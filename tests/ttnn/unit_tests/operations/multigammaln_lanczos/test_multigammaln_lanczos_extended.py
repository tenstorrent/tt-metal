# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Extended tests for `multigammaln_lanczos`.

Small, focused matrix — intended to catch real issues without exploding the
test runtime. The acceptance test (`test_multigammaln_lanczos.py`) already
covers the main shape sweep, exact-pole zeroing, hard sub-domain, and the
negative-validation cases. This file fills in:

  - **L1-memory-config output path** — confirms `memory_config=L1_MEMORY_CONFIG`
    is plumbed through correctly.
  - **Large stress shape** — one 512×512 single-tile-batch shape that exercises
    work distribution past the full Tensix grid.
  - **Determinism** — running the same input twice yields bit-identical output
    (no stale-CB or use-after-pop bug).

Refinements (dtype, compute config exposure, sharded I/O, rank generalisation,
non-tile-aligned shapes) are deliberately NOT covered here — those belong to
their respective refinement phases.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos


PCC = 0.999  # Lanczos-at-fp32 precision bound.


def _torch_ref(a: torch.Tensor) -> torch.Tensor:
    return torch.special.multigammaln(a.float(), 4)


def _in_domain_input(shape, seed=42):
    torch.manual_seed(seed)
    return (2.0 + 8.0 * torch.rand(shape, dtype=torch.float32)).clamp(min=2.0, max=10.0)


@pytest.mark.parametrize(
    "out_memcfg",
    [
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram_output"),
        pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1_output"),
    ],
)
def test_multigammaln_lanczos_output_memory_config(device, out_memcfg):
    """`memory_config=` should plumb through to the output tensor (DRAM or L1)."""
    shape = (1, 1, 64, 128)
    torch_input = _in_domain_input(shape)
    torch_expected = _torch_ref(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln_lanczos(ttnn_input, memory_config=out_memcfg)
    actual = ttnn.to_torch(ttnn_output).float()

    ok, msg = check_with_pcc(torch_expected, actual, pcc=PCC)
    assert ok, msg


def test_multigammaln_lanczos_large_shape_stress(device):
    """
    One 512x512 shape — exercises work distribution beyond the full compute
    grid (1024 tiles across ~64 cores → ~16 tiles per core). Sanity check
    that multi-core dispatch and per-core RT args remain correct at scale.
    """
    shape = (1, 1, 512, 512)
    torch_input = _in_domain_input(shape, seed=7)
    torch_expected = _torch_ref(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln_lanczos(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    ok, msg = check_with_pcc(torch_expected, actual, pcc=PCC)
    assert ok, msg


def test_multigammaln_lanczos_determinism(device):
    """
    Running the same input twice must produce bit-identical output. Catches
    use-after-pop on `cb_input_tiles`, stale-CB reads in sub-phase B, and
    other timing-dependent bugs that random shape variation can mask.
    """
    shape = (1, 1, 64, 64)
    torch_input = _in_domain_input(shape, seed=11)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_1 = ttnn.to_torch(multigammaln_lanczos(ttnn_input)).float()
    out_2 = ttnn.to_torch(multigammaln_lanczos(ttnn_input)).float()

    assert torch.equal(out_1, out_2), "Two identical calls produced different output (non-deterministic)."
