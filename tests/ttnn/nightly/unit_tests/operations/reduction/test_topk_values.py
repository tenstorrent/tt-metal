# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Value-fidelity characterization for ttnn.topk (issue #51180 nightly recharter).

Relationship to existing topk tests (do not duplicate; different goal):

- unit-tier ``tests/ttnn/unit_tests/operations/reduce/test_topk.py``: functional
  coverage of shapes / k / dim / dtypes with PCC-style value checks.
- ``tests/ttnn/unit_tests/gtests/test_reduction.cpp`` (merge gate): one exact smoke
  cell per topk program factory / index-dtype selection (incl. the #53453 multi-core
  case).
- the ``reduction/topk`` sweep is currently non-functional (100% invalidated vectors,
  see the CI-grid analysis) — nothing here depends on it.

topk is a **selection**, not an accumulation: its output values are elements of the
input, so unlike sum/mean there is no rounding budget to characterize — the sorted
top-k value sequence must match torch's **exactly**, ties included (ties make the
*indices* ambiguous, but the sorted value multiset is uniquely determined, so
comparing sorted values is tie-proof). Two invariants are asserted per case:

1. values == torch.topk(...).values, bit-exact in the input dtype;
2. self-consistency: values == input gathered at the returned indices (tie-proof,
   catches index corruption independently of golden agreement).

Tile padding is poisoned with a value that would win the selection if the kernel
ever read it (padding must never enter the top-k window).
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch
from loguru import logger

import ttnn
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_topk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topk_input(distribution: str, shape, seed: int) -> torch.Tensor:
    """FP32 test input. Distributions target selection edge regimes."""
    torch.manual_seed(seed)
    if distribution == "normal":
        return torch.randn(shape, dtype=torch.float32)
    if distribution == "all_negative":  # padding-sentinel probe: any leaked pad value wins largest=True
        return -1.0 - torch.rand(shape, dtype=torch.float32)
    if distribution == "wide_uniform":
        return torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)
    raise ValueError(f"unknown distribution {distribution}")


def _run_case(device, x, ttnn_dtype, dim, k, largest):
    """Run ttnn.topk (twice, via the determinism wrapper) with poisoned tile padding."""
    tt_input = ttnn.from_torch(x, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    # Poison padding with a value that wins largest=True selections (and its negation
    # loses smallest=False ones): if the kernel window ever covers padding, values
    # diverge from the golden and the assert below names it.
    ttnn.fill_implicit_tile_padding(tt_input, 1.0e4 if largest else -1.0e4)
    tt_values, tt_indices = ttnn_topk(tt_input, k, dim=dim, largest=largest, sorted=True)
    return ttnn.to_torch(tt_values), ttnn.to_torch(tt_indices)


def _assert_values_exact(x_quantized, torch_dtype, dim, k, largest, values, indices, spec):
    """The two invariants from the module docstring."""
    golden_values = torch.topk(x_quantized.float(), k, dim=dim, largest=largest, sorted=True).values.to(torch_dtype)
    values = values.to(torch_dtype)
    exact = torch.equal(values, golden_values)
    # Self-consistency: gather the input at the returned indices along `dim`.
    gathered = torch.gather(x_quantized, dim, indices.to(torch.int64))
    consistent = torch.equal(values, gathered.to(torch_dtype))
    logger.info(
        f"ttnn.topk values | {spec} | exact={'ok' if exact else 'FAIL'} gather={'ok' if consistent else 'FAIL'}"
    )
    if not exact:
        diff = (values.float() - golden_values.float()).abs()
        logger.info(f"  max |value - golden| = {diff.max().item():.6g} at flat {diff.argmax().item()}")
    assert consistent, f"[{spec}] returned values do not match the input gathered at the returned indices"
    assert exact, f"[{spec}] sorted top-k values are not bit-exact vs torch"


# ---------------------------------------------------------------------------
# Last-dim cases: single-core (W=64/256) and multi-core (W=8192) windows,
# k below / straddling / at the tile boundary (k=50 exercises pad+slice).
# ---------------------------------------------------------------------------

_LAST_DIM_CASES = [
    # (shape, k, id)
    ((1, 1, 32, 64), 32, "W64-k32"),
    ((1, 1, 32, 64), 50, "W64-k50-padslice"),
    ((1, 1, 32, 64), 64, "W64-k64"),
    ((1, 1, 64, 256), 32, "W256-k32"),
    ((1, 1, 64, 8192), 32, "W8192-k32-multicore"),
]


@pytest.mark.parametrize("shape, k, desc", _LAST_DIM_CASES, ids=[c[2] for c in _LAST_DIM_CASES])
@pytest.mark.parametrize("distribution", ["normal", "all_negative"])
@pytest.mark.parametrize("largest", [True, False], ids=["largest", "smallest"])
@pytest.mark.parametrize(
    "ttnn_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)], ids=["bf16", "fp32"]
)
def test_topk_values_last_dim(device, shape, k, desc, distribution, largest, ttnn_dtype, torch_dtype):
    """Sorted top-k values along -1 must be bit-exact vs torch and self-consistent with indices."""
    x = _topk_input(distribution, shape, seed=42).to(torch_dtype)
    values, indices = _run_case(device, x, ttnn_dtype, dim=-1, k=k, largest=largest)
    spec = f"{desc} {distribution} dim=-1 largest={largest} dtype={torch_dtype}"
    _assert_values_exact(x, torch_dtype, -1, k, largest, values, indices, spec)


# ---------------------------------------------------------------------------
# Non-last-dim (transpose front-end path) and non-tile-aligned rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("largest", [True, False], ids=["largest", "smallest"])
def test_topk_values_non_last_dim(device, largest):
    """dim=1 routes through the front-end transpose wrapper; value exactness must survive it."""
    shape, k = (1, 64, 32, 64), 32
    x = _topk_input("normal", shape, seed=7).to(torch.bfloat16)
    values, indices = _run_case(device, x, ttnn.bfloat16, dim=1, k=k, largest=largest)
    spec = f"C64-k32 normal dim=1 largest={largest} dtype=bf16"
    _assert_values_exact(x, torch.bfloat16, 1, k, largest, values, indices, spec)


@pytest.mark.parametrize("largest", [True, False], ids=["largest", "smallest"])
def test_topk_values_padded_rows(device, largest):
    """Non-tile-aligned H (30 rows): the padded rows are sliced away and must not
    perturb the 30 real rows' selections (padding is poisoned to win if read)."""
    shape, k = (1, 1, 30, 64), 32
    x = _topk_input("wide_uniform", shape, seed=11).to(torch.bfloat16)
    values, indices = _run_case(device, x, ttnn.bfloat16, dim=-1, k=k, largest=largest)
    spec = f"H30-k32 wide_uniform dim=-1 largest={largest} dtype=bf16"
    _assert_values_exact(x, torch.bfloat16, -1, k, largest, values, indices, spec)
