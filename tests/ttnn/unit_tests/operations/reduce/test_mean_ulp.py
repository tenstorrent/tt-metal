# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ULP-based accuracy characterization for ttnn.mean (issue #33740).

Relationship to existing mean tests (do not duplicate; different goal):

- ``test_reduction.py`` (e.g. ``test_mean_4d_tensor_dims``, ``test_mean_3d_tensor_dims``,
  ``test_mean_2d_tensor_dims``): equivalence vs torch via PCC / rtol / atol on modest
  shapes and many ``dim`` / ``keepdim`` combinations.
- ``test_reduction_mean.py``: batched 3D means (``dim`` -1 / -2) with tile padding
  stress, ``scalar=`` scaling, and sharded DRAM/block configs—still PCC-style metrics.
- ``tests/ttnn/nightly/unit_tests/operations/reduction/test_reduction_ops.py``:
  reduction corner cases (empty tensors, preallocated outputs, etc.) with ``op`` including
  ``mean``—not large numeric sweeps.

This file is only for **bounded max-ULP** characterization (plus near-zero absolute
tolerance) and **wide shape sweeps** to track regression in reduction length; it does
not replace functional, sharding, or corner-case coverage above.

BF16 golden: torch.mean(bf16_input.float(), dim=...).to(torch.bfloat16)
  - Accumulates in FP32 on the host, then casts the result to BF16.
  - This matches the "best practice" (FP32 accumulation) that the device
    path should also follow.

FP32 golden: torch.mean(fp32_input, dim=...)
  - PyTorch FP32 accumulation (same precision as device).
  - Both device and PyTorch use FP32 arithmetic; differences arise from
    operation ordering (tile-based vs sequential).

ULP is measured in the output dtype (BF16 or FP32).  Elements where
|golden| is very small relative to the tensor's dynamic range are excluded
from ULP (where the metric breaks down due to division by a tiny ULP
quantum) and validated with a scaled absolute-tolerance check instead.

Metrics are logged at INFO for every parametrized case (pass or fail).  To
print them in the terminal, run pytest with e.g.:

  pytest .../test_mean_ulp.py --log-cli-level=INFO

or capture to a file:

  pytest .../test_mean_ulp.py --log-file=mean_ulp.log --log-file-level=INFO
"""

import logging

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import measure_ulp_with_near_zero_atol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _golden_mean_bf16(input_bf16: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """FP32-accumulated mean, result cast back to BF16."""
    return torch.mean(input_bf16.float(), dim=dim, keepdim=keepdim).to(torch.bfloat16)


def _golden_mean_fp32(input_fp32: torch.Tensor, dim, keepdim: bool) -> torch.Tensor:
    """PyTorch FP32 mean — same accumulation width as device."""
    return torch.mean(input_fp32, dim=dim, keepdim=keepdim)


def _run_ttnn_mean(input_torch: torch.Tensor, ttnn_dtype, device, dim, keepdim: bool) -> torch.Tensor:
    """Send tensor to device, run ttnn.mean, return host torch tensor."""
    tt_input = ttnn.from_torch(input_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(tt_input, 42)
    tt_out = ttnn.mean(tt_input, dim=dim, keepdim=keepdim)
    return ttnn.to_torch(tt_out)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

_SHAPES_AND_DIMS = [
    # Tile-axis reductions: W (-1), H (-2), HW ([-2,-1]).
    ((1, 1, 32, 256), -1, "W-256"),
    ((1, 1, 32, 512), -1, "W-512"),
    ((1, 1, 32, 1024), -1, "W-1024"),
    ((1, 1, 32, 2048), -1, "W-2048"),
    ((1, 1, 32, 4096), -1, "W-4096"),
    ((1, 1, 32, 8192), -1, "W-8192"),
    ((1, 1, 128, 32), -2, "H-128"),
    ((1, 1, 512, 32), -2, "H-512"),
    ((1, 1, 1024, 32), -2, "H-1024"),
    ((1, 1, 2048, 32), -2, "H-2048"),
    ((1, 1, 4096, 32), -2, "H-4096"),
    ((1, 1, 64, 64), [-2, -1], "HW-64x64"),
    ((1, 1, 96, 160), [-2, -1], "HW-96x160"),
    ((1, 1, 128, 128), [-2, -1], "HW-128x128"),
    ((1, 1, 256, 256), [-2, -1], "HW-256x256"),
    ((1, 1, 32, 32768), -1, "W-32768"),
    ((1, 1, 37, 41), -1, "W-odd-41"),
    ((1, 1, 37, 41), -2, "H-odd-37"),
    ((2, 1, 48, 64), -1, "W-2batch-48x64"),
    # Non-HW-axis reductions: batch (0), channel (1), multi-dim ([0,1]).
    # These go through transpose + sequential reduce (reduce_nd_loop).
    ((4, 3, 32, 32), 0, "batch-4"),
    ((8, 3, 32, 32), 0, "batch-8"),
    ((4, 3, 32, 32), 1, "channel-3"),
    ((2, 5, 64, 64), 1, "channel-5"),
    ((4, 3, 32, 32), [0, 1], "batch+channel"),
]


# ---------------------------------------------------------------------------
# BF16 tests
# ---------------------------------------------------------------------------

# BF16 ULP threshold: 128 is one order of magnitude (2^7 mantissa bits).
# With FP32 accumulation (the default), measured max ULP is 20 for normal
# distribution across all reduction patterns.
_BF16_ULP_THRESHOLD = 30
_BF16_NEAR_ZERO_ATOL_FRACTION = 0.02


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
def test_mean_ulp_bf16(device, shape, dim, desc, distribution):
    """Characterize BF16 mean ULP vs FP32-accumulated Torch golden."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3).to(torch.bfloat16)

    golden = _golden_mean_bf16(x, dim=dim, keepdim=True)
    actual = _run_ttnn_mean(x, ttnn.bfloat16, device, dim=dim, keepdim=True)

    passed, max_ulp, max_atol_err, msg = measure_ulp_with_near_zero_atol(
        golden, actual, _BF16_ULP_THRESHOLD, _BF16_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.mean ULP dtype=BF16 desc=%r distribution=%r shape=%s dim=%s "
        "max_ulp=%s ulp_threshold=%s max_atol_err=%s passed=%s | %s",
        desc,
        distribution,
        shape,
        dim,
        max_ulp,
        _BF16_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[BF16 {desc} {distribution}] {msg}"


# ---------------------------------------------------------------------------
# FP32 tests
# ---------------------------------------------------------------------------

# FP32: tile vs sequential ordering; BH sweeps topped ~5.1e5 ULP (batch+channel).
_FP32_ULP_THRESHOLD = 700_000
_FP32_NEAR_ZERO_ATOL_FRACTION = 0.005


@pytest.mark.parametrize(
    "shape, dim, desc",
    _SHAPES_AND_DIMS,
    ids=[c[2] for c in _SHAPES_AND_DIMS],
)
@pytest.mark.parametrize("distribution", ["normal", "wide_uniform"])
def test_mean_ulp_fp32(device, shape, dim, desc, distribution):
    """Characterize FP32 mean ULP vs Torch FP32 golden."""
    torch.manual_seed(42)
    if distribution == "normal":
        x = torch.randn(shape, dtype=torch.float32)
    else:
        x = torch.empty(shape, dtype=torch.float32).uniform_(-1e3, 1e3)

    golden = _golden_mean_fp32(x, dim=dim, keepdim=True)
    actual = _run_ttnn_mean(x, ttnn.float32, device, dim=dim, keepdim=True)

    passed, max_ulp, max_atol_err, msg = measure_ulp_with_near_zero_atol(
        golden, actual, _FP32_ULP_THRESHOLD, _FP32_NEAR_ZERO_ATOL_FRACTION
    )
    logger.info(
        "ttnn.mean ULP dtype=FP32 desc=%r distribution=%r shape=%s dim=%s "
        "max_ulp=%s ulp_threshold=%s max_atol_err=%s passed=%s | %s",
        desc,
        distribution,
        shape,
        dim,
        max_ulp,
        _FP32_ULP_THRESHOLD,
        max_atol_err,
        passed,
        msg,
    )
    assert passed, f"[FP32 {desc} {distribution}] {msg}"
