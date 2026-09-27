# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Direct probe of the five cell regions parked in `feature_spec.INVALID`.

WHY THIS FILE EXISTS. `eval/golden_tests/rms_norm_ttnn/feature_spec.py`'s `INVALID`
list ends with five entries under a heading that says what they are:

    # --- Author-scoped exclusions ("for now", NOT structural impossibility) ---
    # Unlike the entries above (truly universe-would-have-to-change), these COULD
    # eventually be supported -- they are deliberately parked in INVALID to keep
    # them out of the refinement backlog (harness skips them instead of xfailing)

    {layout: ROW_MAJOR, memory_layout: HEIGHT_SHARDED, gamma_layout: TILE}
    {layout: ROW_MAJOR, memory_layout: WIDTH_SHARDED,  gamma_layout: TILE}
    {layout: ROW_MAJOR, memory_layout: BLOCK_SHARDED,  gamma_layout: TILE}
    {dtype: bfloat8_b, alignment: w_non_aligned}
    {dtype: bfloat8_b, alignment: h_non_aligned}

An `INVALID` cell is `pytest.mark.skip`ped, so the golden suite never calls the op
for any of them and neither a pass nor a failure is observable there. But the op's
`SUPPORTED` **claims every one of those axis values**, `validate()` accepts them,
and `eval/prompts/rms_norm_ttnn.txt`'s Phase-0 list requires them ("dtype:
float32, bfloat16, bfloat8_b", "alignment: tile_aligned, w_non_aligned,
h_non_aligned", "layout: TILE and ROW_MAJOR, both native", "at every memory
placement the op accepts"). So the five entries are exactly the case the registry
model routes to `EXCLUSIONS`, not to `INVALID` — and while they sit in `INVALID`
the claim goes untested.

This file closes that hole from the op side: it runs the op on all five regions and
PCC-gates the result. It is the evidence behind `verification_report.md`'s
INVALID-audit recommendation, and it is a standing regression test for five cell
regions the golden suite structurally cannot reach.

Three of the five additionally couple axes describing DIFFERENT TENSORS — the
activation's `layout` and `memory_layout` against the *weight*'s `gamma_layout` —
which is the canonical `INVALID` authoring mistake the registry model calls out by
name. The activation's placement does not constrain what layout a per-channel
vector may arrive in, and this file demonstrates it does not.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from eval.sharding import auto_shard_config
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

EPSILON = 1e-5

#: The golden suite's bands (eval/golden_tests/rms_norm_ttnn/helpers.py:TOLERANCES).
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def _weight(device, W, dtype, layout):
    torch.manual_seed(1)
    values = torch.randn(W, dtype=torch.float32)
    ttnn_w = ttnn.from_torch(
        values.reshape(1, 1, 1, W).to(torch.bfloat16 if dtype != ttnn.float32 else torch.float32),
        dtype=dtype,
        layout=layout,
        device=device,
    )
    return values, ttnn_w


# ---------------------------------------------------------------------------
# Region A — ROW_MAJOR activation, sharded placement, TILE-layout weight
# ---------------------------------------------------------------------------
#
# The three cross-tensor INVALID entries. A ROW_MAJOR activation on a *_SHARDED
# placement takes the BAND scheme (WIDTH/BLOCK) or the native height shard
# (HEIGHT); the weight's layout is read by an entirely separate reader branch, so
# a TILE weight there is the ordinary mixed-form case, not a new scheme.


@pytest.mark.parametrize(
    "memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("shape", [(1, 1, 256, 512), (1, 1, 64, 128)], ids=lambda s: "x".join(map(str, s)))
def test_row_major_input_sharded_with_tile_weight(device, shape, memory_layout):
    """INVALID entries 1-3: `{layout: RM, memory_layout: *_SHARDED, gamma_layout: TILE}`."""
    W = shape[-1]
    dtype = ttnn.bfloat16
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)

    mem_cfg = auto_shard_config(shape, memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_cfg)
    torch_w, ttnn_w = _weight(device, W, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    got = ttnn.to_torch(rms_norm_ttnn(ttnn_x, epsilon=EPSILON, weight=ttnn_w))
    expected = torch_rms_norm_ttnn(x, epsilon=EPSILON, weight=torch_w)

    assert got.shape == expected.shape
    assert_with_pcc(expected.to(torch.float32), got.to(torch.float32), PCC[dtype])


# ---------------------------------------------------------------------------
# Region B — bfloat8_b on a non-tile-aligned shape
# ---------------------------------------------------------------------------
#
# The two alignment INVALID entries. bf8b is a block format so it is TILE-only
# (that part IS structural, and the `{bf8b, ROW_MAJOR}` entry above it is correct);
# what these two entries add is that a bf8b TILE tensor may not have a
# non-multiple-of-32 H or W. Nothing in the format says that: the tile grid is
# padded either way and the reduce's partial-W mask is dtype-independent.


@pytest.mark.parametrize(
    "shape, bucket",
    [
        ((1, 1, 32, 50), "w_non_aligned"),
        ((1, 1, 64, 17), "w_non_aligned"),
        ((1, 1, 47, 64), "h_non_aligned"),
        ((1, 1, 17, 128), "h_non_aligned"),
        ((1, 1, 47, 50), "w_non_aligned"),  # BOTH non-aligned; w dominates the tag
    ],
    ids=lambda v: v if isinstance(v, str) else "x".join(map(str, v)),
)
@pytest.mark.parametrize("gamma_mode", ["no_gamma", "gamma"])
def test_bfloat8_b_non_tile_aligned(device, shape, bucket, gamma_mode):
    """INVALID entries 4-5: `{dtype: bfloat8_b, alignment: w/h_non_aligned}`.

    The RMS denominator must reflect only VALID elements along the reduced
    dimension, so the padding lanes of a non-aligned W must not enter the sum.
    The reference divides by the LOGICAL width, which is what makes this a real
    check and not a shape assertion.
    """
    dtype = ttnn.bfloat8_b
    W = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)

    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {}
    torch_w = None
    if gamma_mode == "gamma":
        torch_w, ttnn_w = _weight(device, W, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kwargs["weight"] = ttnn_w

    got = ttnn.to_torch(rms_norm_ttnn(ttnn_x, epsilon=EPSILON, **kwargs))
    # bf8b quantizes on the way in, so the reference must see the SAME values the
    # device did — read the input back rather than comparing against the pre-cast
    # torch tensor, or the measurement folds the input quantization into the op's
    # error budget.
    x_device = ttnn.to_torch(ttnn_x).to(torch.float32)
    expected = torch_rms_norm_ttnn(x_device, epsilon=EPSILON, weight=torch_w)

    assert got.shape == expected.shape
    assert_with_pcc(expected.to(torch.float32), got.to(torch.float32), PCC[dtype])
