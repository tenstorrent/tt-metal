# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The per-lane STATE tensor checks of the fused wrappers take a lane extent >= rows (a 32-lane parent state serves a
trace of R < 32 lanes), with every other dim, the dtype and the layout still exact.  No device: fake tensors; a check
that passes is seen by the later, device-side step (a sentinel) being reached."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step, ple
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

BF16, FP32 = ttnn.bfloat16, ttnn.float32
ROWS = 8


class Reached(Exception):
    """Raised by a fake's ``device()``: the checks before it passed."""


def fake(shape, dtype=BF16, layout=ttnn.TILE_LAYOUT):
    def device():
        raise Reached

    return SimpleNamespace(
        shape=tuple(shape),
        padded_shape=(*shape[:-2], 32, shape[-1]) if len(shape) == 4 and shape[:2] == (1, 1) else tuple(shape),
        dtype=dtype,
        layout=layout,
        device=device,
    )


def row_tile(width, rows=ROWS):
    return fake((1, 1, rows, width))


EXTENTS = [(ROWS, True), (32, True), (ROWS - 1, False)]


@pytest.mark.parametrize("extent,ok", EXTENTS)
def test_gdn_step_run_recurrent_lane_extent(extent, ok):
    recurrent = fake((extent, gdn_step.HEADS, gdn_step.HEAD_DIM, gdn_step.HEAD_DIM), FP32, ttnn.TILE_LAYOUT)
    projected = row_tile(gdn_step.PROJECTION_WIDTH)
    args = (projected, [], None, [], None, None, recurrent, None)
    if ok:  # past the state check the next one (three older slots) fires
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError, match="three older ring slots"
        ):
            gdn_step.run(*args)
    else:
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError, match=r"state must be \[>= 8,"
        ):
            gdn_step.run(*args)


def test_gdn_step_run_recurrent_other_dims_exact():
    projected = row_tile(gdn_step.PROJECTION_WIDTH)
    for shape, dtype in (
        ((32, gdn_step.HEADS + 1, gdn_step.HEAD_DIM, gdn_step.HEAD_DIM), FP32),
        ((32, gdn_step.HEADS, gdn_step.HEAD_DIM, gdn_step.HEAD_DIM), BF16),
        ((32, gdn_step.HEADS, gdn_step.HEAD_DIM), FP32),
    ):
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError, match="state must be"
        ):
            gdn_step.run(projected, [], None, [], None, None, fake(shape, dtype), None)


@pytest.mark.parametrize("extent,ok", EXTENTS)
def test_gdn_step_admits_recurrent_lane_extent(extent, ok):
    projected = row_tile(gdn_step.PROJECTION_WIDTH)
    recurrent = fake((extent, gdn_step.HEADS, gdn_step.HEAD_DIM, gdn_step.HEAD_DIM), FP32)
    state = SimpleNamespace(recurrent=recurrent)
    assert gdn_step.admits(None, projected, [0] * 4, state) is ok


# index_tail / main_tail reach their ring and staging checks only after allocating on a device, so the shared helper
# they call is tested directly
@pytest.mark.parametrize("extent,ok", EXTENTS)
@pytest.mark.parametrize(
    "label,tail", [("raw-key ring", (fp.TILE, qsa_block.INDEX_HEAD_DIM)), ("KV staging", (fp.TILE, qsa_block.KV_WIDTH))]
)
def test_qsa_lane_state_helper(extent, ok, label, tail):
    tensor = fake((1, extent, *tail))
    if ok:
        qsa_block._expect_lane_state(tensor, ROWS, tail, BF16, ttnn.TILE_LAYOUT, label)
    else:
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError, match=rf"{label} must be .*>= 8"
        ):
            qsa_block._expect_lane_state(tensor, ROWS, tail, BF16, ttnn.TILE_LAYOUT, label)


def test_qsa_lane_state_helper_other_dims_exact():
    tail = (fp.TILE, qsa_block.KV_WIDTH)
    for tensor in (
        fake((1, 32, fp.TILE, qsa_block.KV_WIDTH + 32)),
        fake((2, 32, *tail)),
        fake((1, 32, *tail), FP32),
        fake((1, 32, *tail), BF16, ttnn.ROW_MAJOR_LAYOUT),
        fake((1, 32, tail[1])),
    ):
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError
        ):
            qsa_block._expect_lane_state(tensor, ROWS, tail, BF16, ttnn.TILE_LAYOUT, "x")


@pytest.mark.parametrize("extent,ok", EXTENTS)
def test_lane_score_rows_width(extent, ok):
    cache_rows, blocks = 64, 32
    scores = fake((1, 1, 32, extent * cache_rows), BF16, ttnn.ROW_MAJOR_LAYOUT)
    if ok:
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            Reached
        ):
            qsa_block.lane_score_rows(scores, ROWS, cache_rows, blocks)
    else:
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError, match="lane scores must be"
        ):
            qsa_block.lane_score_rows(scores, ROWS, cache_rows, blocks)


@pytest.mark.parametrize("extent,ok", EXTENTS)
def test_ple_expect_lane_state(extent, ok):
    tensor = fake((1, extent, ple.BRANCHES, ple.LOCAL_HIDDEN))
    if ok:
        ple._expect_lane_state(tensor, ROWS, ple.LOCAL_HIDDEN, BF16, "state row 0")
    else:
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError, match="state row 0 must be"
        ):
            ple._expect_lane_state(tensor, ROWS, ple.LOCAL_HIDDEN, BF16, "state row 0")


def test_ple_expect_lane_state_other_dims_exact_and_lanes_keep_equality():
    for tensor in (
        fake((1, 32, ple.BRANCHES, ple.LOCAL_HIDDEN + 32)),
        fake((1, 32, ple.BRANCHES, ple.LOCAL_HIDDEN), FP32),
        fake((1, 32, ple.BRANCHES, ple.LOCAL_HIDDEN), BF16, ttnn.ROW_MAJOR_LAYOUT),
    ):
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError
        ):
            ple._expect_lane_state(tensor, ROWS, ple.LOCAL_HIDDEN, BF16, "x")
    with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
        ValueError
    ):  # normalized / gated keep the exact check
        ple._expect_lanes(fake((1, 32, ple.BRANCHES, ple.LOCAL_HIDDEN)), ROWS, ple.LOCAL_HIDDEN, BF16, "x")


@pytest.mark.parametrize("extent,ok", EXTENTS)
def test_ple_conv_lanes_state_rows(extent, ok):
    state = [fake((1, extent, ple.BRANCHES, ple.LOCAL_HIDDEN)) for _ in range(ple.CONV_STATE_LENGTH)]
    lane = fake((1, ROWS, ple.BRANCHES, ple.LOCAL_HIDDEN))
    taps = [fake((1, 1, ple.BRANCHES, ple.LOCAL_HIDDEN)) for _ in range(ple.CONV_TAPS)]
    if ok:  # past the checks: normalized.device() is the sentinel
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            Reached
        ):
            ple.conv_lanes(state, lane, taps, lane, ROWS)
    else:
        with pytest.raises(  # allow-pytest.raises: no-device Python-level check, no device error text to triage
            ValueError, match="state row 0 must be"
        ):
            ple.conv_lanes(state, lane, taps, lane, ROWS)
