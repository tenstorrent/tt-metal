# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The narrow verify head's window arithmetic — no device, no weights, no mesh.

``TtTarget.forward`` returns ``logits[:, -S:]`` with S <= block_size (16), but the LM head and its
vocab all-gather run over the whole ``ANCHOR``-row bucket (128). That is up to 8x of the tail matmul,
of the replicating all-gather, and of the 26.8 ms readback
(tests/perf/test_traced_verify_host_breakdown.py) spent on rows that are thrown away.

Running the head on exactly the wanted rows was considered and REJECTED, in a comment at
``Qwen36Model.prefill_block_all_logits``: *"A pre-head slice would need tile alignment and would
recompile per length."* Both objections are correct and both are about cutting the window EXACTLY.
``Qwen36Model.lm_head_window`` snaps it out to tile boundaries instead, which answers them:

  * the start is always a multiple of 32, so no slice ever straddles a tile row;
  * with keep_rows <= 16 the window spans at most two tile rows, so its WIDTH is only ever 32 or 64
    -- TWO program shapes for the whole loop, not one per length. That bound is what keeps the
    narrow head from re-introducing the compile-under-a-parked-trace hang (DFLASH_HANDOFF.md §0),
    and it is the property this file pins.

This is pure integer arithmetic, so it is tested exhaustively and without hardware. If it ever
yields a third width, the pre-capture warm-up in ``capture_verify_trace`` (which compiles exactly
{32, 64}) silently stops covering the loop, and the next narrow-head run hangs instead of failing.
Hence the assertion on the width SET, not just on correctness.
"""

from __future__ import annotations

import pytest

from models.demos.blackhole.qwen36.tt.model import Qwen36Model

ANCHOR = 128
BLOCK = 16
TILE = 32

window = Qwen36Model.lm_head_window


@pytest.mark.parametrize("bucket", [128, 64])
def test_window_is_tile_aligned_and_contains_the_wanted_rows(bucket):
    """For every (valid_len, keep_rows) the loop can produce: aligned, in bounds, and correct."""
    checked = 0
    for valid_len in range(1, bucket + 1):
        for keep in range(1, min(BLOCK, valid_len) + 1):
            start, width = window(valid_len, keep, bucket)
            assert start % TILE == 0, f"start {start} not tile-aligned (valid_len={valid_len}, keep={keep})"
            assert start + width <= bucket, f"window [{start}, {start + width}) past bucket {bucket}"
            # The caller slices the returned window's rows [lo, lo+keep) on host.
            lo = (valid_len - keep) - start
            assert 0 <= lo and lo + keep <= width, f"wanted rows fall outside the window ({lo}, {keep}, {width})"
            # Those rows must be exactly the absolute rows [valid_len-keep, valid_len).
            got = [start + lo + i for i in range(keep)]
            assert got == list(range(valid_len - keep, valid_len)), f"wrong absolute rows for {valid_len}/{keep}"
            checked += 1
    assert checked > 0


def test_only_two_head_widths_exist():
    """THE LOAD-BEARING PROPERTY: the pre-capture warm-up compiles {32, 64} and nothing else.

    A third width would be a program first compiled with the verify trace parked, which does not
    raise -- it hangs the process at ~110 % CPU and wedges the device.
    """
    widths = {window(v, k, ANCHOR)[1] for v in range(1, ANCHOR + 1) for k in range(1, min(BLOCK, v) + 1)}
    assert widths == {TILE, 2 * TILE}, f"expected only 32/64-row heads, got {sorted(widths)}"


def test_the_window_actually_narrows_the_common_case():
    """A full bucket with a full block -- the steady-state verify -- must shrink 128 -> 32."""
    start, width = window(ANCHOR, BLOCK, ANCHOR)
    assert (start, width) == (96, 32)
    assert width * 4 == ANCHOR, "the steady-state head should be 4x narrower"


def test_keep_rows_must_fit_the_valid_span(expect_error):
    with expect_error(AssertionError, "keep_rows"):
        window(8, 16, ANCHOR)  # cannot want more rows than exist
    with expect_error(AssertionError, "keep_rows"):
        window(16, 0, ANCHOR)  # zero is the caller's "skip the head" signal, not a window
