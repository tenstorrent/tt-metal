# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Row counts of the decode traces and the choice of one for a step.

A decode trace has a fixed number of rows. Rows without a request still cost device time (the full-attention SDPA of
one live row is 8-10x slower in a 32-row trace than in a 1-row trace at 28K context), so a server keeps traces
for a few row counts and each step runs the smallest one that covers its live rows. The vLLM batch is front-packed
(live requests occupy rows 0..n-1), so the live row count is the index of the last live row plus one.
"""

ROW_SIZES = (1, 4, 8, 16, 32)


def decode_row_sizes(slots):
    """Row counts of the extra decode traces for a server with `slots` slots (the full trace is always present)."""
    return tuple(n for n in ROW_SIZES if n < slots)


def rows_covering(live_rows, sizes, full_rows):
    """The smallest trace of `sizes` (or the full one) with at least `live_rows` rows."""
    return next((n for n in sizes if n >= live_rows), full_rows)
