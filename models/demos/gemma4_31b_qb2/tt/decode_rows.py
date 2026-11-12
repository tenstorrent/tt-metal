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


# Speculation by active requests: (max live requests, draft length cap). One request drafts up to MAX_K tokens; up to four
# requests draft 3 each (measured on recorded agent trajectories: K=5 pays for one user, K=3 for four with
# NUMBA_NUM_THREADS=1); beyond four requests a step is a plain decode. Verify traces have users x (1 + K) rows.
VERIFY_K_BY_USERS = ((1, 5), (4, 3))


def verify_shapes(slots, requested_k):
    """(users, draft length) of each verify trace of a server with `slots` slots and draft length `requested_k`."""
    shapes = []
    for users, k in VERIFY_K_BY_USERS:
        users, k = min(users, slots), min(k, requested_k)
        if k >= 1 and users * (1 + k) <= 32 and (not shapes or users > shapes[-1][0]):
            shapes.append((users, k))
    return tuple(shapes)


def draft_cap(live_rows, shapes):
    """Most drafts per request on a step with `live_rows` live requests (0: a plain decode)."""
    return next((k for users, k in shapes if live_rows <= users), 0)


def verify_users(live_rows, shapes):
    """The users of the smallest verify trace that covers `live_rows`, or None when no verify trace does."""
    return next((users for users, _ in shapes if live_rows <= users), None)
