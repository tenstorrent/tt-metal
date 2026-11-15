# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Pure (no ttnn) arithmetic of the paged-bucket prefill (prefill_paged.py): row buckets, start classes, the trace
retire key and the warmed key sets. bench/packaging/DRAM_PREFILL_FIX_PLAN_v2.md §3.2 + v2 ADDENDUM_1 (a); kept
import-free so host tests run without touching a device.

A prefill row of length L from scheduler start s computes rows [a, L) with a = s // 128 * 128 (the page-aligned
rewind). PB pads the chunk n = L - a to n128 = ceil(n / 128) * 128 rows, so every program sees whole pages. A
program-affecting value not fixed by n128 is either pre-warmed at boot or part of the row key below.
"""

PAGE = 128
CHUNK_LIMIT = 8192  # the vLLM prefill budget (MAX_PREFILL_TOKENS); a chunk also recomputes its <= 127-row rewind
HISTORY_ROWS = 1152  # the sliding tail kept between chunks: window (1,024) + one page (Decoder.SLIDING_WINDOW_PAGES * 128)
MERGE_ROWS = 32  # sample_prefill_outputs merges up to 32 rows (concat + pad to 32)


def pages128(n):
    """Rows the PB chunk computes: the next multiple of 128 (n >= 1)."""
    if n < 1:
        raise ValueError(f"empty prefill chunk: {n}")
    return -(-n // PAGE) * PAGE


def aligned(start):
    return start // PAGE * PAGE


def start_class(start):
    """The full-attention SDPA Q-chunk class of an absolute start: chunks are 32..256 rows (decoder.py
    pref_sdpa_by_q), starts are multiples of 128, so eligibility depends only on start % 256. 0 for a first chunk."""
    if start % PAGE:
        raise ValueError(f"absolute start {start} is not page aligned")
    return 0 if start == 0 else (256 if start % 256 == 0 else 128)


def history_rows(aligned_start):
    """Sliding history a continuation from `aligned_start` receives: the last min(aligned_start, 1152) rows."""
    return min(aligned_start, HISTORY_ROWS)


def row_key(length, start, *, flex=True):
    """The retire key of one prefill row (length L from scheduler start s). Every program the row creates is a
    function of this tuple (plus the pre-warmed families). A continuation's stored sliding tail ends at the previous
    chunk's padded end, so an unaligned previous end (s % 128 != 0; never under the plugin's 128 alignment) trims one
    page: its own key. Under the F2-b fallback (flex=False) the absolute start itself enters the key."""
    a = aligned(start)
    b = pages128(length - a)
    if a == 0:
        return ("first", b)
    return ("cont", b, start_class(a), history_rows(a), int(start != a), None if flex else a)


def call_keys(lengths, starts, *, flex=True):
    """Keys of one prefill call: one per row, plus the row count (the sampling merge's concat / pad)."""
    if len(lengths) != len(starts) or not lengths:
        raise ValueError("one start per prefill row")
    keys = {row_key(n, s, flex=flex) for n, s in zip(lengths, starts)}
    keys.add(("rows", len(lengths)))
    return keys


def first_buckets(limit=CHUNK_LIMIT):
    """First-chunk buckets the boot warmup builds: 128 .. limit (64 at 8,192). A first chunk is at most `limit`
    rows (no rewind at start 0)."""
    return list(range(PAGE, limit + 1, PAGE))


def warmed_keys(max_rows, limit=CHUNK_LIMIT):
    """The keys a completed boot warmup covers: every first-chunk bucket and row counts 1..max_rows."""
    return {("first", b) for b in first_buckets(limit)} | {("rows", k) for k in range(1, max_rows + 1)}


def terminal_slices(b):
    """Terminal tile starts for a bucket b: the last real row n - 1 lies in [b - 128, b), so its 32-row tile starts
    at one of four offsets."""
    return [b - 128 + 32 * i for i in range(4)]


def tap_windows(b, seed_rows):
    """Drafter tap windows (lo, hi) for a bucket b: hi = ceil32(n) for n in (b - 128, b], lo = max(0, hi - seed_rows)."""
    return [(max(0, hi - seed_rows), hi) for hi in (b - 96, b - 64, b - 32, b)]


def continuation_starts():
    """One aligned start per distinct (start class, history rows) pair: 128 .. 1,152 (8 history lengths below 1,152;
    the class follows from start % 256) and 1,280 (history 1,152, class 256); 1,152 itself is class 128."""
    return list(range(PAGE, HISTORY_ROWS + PAGE + 1, PAGE))


def continuation_key_space(limit=CHUNK_LIMIT, *, flex=True, unaligned=False):
    """Every continuation key reachable (flex): buckets up to limit + 128 (a chunk that recomputes its <= 127-row
    rewind). unaligned=False: the plugin's 128 alignment (every continuation starts on the grid); True: also the
    unaligned-previous-end keys."""
    keys = set()
    for a in continuation_starts():
        for b in range(PAGE, limit + PAGE + 1, PAGE):
            for r in (0, 1) if unaligned else (0,):
                keys.add(("cont", b, start_class(a), history_rows(a), r, None if flex else a))
    return keys
