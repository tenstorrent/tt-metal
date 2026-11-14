# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure host helpers for the spec-step row buckets (no ttnn import).

A bucket is a row count b in {1, 2, 4, 8} for the verify/drafter traces. A step that runs only engine rows 0..b-1 can replay the
b-row trace instead of the 8-row one. The engine width B (8) must always be a bucket, because it is the fallback for any step.
QWEN36_SPEC_BUCKETS (default "1,2,4,8" = on) lists the buckets; "8" means off: one 8-row bucket.

Engine rows and runner slots: ``phys[slot]`` is the engine row of runner slot ``slot`` (a permutation). A new request should land
in the lowest free engine row, so the live rows stay low and the small buckets stay usable.
"""

VALID_BUCKET_SIZES = (1, 2, 4, 8)


def parse_buckets(text, B=8, clip=False):
    """Parse a comma list like "1,2,4,8" into a sorted tuple of unique ints.

    An empty or blank text gives ``(B,)``. Every value must be in {1, 2, 4, 8} and <= B, and B must be present (the largest).
    Raises ValueError otherwise. With ``clip=True`` a value larger than B is dropped instead of an error, and B is always
    added (so a flag meant for B=8 works on a narrower engine); bad values (not an integer, not in {1, 2, 4, 8}) still raise.
    """
    text = "" if text is None else str(text)
    if not text.strip():
        return (B,)
    vals = set()
    for part in text.split(","):
        part = part.strip()
        try:
            v = int(part)
        except ValueError:
            raise ValueError(f"QWEN36_SPEC_BUCKETS: {part!r} is not an integer (text {text!r})") from None
        if v not in VALID_BUCKET_SIZES:
            raise ValueError(f"QWEN36_SPEC_BUCKETS: {v} is not in {VALID_BUCKET_SIZES} (text {text!r})")
        if v > B and clip:
            continue
        if v > B:
            raise ValueError(f"QWEN36_SPEC_BUCKETS: {v} is larger than the engine width B={B} (text {text!r})")
        vals.add(v)
    if clip:
        vals.add(B)
    if B not in vals:
        raise ValueError(f"QWEN36_SPEC_BUCKETS: must contain the engine width B={B} (text {text!r})")
    return tuple(sorted(vals))


def pick_bucket(buckets, live_rows):
    """Smallest bucket b with b > max(live_rows) (b >= max + 1).

    ``live_rows``: engine row indices that run in this step. An empty set gives the smallest bucket. Raises ValueError if
    max(live_rows) >= max(buckets) (the engine width B), or if a row is negative.
    """
    bs = sorted(buckets)
    if not bs:
        raise ValueError("pick_bucket: no buckets")
    rows = [int(r) for r in live_rows]
    if not rows:
        return bs[0]
    top = max(rows)
    if min(rows) < 0:
        raise ValueError(f"pick_bucket: negative row {min(rows)}")
    if top >= bs[-1]:
        raise ValueError(f"pick_bucket: row {top} is outside the engine width {bs[-1]}")
    for b in bs:
        if b > top:
            return b
    raise AssertionError("unreachable")  # pragma: no cover


def lowest_free_row(free_rows):
    """Smallest engine row in ``free_rows`` (any iterable), or None when it is empty."""
    rows = list(free_rows)
    return min(rows) if rows else None


def plan_row_swap(slot, free_rows, phys):
    """Decide the slot-map swap that puts a NEW request on the lowest free engine row.

    ``slot``: the runner slot the plugin chose for the new request (its engine row is ``phys[slot]``).
    ``free_rows``: the engine rows with no live request (the row of ``slot`` is normally among them).
    ``phys``: the current host slot map, ``phys[runner_slot] = engine_row`` (a permutation). It is not changed.

    Returns ``(slot, other)``: swap ``phys[slot]`` and ``phys[other]``, where ``phys[other]`` is the lowest free row. Returns None
    (no change) when:
      * ``free_rows`` is empty, or
      * the row of ``slot`` is not free (a live request or a re-prefill: a live row never moves), or
      * the row of ``slot`` already is the lowest free row.
    The swap only exchanges two FREE engine rows between two runner slots, so it never moves an occupied row. ``other`` is the
    slot that held the lowest free row; it is empty by construction. Raises ValueError for a bad slot or a free row not in phys.
    """
    if not 0 <= slot < len(phys):
        raise ValueError(f"plan_row_swap: slot {slot} is outside 0..{len(phys) - 1}")
    free = set(int(r) for r in free_rows)
    if not free:
        return None
    if not free <= set(phys):
        raise ValueError(f"plan_row_swap: free rows {sorted(free - set(phys))} are not in the slot map")
    cur = phys[slot]
    if cur not in free:
        return None
    low = min(free)
    if cur == low:
        return None
    return (slot, phys.index(low))


def apply_row_swap(phys, swap):
    """Return a new slot map with ``swap`` (from plan_row_swap) applied; ``phys`` unchanged. None gives a copy."""
    out = list(phys)
    if swap is not None:
        a, b = swap
        out[a], out[b] = out[b], out[a]
    return out


# --------------------------------------------------------------------------------------------------------------------
# In-trace hand-off primitives (ttnn; imported lazily so the helpers above stay pure host).
#
# A bucket trace with b < B rows ends by copying its b-row outputs into rows 0..b-1 of the SHARED B-row buffers the accept /
# chain / host read. Rows b..B-1 of the shared buffer keep their previous content. Each primitive is pure data movement (no
# arithmetic on the source), so a NaN in the TILE padding rows of the source cannot reach the destination.
# tests/test_spec_bucket_handoff.py proves which candidates are exact and trace-safe.
# --------------------------------------------------------------------------------------------------------------------
def handoff_rows_slice_write(src, dst, b):
    """TILE rows: ttnn.experimental.slice_write of src [1,1,b,d] into dst [1,1,B,d] rows 0..b-1."""
    import ttnn

    d = dst.shape[-1]
    ttnn.experimental.slice_write(src, dst, [0, 0, 0, 0], [1, 1, b, d], [1, 1, 1, 1])


def handoff_rows_concat(src, dst, b):
    """TILE rows: concat(src, dst rows b..B-1) then copy into dst (same shape, same address). ``b`` == every row of dst
    (a full-width split variant, QWEN36_ATTN_KV_SPLIT gate): a plain copy (nothing of dst survives)."""
    import ttnn

    B, d = dst.shape[-2], dst.shape[-1]
    if int(b) >= int(B):
        ttnn.copy(src, dst)
        return
    rest = ttnn.slice(dst, [0, 0, b, 0], [1, 1, B, d])
    cat = ttnn.concat([src, rest], dim=2)
    ttnn.copy(cat, dst)
    ttnn.deallocate(rest)
    ttnn.deallocate(cat)


def handoff_ids_slice_write(src, dst, b):
    """u32 ROW_MAJOR ids: slice_write of src [1,1,b,1] into dst [1,1,B,1] rows 0..b-1."""
    import ttnn

    ttnn.experimental.slice_write(src, dst, [0, 0, 0, 0], [1, 1, b, 1], [1, 1, 1, 1])


def handoff_ids_concat(src, dst, b):
    """u32 ROW_MAJOR ids: concat(src, dst rows b..B-1) then copy into dst. ``b`` == B (a full-width split variant): a copy."""
    import ttnn

    B = dst.shape[-2]
    if int(b) >= int(B):
        ttnn.copy(src, dst)
        return
    rest = ttnn.slice(dst, [0, 0, b, 0], [1, 1, B, 1])
    cat = ttnn.concat([src, rest], dim=2)
    ttnn.copy(cat, dst)
    ttnn.deallocate(rest)
    ttnn.deallocate(cat)
