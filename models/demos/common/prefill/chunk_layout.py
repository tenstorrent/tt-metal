# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chip layout of one prefill chunk over the block-cyclic SP KV cache.

The update_padded_kv_cache writer gives SP chip c the chunk_local-token blocks c, c+sp, ... of the global sequence,
so a chunk that starts mid-slab (a multi-turn resume at a tile boundary) is not a contiguous split of its tokens:
the first-rank input must be rotated to match (as the engine's H2D prefill connector does), and per-chip counts of
real rows follow the same placement. A packed forward (several segment-aligned segments in one chunk) is the
un-rotated case repeated per segment; ``packed_segments_real_counts`` gives its per-chip real-row counts.
"""


def rotated_chunk_positions(chunk_start: int, sp: int, chunk_local: int) -> list[list[int]]:
    """Global position of each SP chip's local row for the chunk [chunk_start, chunk_start + sp*chunk_local).

    positions[c][r] is what update_padded_kv_cache writes chip c's r-th input row to: chip c owns the
    chunk_local-token blocks c, c+sp, ... of the global sequence, so a chunk that starts mid-slab rotates
    which chip holds its first block, and the boundary chip's rows run from the tail of one of its blocks
    into the head of its next one. Positions increase with r on every chip. Chunk-aligned starts reduce to
    chunk_start + c*chunk_local + r.
    """
    chunk_global = sp * chunk_local
    slab, boundary_chip, offset = (
        chunk_start // chunk_global,
        (chunk_start // chunk_local) % sp,
        chunk_start % chunk_local,
    )
    positions = []
    for c in range(sp):
        first_row = (slab + (c < boundary_chip)) * chunk_local + (offset if c == boundary_chip else 0)
        rows = range(first_row, first_row + chunk_local)
        positions.append([(lr // chunk_local) * chunk_global + c * chunk_local + lr % chunk_local for lr in rows])
    return positions


def rotate_chunk_tokens(chunk_token_ids: list, chunk_start: int, sp: int) -> list:
    """Reorder one chunk's natural-order tokens into the chip-major order a plain SP shard hands each chip.

    Chip c's shard receives the tokens at rotated_chunk_positions(chunk_start)[c], so the KV writer and the
    indexed RoPE (both keyed off chunk_start) see each token at its true position. This is the host-side
    reshuffle the engine's H2D prefill connector applies (tt-llm-engine ring_sdpa_reshuffle); identity for a
    chunk-aligned chunk_start.
    """
    assert len(chunk_token_ids) % sp == 0, f"chunk of {len(chunk_token_ids)} tokens does not split over sp={sp}"
    chunk_local = len(chunk_token_ids) // sp
    return [
        chunk_token_ids[p - chunk_start] for row in rotated_chunk_positions(chunk_start, sp, chunk_local) for p in row
    ]


def rotated_chunk_real_counts(chunk_start: int, actual_isl: int, sp: int, chunk_local: int) -> list[int]:
    """Per-chip count of real (non-pad) rows of a rotated chunk holding `actual_isl` real tokens.

    Positions increase with the local row, so each chip's real rows are a prefix of its rows (right padding).
    Reduces to min(chunk_local, actual_isl - c*chunk_local) for a chunk-aligned chunk_start.
    """
    end = chunk_start + actual_isl
    return [sum(p < end for p in row) for row in rotated_chunk_positions(chunk_start, sp, chunk_local)]


def pack_segment_rows(segments: list[list], sp: int) -> list[list]:
    """Chip-major order of a packed forward's rows: ``segments`` holds each segment's rows (tokens, or row ids)
    in natural order; chip c receives every segment's c-th ``seg/sp`` slice back to back, which is how
    ``prefill_segments`` slices the segments out of the packed activations again."""
    lengths = {len(s) for s in segments}
    assert len(lengths) == 1, f"segments of different lengths: {sorted(lengths)}"
    (seg,) = lengths
    assert seg % sp == 0, f"segment of {seg} rows does not split over sp={sp}"
    s_local = seg // sp
    return [[r for s in segments for r in s[c * s_local : (c + 1) * s_local]] for c in range(sp)]


def packed_segments_real_counts(n_reals: list[int], sp: int, s_local: int) -> list[int] | None:
    """Per-chip count of leading real rows of a packed forward of segments holding ``n_reals`` real tokens
    each, or None when no chip can skip a row (every row real, or pad rows only ahead of a later segment's
    real rows).

    Chip c's rows are the segments' c-th ``s_local``-row slices back to back. The MoE gate can skip pad rows
    only when they all follow the chip's real rows, so a chip where a segment's pad precedes a later
    segment's real rows counts every row as real (routed, still correct).
    """
    total = len(n_reals) * s_local
    counts = []
    for c in range(sp):
        reals = [max(0, min(s_local, n - c * s_local)) for n in n_reals]
        lead = 0
        for k, r in enumerate(reals):
            lead += r
            if r < s_local:
                if any(reals[k + 1 :]):
                    lead = total
                break
        counts.append(lead)
    return None if all(x == total for x in counts) else counts
