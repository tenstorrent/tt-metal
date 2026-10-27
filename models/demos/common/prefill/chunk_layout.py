# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chip layout of one prefill chunk over the block-cyclic SP KV cache.

The update_padded_kv_cache writer gives SP chip c the chunk_local-token blocks c, c+sp, ... of the global sequence,
so a chunk that starts mid-slab (a multi-turn resume at a tile boundary) is not a contiguous split of its tokens:
the first-rank input must be rotated to match (as the engine's H2D prefill connector does), and per-chip counts of
real rows follow the same placement.
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
