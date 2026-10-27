# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host planner of the sparse-decode LIVE BOUND (``GLM53_SPARSE_DECODE_BOUND=1``, ``tt/dsa.py`` ``_decode_unified``):
the per-step number of pooled-key columns the traced decode's scorer and top-k touch, instead of the pool
capacity ``W`` the model was built with.

A served model is built at max_seq_len 1 M = 262 400 pools per user, and the captured decode graph scores, masks and
ranks every one of them for every user at every step: on the (8,4) Galaxy at position 1024 (512 live pools) one DSA
layer's scorer costs 731 us and the segmented top-k's stage 1 228 us against 11 + 5 us when the model is built at 1088
positions (``_logs/mesh/profiler/dsa_cap1m`` vs ``dsa_capshort``, 2026-10-27). The shapes of a captured trace are
fixed, so the bound is a RUNTIME value: one ``[1, 1]`` uint32 tensor per chip (``SparseDecodeMeta.live_bound``, refreshed
with the other per-step tensors) = the largest per-user top-k width of the chip's users (``valid = max(n, K)``),
rounded up to ``GRANULE`` columns. The kernels loop to that bound:

* the scorer (``index_score_decode``) streams / scores / writes the first ``live_chunks = ceil(bound / (32*KC))`` chunks
  of every user's row, the (user, band) work items splitting those chunks instead of the capacity's
  (``scorer_item_range``), and -inf-fills ``[valid[u], bound)`` as before; columns beyond the bound are never written;
* the segmented top-k's stage 1 (``topk_rows.segmented(..., valid_rows=...)``) ranks every segment row over its own live
  columns ``clamp(valid[u] - s * W_seg, 0, W_seg)`` (``segment_row_bounds``): a segment entirely beyond the user's bound is
  one synthesized all--inf chunk (``topk_row_chunks``), never a DRAM read of columns the scorer did not write;
* the unsegmented top-k (``TopkRows.__call__``) already ranks ``[0, valid[u])`` only and reads nothing else.

Every column any top-k reads lies below ``valid[u] <= bound``, every scored column is scored by the same per-chunk
arithmetic as without the bound, and the top-k processes the live chunks in the same order with the same contents
(the dropped chunks are entirely -inf, below every finite survivor), so the ranked ids, the gathered compact cache
and the attention are the unbounded path's (``tests/test_sparse_decode_bound_mesh.py`` asserts bit-identical logits).
This module imports nothing from ttnn or torch (host tests run with ``--noconftest`` and no device).
"""

from __future__ import annotations

from math import gcd
from typing import Sequence

TILE = 32
TOPK_CHUNK = 512  # the K=512 LLK window: the top-k reads and ranks 512-element chunks
DEFAULT_KC = 8  # k-tiles per scorer chunk (IndexScoreDecode.KC): a chunk is 32*KC columns


def scorer_chunk_cols(kc: int = DEFAULT_KC) -> int:
    """Columns of one scorer chunk (``KC`` k-tiles of 32 pools)."""
    return TILE * kc


def bound_granule(kc: int = DEFAULT_KC) -> int:
    """The bound's rounding unit: a multiple of the scorer chunk (whole chunks are scored) and of the top-k chunk (a
    bound at a top-k chunk boundary keeps every tail read inside scored columns even if a reader rounds up)."""
    a, b = scorer_chunk_cols(kc), TOPK_CHUNK
    return a * b // gcd(a, b)


GRANULE = bound_granule()  # 512 columns with KC = 8


def live_bound_pools(valid: Sequence[int], kc: int = DEFAULT_KC, granule: int | None = None) -> int:
    """The live bound of one chip for its users' top-k widths ``valid`` (``SparseDecodeMeta.valid_len_rows``, each
    ``>= K``): ``max(valid)`` rounded up to the granule (never below one granule). Not clamped to the capacity: the
    kernels clamp (``min(bound, W)``), so the host needs no cache geometry."""
    g = bound_granule(kc) if granule is None else granule
    top = max((int(v) for v in valid), default=0)
    return max(g, -(-top // g) * g)


def live_chunks(bound: int, W: int, kc: int = DEFAULT_KC) -> int:
    """Scorer chunks the kernels process for ``bound`` live columns of a ``W``-wide row (the kernels' derivation:
    clamp to the capacity, whole chunks, at least one)."""
    cols = scorer_chunk_cols(kc)
    live = min(int(bound), int(W))
    return max(1, -(-live // cols))


def scorer_bands(total_chunks: int, n_cores: int, B: int) -> tuple[int, int]:
    """The host split of ``IndexScoreDecode``: (n_bands, band_chunks) = about one (user, band) item per core, bands of
    whole chunks of the CAPACITY (the static split the unbounded kernels loop over)."""
    n_bands = max(1, min(-(-n_cores // B), total_chunks))
    band_chunks = -(-total_chunks // n_bands)
    n_bands = -(-total_chunks // band_chunks)
    return n_bands, band_chunks


def scorer_item_range(
    band: int, n_bands: int, band_chunks: int, total_chunks: int, live: int | None
) -> tuple[int, int]:
    """The chunk range ``[c_begin, c_end)`` of work item (user, ``band``): the static split ``[band * band_chunks,
    min(+band_chunks, total_chunks))`` without a bound; with ``live`` chunks the same ``n_bands`` bands split the live
    chunks evenly (``ceil(live / n_bands)`` each; the bands past the live count are empty). The three kernels of the
    op derive this identically from the same runtime bound."""
    if live is None:
        c0 = band * band_chunks
        return c0, min(c0 + band_chunks, total_chunks)
    per = -(-live // n_bands)
    c0 = band * per
    return c0, min(c0 + per, live)


def scorer_item_ranges(n_bands: int, band_chunks: int, total_chunks: int, live: int | None) -> list[tuple[int, int]]:
    """``scorer_item_range`` for every band."""
    return [scorer_item_range(b, n_bands, band_chunks, total_chunks, live) for b in range(n_bands)]


def segment_row_bounds(valid: Sequence[int], segments: int, seg_cols: int) -> list[int]:
    """The live columns of every segment row of the two-stage top-k (row ``u * segments + s`` = columns
    ``[s * seg_cols, (s+1) * seg_cols)`` of user ``u``): ``clamp(valid[u] - s * seg_cols, 0, seg_cols)``; the user's
    bound is clamped to the row's capacity ``segments * seg_cols`` first (as the reader does)."""
    cap = segments * seg_cols
    out = []
    for v in valid:
        vb = min(max(int(v), 0), cap)
        for s in range(segments):
            out.append(min(max(vb - s * seg_cols, 0), seg_cols))
    return out


def topk_row_chunks(row_len: int, llk_k: int = TOPK_CHUNK) -> tuple[int, int, bool]:
    """(num_chunks, tail_elements, synthesized) of a top-k row of ``row_len`` live elements: the LLK bounds
    ``ceil(row_len / K)`` chunks with the last one ``row_len - (num_chunks - 1) * K`` wide; an EMPTY row (a segment
    entirely beyond the user's bound) is one synthesized full chunk of -inf that the reader fills in L1 (no DRAM
    read of the unwritten columns; the same -inf words the unbounded path reads back from the scorer's mask)."""
    if row_len <= 0:
        return 1, llk_k, True
    n = -(-row_len // llk_k)
    return n, row_len - (n - 1) * llk_k, False


def topk_row_chunks_unbounded(seg_cols: int, llk_k: int = TOPK_CHUNK) -> tuple[int, int]:
    """The chunk count / tail of a full segment row (the unbounded stage 1)."""
    n = -(-seg_cols // llk_k)
    return n, seg_cols - (n - 1) * llk_k


def check_bound_covers(valid: Sequence[int], bound: int, W: int, kc: int = DEFAULT_KC) -> None:
    """The invariants the bit-identity rests on, for one chip: every user's top-k width lies within the scored columns
    (``valid[u] <= min(bound, W)``) and the bound is whole scorer chunks (or the capacity)."""
    live_cols = min(int(bound), int(W))
    cols = scorer_chunk_cols(kc)
    for u, v in enumerate(valid):
        if int(v) > live_cols:
            raise ValueError(f"user {u}: top-k width {int(v)} beyond the live bound {live_cols} (bound {bound}, W {W})")
    if live_cols != int(W) and live_cols % cols:
        raise ValueError(f"live bound {live_cols} is not a multiple of the scorer chunk ({cols} columns)")
