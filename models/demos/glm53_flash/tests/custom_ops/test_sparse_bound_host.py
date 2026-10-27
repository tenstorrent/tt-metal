# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the sparse-decode live bound (``custom_ops/sparse_bound.py``; ``GLM53_SPARSE_DECODE_BOUND`` in
``tt/dsa.py``), no device and no ttnn import:

* the bound's granule (a multiple of the scorer chunk and of the top-k chunk) and the rounding of the users' top-k
  widths (never below the largest, never below one granule);
* the scorer's runtime work split on the 8x8 (Wormhole), 12x10 and 13x10 (Blackhole) grids: the (user, band) items
  partition the LIVE chunks exactly (every live chunk once, in order), the per-band load is ceil(live / n_bands), the
  unbounded split is the capacity's static one; a Python port of the kernels' derivation agrees with the planner;
* the segmented top-k's per-segment-row bounds (the live columns of every user sum to its width, empty segments are
  one synthesized -inf chunk, partial ones the LLK's chunk / tail) and ``check_bound_covers``;
* the phantom plan of a partial classic row (``GLM53_TOPK_PHANTOM``: cheap = the mailbox's merge + rebuild replays,
  full = synthesized pages; a fused row drops its trailing chunks): the compute always sees the unbounded row's chunk
  count, the DRAM reads never pass the live tail;
* a torch reference of the two-stage top-k under the bound against the unbounded one (distinct scores: identical
  candidate values and ids per segment, identical final top-512 per user; the K + tail invariant: >= 512 finite
  candidates for every user whose width is >= 512);
* the kernel ABI: the bound's compile-time / runtime indices in the scorer's three kernels and the top-k reader's
  segment-mode arguments against the vectors ``op.py`` packs, the mailbox CB index, the exit full barriers.

    pytest --noconftest models/demos/glm53_flash/tests/custom_ops/test_sparse_bound_host.py -q
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.custom_ops.sparse_bound import (
    DEFAULT_KC,
    FUSED_MAX_CHUNKS,
    GRANULE,
    PHANTOM_CHEAP,
    PHANTOM_ENV,
    PHANTOM_FULL,
    PHANTOM_NONE,
    TOPK_CHUNK,
    bound_granule,
    check_bound_covers,
    live_bound_pools,
    live_chunks,
    phantom_code,
    phantom_mode,
    scorer_bands,
    scorer_chunk_cols,
    scorer_item_ranges,
    segment_row_bounds,
    topk_row_chunks,
    topk_row_chunks_unbounded,
    topk_row_plan,
)

OPS = Path(__file__).resolve().parents[2] / "custom_ops"
SCORER_KERNELS = OPS / "index_score_decode" / "kernels"
TOPK_KERNELS = OPS / "topk_rows" / "kernels"
GRIDS = [(8, 8), (12, 10), (13, 10)]
K = 512
# pools per user of the index cache (TtIndexCache.pools_per_user: ceil((pools + 8) / 256) * 256) at the capacities the
# bound matters for: 1088 (the profiler's short build), 131072 (the block test), 524288, 1048576 (served)
CAPACITIES = {1088: 768, 131072: 33024, 524288: 131328, 1048576: 262400}


@contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture, local to this module (``--noconftest``)."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


# ------------------------------------------------------------------------------------------ the kernels' derivation
def _kernel_live_chunks(bound_pools: int, Tt: int, KC: int) -> int:
    """index_score_decode_bound.hpp ``index_score_live_chunks``."""
    cap = Tt * 32
    live = bound_pools if bound_pools < cap else cap
    chunk = 32 * KC
    n = (live + chunk - 1) // chunk
    return 1 if n == 0 else n


def _kernel_item_range(band: int, n_bands: int, live: int) -> tuple[int, int]:
    """``index_score_band_chunks`` + ``index_score_item_range``."""
    per = (live + n_bands - 1) // n_bands
    c0 = band * per
    return c0, (c0 + per) if c0 + per < live else live


def _topk_reader_row_len(valid_u: int, row: int, seg: int, seg_cols: int) -> int:
    """reader_topk_rows.cpp segment mode: the row's live columns (the user's bound clamped into the segment)."""
    user_width = seg * seg_cols
    user_len = valid_u if valid_u < user_width else user_width
    seg_start = (row % seg) * seg_cols
    row_len = user_len - seg_start if user_len > seg_start else 0
    return min(row_len, seg_cols)


def _reader_row_plan(row_len: int, seg_cols: int, code: int) -> dict:
    """reader_topk_rows.cpp segment mode after the row's bound: the mailbox (num_chunks, tail, phantom), the DRAM
    reads (live chunks / tail) and the synthesized pages, as the kernel derives them."""
    total_n = (seg_cols + K - 1) // K
    total_tail = seg_cols - (total_n - 1) * K
    NO = None
    if row_len == 0:
        bounds, synth = (1, K, 0), 0
    else:
        n = (row_len + K - 1) // K
        bounds, synth = (n, row_len - (n - 1) * K, 0), NO
        trailing = total_n - n
        if code == 1:
            bounds = (n, bounds[1], trailing)
        elif code == 2 and trailing:
            synth = n
            bounds = (total_n, total_tail, 0)
    live_chunks = 0 if synth == 0 else (bounds[0] if synth is NO else synth)
    live_tail = bounds[1] if synth is NO else (0 if synth == 0 else row_len - (synth - 1) * K)
    return dict(
        num_chunks=bounds[0],
        tail_elements=bounds[1],
        phantom_chunks=bounds[2],
        live_chunks=live_chunks,
        live_tail=live_tail,
        synth=synth,
    )


def _ct_indices(src: str) -> list[int]:
    return [int(m) for m in re.findall(r"get_compile_time_arg_val\((\d+)\)", src)]


def _rt_indices(src: str) -> list[int]:
    return [int(m) for m in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", src)]


# ------------------------------------------------------------------------------------------ granule + rounding
def test_granule():
    assert scorer_chunk_cols(DEFAULT_KC) == 256 and TOPK_CHUNK == 512
    assert GRANULE == bound_granule(DEFAULT_KC) == 512
    assert bound_granule(4) == 512 and bound_granule(3) == 1536  # lcm(96, 512)
    for kc in (1, 2, 3, 4, 8):
        g = bound_granule(kc)
        assert g % scorer_chunk_cols(kc) == 0 and g % TOPK_CHUNK == 0


@pytest.mark.parametrize(
    "valid, expect",
    [([512], 512), ([512, 512, 512, 512], 512), ([513], 1024), ([600, 512], 1024), ([2051, 512, 700], 2560), ([], 512)],
)
def test_live_bound_rounding(valid, expect):
    b = live_bound_pools(valid)
    assert b == expect
    assert b % GRANULE == 0 and b >= max(valid, default=0) and b >= GRANULE


@pytest.mark.parametrize("W", sorted(CAPACITIES.values()))
def test_live_chunks(W):
    assert live_chunks(512, W) == 2 and live_chunks(1024, W) == min(4, -(-W // 256))  # W = 768 clamps
    assert live_chunks(0, W) == 1
    assert live_chunks(W, W) == live_chunks(10 * W, W) == -(-W // 256)
    for bound in (512, 1024, 2560, W, 2 * W):
        assert live_chunks(bound, W) == _kernel_live_chunks(bound, W // 32, DEFAULT_KC)


def test_check_bound_covers():
    check_bound_covers([512, 600], 1024, 33024)
    check_bound_covers([33024], 40000, 33024)  # the bound beyond the capacity clamps to it
    with expect_error(ValueError, "beyond the live bound"):
        check_bound_covers([1025], 1024, 33024)
    with expect_error(ValueError, "not a multiple"):
        check_bound_covers([512], 600, 33024)


# ------------------------------------------------------------------------------------------ the scorer's split
@pytest.mark.parametrize("grid", GRIDS)
@pytest.mark.parametrize("B", [1, 4, 8, 32])
@pytest.mark.parametrize("W", sorted(CAPACITIES.values()))
def test_scorer_split_partitions_live_chunks(grid, B, W):
    n_cores = grid[0] * grid[1]
    total = W // 256
    n_bands, band_chunks = scorer_bands(total, n_cores, B)
    # about one item per core: B * n_bands may exceed the grid by up to B - 1 (the op deals the items round-robin)
    assert 1 <= n_bands <= max(1, -(-n_cores // B)) and B * n_bands < n_cores + B
    assert n_bands == -(-total // band_chunks)
    # unbounded: the static split covers the capacity exactly
    static = [r for r in scorer_item_ranges(n_bands, band_chunks, total, None) if r[0] < r[1]]
    assert static[0][0] == 0 and static[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(static, static[1:]))
    assert all(r[1] - r[0] == band_chunks for r in static[:-1])
    lives = sorted({1, 2, 4, 5, 37, max(1, total // 3), max(1, total - 1), total})
    for live in lives:
        if live > total:
            continue
        ranges = scorer_item_ranges(n_bands, band_chunks, total, live)
        busy = [r for r in ranges if r[0] < r[1]]
        # a partition of [0, live): contiguous, ascending, every chunk once
        assert busy[0][0] == 0 and busy[-1][1] == live, (grid, B, W, live, busy[:3], busy[-1])
        assert all(a[1] == b[0] for a, b in zip(busy, busy[1:]))
        per = -(-live // n_bands)
        assert all(r[1] - r[0] <= per for r in busy) and all(r[1] - r[0] == per for r in busy[:-1])
        # the empty bands follow the busy ones (no CB traffic on their cores, their epilogue still runs)
        assert all(r[0] >= r[1] for r in ranges[len(busy) :])
        # the kernels' port agrees with the planner band by band
        for band in range(n_bands):
            assert ranges[band] == _kernel_item_range(band, n_bands, live)
    logger.info(
        f"grid {grid} B {B} W {W}: n_bands {n_bands} x band_chunks {band_chunks} static; live 2 chunks -> "
        f"{len([r for r in scorer_item_ranges(n_bands, band_chunks, total, 2) if r[0] < r[1]])} busy bands per user"
    )


def test_scorer_split_served_point():
    """The measured point: (8,4) Galaxy chip = 120 cores, 4 users, 262 400 pools (1 M build), position 1024."""
    total = 262400 // 256
    n_bands, band_chunks = scorer_bands(total, 120, 4)
    assert (n_bands, band_chunks) == (30, 35)
    live = live_chunks(live_bound_pools([512, 512, 512, 512]), 262400)
    assert live == 2
    busy = [r for r in scorer_item_ranges(n_bands, band_chunks, total, live) if r[0] < r[1]]
    assert busy == [(0, 1), (1, 2)]  # 2 chunks (64 KB of bf16 keys) per user instead of 1025 (32.8 MB)


# ------------------------------------------------------------------------------------------ the segmented top-k
def test_segment_row_bounds():
    S, Wseg = 5, 52480  # the 1 M build: 262 400 pools, segments_for -> 5
    valid = [512, 600, 40000, 52480, 52481, 262400, 300000]
    rows = segment_row_bounds(valid, S, Wseg)
    assert len(rows) == len(valid) * S
    for u, v in enumerate(valid):
        mine = rows[u * S : (u + 1) * S]
        assert sum(mine) == min(v, S * Wseg) and all(0 <= r <= Wseg for r in mine)
        # the live columns are a prefix of the segments: once a segment is partial, the later ones are empty
        seen_partial = False
        for r in mine:
            if seen_partial:
                assert r == 0
            if r < Wseg:
                seen_partial = True
        for s in range(S):
            assert _topk_reader_row_len(v, u * S + s, S, Wseg) == mine[s]
    assert topk_row_chunks(0) == (1, K, True)
    assert topk_row_chunks(88) == (1, 88, False)
    assert topk_row_chunks(600) == (2, 88, False)
    assert topk_row_chunks(512) == (1, 512, False)
    assert topk_row_chunks(Wseg)[:2] == topk_row_chunks_unbounded(Wseg) == (103, 256)


def test_phantom_mode_env():
    assert phantom_mode({}) == "cheap" and phantom_mode({PHANTOM_ENV: "full"}) == "full"
    with expect_error(ValueError, "expected one of"):
        phantom_mode({PHANTOM_ENV: "none"})
    # a fused row (<= 32 chunks) never needs phantoms; a classic row takes the mode's code
    assert phantom_code(8192, "cheap") == PHANTOM_NONE and phantom_code(16384, "full") == PHANTOM_NONE
    assert phantom_code(16385, "cheap") == PHANTOM_CHEAP and phantom_code(52480, "full") == PHANTOM_FULL
    assert FUSED_MAX_CHUNKS * K == 16384


@pytest.mark.parametrize("seg_cols", [52480, 43776, 33024, 8192])
@pytest.mark.parametrize("code", [PHANTOM_NONE, PHANTOM_CHEAP, PHANTOM_FULL])
def test_topk_row_plan(seg_cols, code):
    """The partial-row plan: under codes 1 / 2 the compute processes the unbounded row's chunk count, the DRAM reads
    never pass the live tail, the empty row is one synthesized chunk; the kernel port agrees with the planner."""
    total, total_tail = topk_row_chunks_unbounded(seg_cols)
    for row_len in (0, 1, 88, 512, 513, 600, 1024, 1025, 5000, seg_cols - 1, seg_cols):
        plan = topk_row_plan(row_len, seg_cols, code)
        assert plan == _reader_row_plan(row_len, seg_cols, code), (seg_cols, code, row_len, plan)
        if row_len == 0:
            assert plan == dict(num_chunks=1, tail_elements=K, phantom_chunks=0, live_chunks=0, live_tail=0, synth=0)
            continue
        n, tail = topk_row_chunks(row_len)[:2]
        assert plan["live_chunks"] == n and plan["live_tail"] == tail  # the reads: the live columns only
        assert (plan["live_chunks"] - 1) * K + plan["live_tail"] == row_len
        if code == PHANTOM_NONE:
            assert plan["num_chunks"] == n and plan["phantom_chunks"] == 0 and plan["synth"] is None
        elif code == PHANTOM_CHEAP:
            assert plan["num_chunks"] + plan["phantom_chunks"] == total and plan["synth"] is None
            assert plan["num_chunks"] == n and plan["tail_elements"] == tail
        else:
            assert plan["num_chunks"] == total and plan["phantom_chunks"] == 0
            if n < total:
                assert plan["tail_elements"] == total_tail and plan["synth"] == n
            else:
                assert plan["synth"] is None and plan["tail_elements"] == tail


def _two_stage(scores: torch.Tensor, valid: list[int], S: int, bounded: bool):
    """Torch model of ``TopkRows.segmented``: stage 1 per segment row (over the live columns when ``bounded``, over the
    whole row with -inf beyond valid otherwise), stage 2 over the S*K candidates -> (vals1 [B, S*K], ids1 [B, S*K],
    top [B, K]). Sentinel -1 for -inf survivors."""
    B, W = scores.shape
    Wseg = W // S
    vals1 = torch.full((B, S * K), float("-inf"))
    ids1 = torch.full((B, S * K), -1, dtype=torch.long)
    for b in range(B):
        row = scores[b].clone()
        row[valid[b] :] = float("-inf")  # the scorer's mask
        lens = segment_row_bounds([valid[b]], S, Wseg)
        for s in range(S):
            seg = row[s * Wseg : (s + 1) * Wseg]
            if bounded:
                n, tail, synth = topk_row_chunks(lens[s])
                seen = torch.full((Wseg,), float("-inf"))
                if not synth:
                    seen[: lens[s]] = seg[: lens[s]]
                seg = seen
            v, i = torch.topk(seg, K)
            i[~torch.isfinite(v)] = -1
            vals1[b, s * K : (s + 1) * K] = v
            ids1[b, s * K : (s + 1) * K] = torch.where(i >= 0, i + s * Wseg, i)
    top = torch.full((B, K), -1, dtype=torch.long)
    for b in range(B):
        v, slot = torch.topk(vals1[b], K)
        g = ids1[b, slot]
        g[~torch.isfinite(v)] = -1
        top[b] = g
    return vals1, ids1, top


@pytest.mark.parametrize("S, Wseg", [(5, 52480), (3, 43776), (8, 8192)])
def test_bounded_two_stage_topk_matches_unbounded(S, Wseg):
    W = S * Wseg
    g = torch.Generator().manual_seed(W)
    B = 4
    # distinct scores (a random permutation of distinct floats): the reference order is then unique, so the equality
    # below covers the order, not just the sets
    scores = torch.randperm(B * W, generator=g).float().reshape(B, W) / 7.0
    valid = [512, 600, Wseg + 1, min(W, 40000)]
    for b in range(B):
        valid[b] = min(valid[b], W)
    bound = live_bound_pools(valid)
    check_bound_covers(valid, bound, W)
    v_ref, i_ref, top_ref = _two_stage(scores, valid, S, bounded=False)
    v_b, i_b, top_b = _two_stage(scores, valid, S, bounded=True)
    assert torch.equal(v_ref, v_b) and torch.equal(i_ref, i_b) and torch.equal(top_ref, top_b)
    for b in range(B):
        finite = int(torch.isfinite(v_b[b]).sum())
        # stage 1 keeps min(live, 512) finite candidates per segment; a user's width >= 512 leaves >= 512 of them, so
        # the top-512 are all real pools (K ranked pools + the tail pool for the gather)
        assert finite == sum(min(n, K) for n in segment_row_bounds([valid[b]], S, Wseg)) and finite >= K
        assert (top_b[b] >= 0).all() and (top_b[b] < valid[b]).all()
        assert len(set(top_b[b].tolist())) == K
    logger.info(f"two-stage top-k S={S} Wseg={Wseg}: bounded == unbounded for valid {valid} (bound {bound})")


# ------------------------------------------------------------------------------------------ kernel ABI
def test_scorer_kernel_abi():
    reader = (SCORER_KERNELS / "reader_index_score_decode.cpp").read_text()
    compute = (SCORER_KERNELS / "compute_index_score_decode.cpp").read_text()
    writer2 = (SCORER_KERNELS / "writer_index_score_decode_v2.cpp").read_text()
    writer1 = (SCORER_KERNELS / "writer_index_score_decode.cpp").read_text()
    op = (OPS / "index_score_decode" / "op.py").read_text()
    # the mailbox CB: one index in op.py, the reader and the compute
    assert "CB_BOUND = 8" in op
    assert re.search(r"cb_bound = 8;", reader) and re.search(r"cb_bound = 8;", compute)
    # reader: [Dt, KC, Tt, k_tile_bytes, slot_base, slot_stride, slot_pages, BOUNDED] then the accessors from 8
    assert _ct_indices(reader) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert "TensorAccessorArgs<8>()" in reader and "bound_args = TensorAccessorArgs<w_args.next" in reader
    assert "int(bounded)]" in op and "acc(q) + acc(pools) + acc(w) + acc(bound_t)" in op
    # reader runtime args: the 9 of the static split then [bound_addr, bound_page_bytes]
    assert _rt_indices(reader) == list(range(11))
    assert re.search(r"get_arg_val<uint32_t>\(9\)", reader) and "bound_page_bytes = get_arg_val<uint32_t>(10)" in reader
    # compute: [Dt, KC, BOUNDED], the mailbox popped once
    assert _ct_indices(compute) == [0, 1, 2] and "compute_ct = [Dt, KC, int(bounded)]" in op
    assert compute.count("read_tile_value(cb_bound, 0, 0)") == 1 and compute.count("cb_pop_front(cb_bound, 1)") == 1
    assert compute.index("cb_wait_front(cb_bound, 1)") < compute.index("for (uint32_t it = 0; it < n_items; ++it)")
    # writer v2: [KC] + acc(out) + [seg_cols, segs] + acc(valid) + [BOUNDED] + acc(bound); runtime 10 / 11
    assert "bounded = get_compile_time_arg_val(b_base)" in writer2 and "TensorAccessorArgs<b_base + 1>" in writer2
    assert "[int(bounded)] + acc(bound if bounded else out)" in op
    assert _rt_indices(writer2) == list(range(12))
    assert "bound_rt = [bound.buffer_address(), bound.buffer_aligned_page_size()] if bounded else [0, 64]" in op
    # every kernel of the op derives the item range from the shared header, and the DM kernels exit on a full barrier
    for src in (reader, compute, writer2):
        assert '#include "index_score_decode_bound.hpp"' in src
        assert src.count("index_score_item_range(band, per_band, live, c_begin, c_end)") == 1
        assert "index_score_band_chunks(live, n_bands)" in src
    for src in (reader, writer1, writer2):
        assert src.rstrip().endswith("noc_async_full_barrier();\n}")
    # the generic op lists the bound tensor (its address is a runtime arg: persistent under trace)
    assert "([bound] if bounded else [])" in op


def test_topk_reader_segment_mode_abi():
    reader = (TOPK_KERNELS / "reader_topk_rows.cpp").read_text()
    op = (OPS / "topk_rows" / "op.py").read_text()
    assert "seg_rows_per_user = get_compile_time_arg_val(per_row_base + 1)" in reader
    assert "seg_cols = get_compile_time_arg_val(per_row_base + 2)" in reader
    assert "phantom_mode = get_compile_time_arg_val(per_row_base + 3)" in reader
    # both call sites append the three segment-mode args after the per-row flag
    flat = re.sub(r"\s+", " ", op)  # black wraps the list: compare whitespace-normalized
    assert "W if has_meta and seg_rows_per_user else 0, phantom_mode_code, ]" in flat
    assert "reader_ct += [0, 0, 0]  # no segment mode" in op
    assert "self._program(scores_seg, valid_rows, ids1, values=vals1, seg_rows_per_user=S, phantom=phantom)" in op
    # the empty segment / the full-mode trailing chunks are synthesized in L1 (no DRAM read), the full-mode partial
    # page is -inf past its tail (disjoint from the read), the reader exits on a full barrier
    assert "fill[w] = NEG_INF_PAIR" in reader and "fill[e] = 0xFF80u" in reader
    assert "bounds.phantom_chunks = trailing;" in reader and "synth_from = bounds.num_chunks;" in reader
    assert "meta_page = seg_rows_per_user ? row / seg_div : row" in reader
    assert reader.rstrip().endswith("noc.async_full_barrier();\n}")
    # the mailbox carries the phantom count (3 words); the seg compute replays it in the classic body only
    meta = (TOPK_KERNELS / "topk_large_indices_metadata.hpp").read_text()
    assert "uint32_t phantom_chunks;" in meta and "sizeof(TopkMetadataBounds) == 3 * sizeof(uint32_t)" in meta
    seg = (TOPK_KERNELS / "compute_topk_rows_seg.cpp").read_text()
    assert "phantom_mode = get_compile_time_arg_val(8)" in seg and "compute_ct += [cb_values, phantom_mode_code]" in op
    assert "read_tile_value(meta_cb, 0, topk_metadata_phantom_chunks_word)" in seg
    assert "reduce_classic_row<K>(input, num_chunks, tail_elements, phantom_chunks)" in seg
    assert "if (num_chunks == 1 && phantom_chunks == 0)" in seg
    assert "topk_xl_copy_tile<K>(input_cb, incoming_slot, 0, 0);" in seg
    phantom = seg[seg.index("FORCE_INLINE void phantom_classic_chunk") :]
    phantom = phantom[: phantom.index("}\n")]
    for call in ("topk_xl_init<K, false>();", "topk_xl_merge<K, false>(survivor_slot);", "topk_xl_rebuild<K, false>"):
        assert call in phantom
    for skipped in ("add_lsb_indices", "local_sort", "separate_indices", "wait_front"):
        assert skipped not in phantom
    # the compute kernels pop the per-row bounds unchanged (the mailbox contract)
    for name in ("compute_topk_rows.cpp", "compute_topk_rows_seg.cpp"):
        src = (TOPK_KERNELS / name).read_text()
        assert "pop_bounds();  // this row's bounds, pushed by the reader before its chunks" in src


def test_dsa_wiring():
    dsa = (Path(__file__).resolve().parents[2] / "tt" / "dsa.py").read_text()
    assert 'SPARSE_BOUND_ENV = "GLM53_SPARSE_DECODE_BOUND"' in dsa
    assert 'mode = env.get(SPARSE_BOUND_ENV, "0")' in dsa  # default off
    assert "self.sparse_bound = sparse_decode_bound_enabled()" in dsa
    assert "bound = meta.live_bound if self.sparse_bound else None" in dsa
    assert "live_bound=_meta_upload(device, cls.live_bound_values(valid, R, C, B), ttnn.uint32, mesh, partition)" in dsa
    assert "put(self.live_bound_values(valid, R, C, B), ttnn.uint32, self.live_bound)" in dsa
    # the single-stage form writes the persistent rows in place (generic_op returns its LAST io tensor: the scorer
    # now lists ``out`` last, and dsa.py uses the buffer, not the return value)
    assert "rows = self._score_buffers(B, W, 1)" in dsa
    scorer = (OPS / "index_score_decode" / "op.py").read_text()
    assert "+ ([bound] if bounded else []) + [out]" in scorer
    # the two scorer forms and the segmented top-k take the bound; the single-stage top-k is per-row bounded already
    assert dsa.count("bound=bound,") == 2
    assert "valid_rows=meta.valid_len_rows if bound is not None else None" in dsa
