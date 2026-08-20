# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only unit test for the bounded sliding-window ring-read plan math (PR2). No device, no ttnn ops.

``_plan`` below is a LINE-FOR-LINE python mirror of the C++ single source of truth
``ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp``
(``chunked_sliding_halo_tile_rows`` / ``chunked_sliding_halo_source_start_tile`` /
``build_sliding_q_work_plan`` with the PR2 ``bounded_kv_slab_count`` wrap and the
``first_global_k_chunk`` range field). That header is compiled into the HOST program factory, the
ring READER kernel, and the COMPUTE kernel — any change to it MUST be mirrored here in lockstep
(and vice versa), exactly like ring_joint_kv_pad_derivation.hpp's host/device copies.

What is verified, cross-checked against PR1's production un-rotation helper
``bounded_blockcyclic_positions`` (kv_cache.py — the same helper the host-readback PCC instrument
uses, itself validated against a write simulation in test_bounded_kv_math.py):

  1. POSITION IDENTITY: for every chunk 0..N-1 (chunk 0 included — #53153 unified the first group
     into the ring path, and the plan/halo helpers now accept a single complete group; including
     across the circular wrap), every ring
     device, every Q chunk, and every K tile-row the bounded plan reads (own slab + one-hop halo),
     the global K position the plan RECONSTRUCTS (``first_global_k_chunk``, what the compute mask
     uses) equals the global position ACTUALLY RESIDENT at that (device, local row) of the bounded
     circular cache. This is the property the slab-mod exists for.
  2. WINDOW COVERAGE + MASK EXACTNESS: for every Q token position, the positions the plan reads,
     filtered by the causal sliding-window predicate on the reconstructed positions, are EXACTLY
     the window ``[max(0, p - window + 1), p]`` per ``bounded_blockcyclic_positions``.
  3. BOUNDED == UNBOUNDED up to slab placement: the bounded plan differs from the unbounded plan
     (on an unbounded cache) ONLY in the local slab base (``group -> group mod n_slabs``): source
     device, chunk counts, compact (halo-buffer) chunks, and global positions are identical. With
     ``bounded_kv_slab_count=0`` the mirror is byte-identical to the pre-PR2 algorithm.
  4. HALO FEASIBILITY: every halo-sourced row lies inside the tail the one-hop gather actually
     sends (``chunked_sliding_halo_source_start_tile`` window), with consistent compact indexing,
     and all bounded local rows lie inside the physical cache extent.

Geometry note: the plan requires ``halo_tile_rows <= q_local_tile_rows`` (the op asserts the same:
halo must fit one per-device slab), so the PR1 write-math case (C=256, sp=4) is too small for a
128-token window here — these tests use chunk_local >= 256 tokens, the op-supported K chunk of 128
tokens, and still exercise the same m=2 wrap over 5 chunks.

Run:
    pytest models/demos/gpt_oss_d_p/tests/unit/test_bounded_ring_plan_math.py
"""

from dataclasses import dataclass, field
from typing import List

import pytest

from models.demos.gpt_oss_d_p.tt.attention.kv_cache import bounded_blockcyclic_positions

TILE = 32


# ---------------------------------------------------------------------------------------------
# Python mirror of sliding_window_work_plan.hpp — keep in lockstep with the C++ (see module doc).
# ---------------------------------------------------------------------------------------------


def _halo_tile_rows(sliding_window_tokens, tile_height, k_chunk_tile_rows):
    """Mirror of chunked_sliding_halo_tile_rows."""
    if tile_height == 0 or k_chunk_tile_rows == 0:
        return 0
    left_window_tokens = sliding_window_tokens - 1 if sliding_window_tokens > 0 else 0
    k_chunk_tokens = k_chunk_tile_rows * tile_height
    return ((left_window_tokens + k_chunk_tokens - 1) // k_chunk_tokens) * k_chunk_tile_rows


def _bounded_kv_local_slab(source_group, bounded_kv_slab_count):
    """Mirror of bounded_kv_local_slab: chunk group g lives in local slab g % n_slabs; 0/1 = identity."""
    return source_group % bounded_kv_slab_count if bounded_kv_slab_count > 1 else source_group


def _halo_source_start_tile(
    source_device, q_local_tile_rows, ring_size, logical_k_tile_rows, halo_tile_rows, bounded_kv_slab_count=0
):
    """Mirror of chunked_sliding_halo_source_start_tile."""
    q_group_tile_rows = q_local_tile_rows * ring_size
    if q_group_tile_rows == 0 or logical_k_tile_rows < q_group_tile_rows or halo_tile_rows > q_local_tile_rows:
        return 0
    current_group = logical_k_tile_rows // q_group_tile_rows - 1
    # First-group clamp (#53153): device R-1's wrap predecessor does not exist in group 0; the
    # payload sent on that edge is never read (device 0 clips), keep the origin in bounds.
    if current_group == 0 and source_device + 1 == ring_size:
        return 0
    source_group = current_group - 1 if source_device + 1 == ring_size else current_group
    return _bounded_kv_local_slab(source_group, bounded_kv_slab_count) * q_local_tile_rows + (
        q_local_tile_rows - halo_tile_rows
    )


@dataclass
class _Range:
    """Mirror of SlidingKVSourceRange."""

    source_ring_id: int = 0
    first_k_chunk: int = 0
    last_k_chunk: int = 0
    first_compact_k_chunk: int = 0
    first_global_k_chunk: int = 0

    @property
    def k_chunk_count(self):
        return self.last_k_chunk - self.first_k_chunk


@dataclass
class _Plan:
    """Mirror of SlidingQWorkPlan (max_source_ranges=2)."""

    source_ranges: List[_Range] = field(default_factory=list)
    total_k_chunk_count: int = 0
    is_valid: bool = False


def _plan(
    q_local_start_tile,
    q_chunk_tile_rows,
    q_device_index,
    q_local_tile_rows,
    ring_size,
    sliding_window_tokens,
    tile_height,
    k_local_tile_rows,
    k_chunk_tile_rows,
    logical_k_tile_rows,
    bounded_kv_slab_count=0,
):
    """Mirror of build_sliding_q_work_plan. Only the local-row derivation wraps under bounded;
    every range_global_* / first_global_k_chunk value stays absolute."""
    plan = _Plan()
    if (
        q_chunk_tile_rows == 0
        or q_local_tile_rows == 0
        or ring_size == 0
        or sliding_window_tokens == 0
        or tile_height == 0
        or k_chunk_tile_rows == 0
        or q_local_tile_rows % k_chunk_tile_rows != 0
    ):
        return plan

    q_group_tile_rows = ring_size * q_local_tile_rows
    # A complete first group is valid (#53153): device 0 clips at token 0 and every other
    # device consumes its predecessor within the same group.
    if logical_k_tile_rows < q_group_tile_rows or q_local_start_tile + q_chunk_tile_rows > q_local_tile_rows:
        return plan

    halo_tile_rows = _halo_tile_rows(sliding_window_tokens, tile_height, k_chunk_tile_rows)
    if halo_tile_rows > q_local_tile_rows:
        return plan

    current_q_group_start = logical_k_tile_rows - q_group_tile_rows
    global_q_start_tile = current_q_group_start + q_device_index * q_local_tile_rows + q_local_start_tile
    left_window_tokens = sliding_window_tokens - 1 if sliding_window_tokens > 0 else 0
    left_window_tile_rows = 0 if tile_height == 0 else (left_window_tokens + tile_height - 1) // tile_height
    window_start_tile = global_q_start_tile - left_window_tile_rows if global_q_start_tile > left_window_tile_rows else 0
    window_end_tile = global_q_start_tile + q_chunk_tile_rows

    clipped_window_start = min(window_start_tile, logical_k_tile_rows)
    clipped_window_end = min(window_end_tile, logical_k_tile_rows)
    if clipped_window_start >= clipped_window_end:
        return plan

    first_slab = clipped_window_start // q_local_tile_rows
    last_slab = (clipped_window_end - 1) // q_local_tile_rows
    for slab in range(first_slab, last_slab + 1):
        source_ring_id = slab % ring_size
        source_group = slab // ring_size
        slab_global_start = slab * q_local_tile_rows
        range_global_start = max(clipped_window_start, slab_global_start)
        slab_global_end = slab_global_start + q_local_tile_rows
        range_global_end = min(clipped_window_end, slab_global_end)
        source_local_base = _bounded_kv_local_slab(source_group, bounded_kv_slab_count) * q_local_tile_rows
        range_local_start = source_local_base + range_global_start - slab_global_start
        range_local_end = source_local_base + range_global_end - slab_global_start
        if range_local_start >= k_local_tile_rows:
            continue
        clipped_range_local_end = min(range_local_end, k_local_tile_rows)
        if range_local_start >= clipped_range_local_end:
            continue

        first_k_chunk = range_local_start // k_chunk_tile_rows
        last_k_chunk = (clipped_range_local_end + k_chunk_tile_rows - 1) // k_chunk_tile_rows
        first_compact_k_chunk = 0
        if source_ring_id != q_device_index:
            halo_source_start = _halo_source_start_tile(
                source_ring_id,
                q_local_tile_rows,
                ring_size,
                logical_k_tile_rows,
                halo_tile_rows,
                bounded_kv_slab_count,
            )
            # #53153 guard: a remote range must begin inside the fixed-size halo (bounded mode wraps
            # halo_source_start and first_k_chunk through the SAME slab base, so this stays exact).
            if halo_source_start > first_k_chunk * k_chunk_tile_rows:
                return _Plan()
            first_compact_k_chunk = (first_k_chunk * k_chunk_tile_rows - halo_source_start) // k_chunk_tile_rows
        first_global_k_chunk = slab_global_start // k_chunk_tile_rows + first_k_chunk - source_local_base // k_chunk_tile_rows
        if len(plan.source_ranges) == 2:  # SlidingQWorkPlan::max_source_ranges
            return _Plan()
        plan.source_ranges.append(
            _Range(source_ring_id, first_k_chunk, last_k_chunk, first_compact_k_chunk, first_global_k_chunk)
        )
        plan.total_k_chunk_count += plan.source_ranges[-1].k_chunk_count
    plan.is_valid = plan.total_k_chunk_count != 0
    return plan


# ---------------------------------------------------------------------------------------------
# Test geometry
# ---------------------------------------------------------------------------------------------

# (sp, chunk_local_tokens, m, n_chunks) — window 128 / K chunk 128 tokens (the op-supported sizes).
CASES = [
    (4, 256, 2, 5),  # SP4 production ring, m=2 wrap over 5 chunks (wrap first crossed at chunk 2)
    (8, 256, 2, 5),  # SP8 test ring, same wrap
    (4, 512, 2, 4),  # larger slab: window+halo well inside one slab
]
WINDOW = 128
K_CHUNK_TOKENS = 128
Q_CHUNK_TOKENS = 128


def _geometry(sp, chunk_local, m):
    q_local_t = chunk_local // TILE
    k_chunk_t = K_CHUNK_TOKENS // TILE
    q_chunk_t = Q_CHUNK_TOKENS // TILE
    chunk_global = chunk_local * sp
    capacity = m * chunk_global
    cap_local_t = (capacity // sp) // TILE
    return q_local_t, k_chunk_t, q_chunk_t, chunk_global, capacity, cap_local_t


def _plans_for_chunk(sp, chunk_local, m, chunk_idx, bounded):
    """Yield (device, q_chunk_start_tile, plan) for every device / Q chunk of one chunked ring read."""
    q_local_t, k_chunk_t, q_chunk_t, chunk_global, capacity, cap_local_t = _geometry(sp, chunk_local, m)
    logical_nt = (chunk_idx + 1) * chunk_global // TILE
    k_local_t = cap_local_t if bounded else logical_nt // sp
    n_slabs = m if bounded else 0
    for dev in range(sp):
        for q_start in range(0, q_local_t, q_chunk_t):
            plan = _plan(
                q_start,
                q_chunk_t,
                dev,
                q_local_t,
                sp,
                WINDOW,
                TILE,
                k_local_t,
                k_chunk_t,
                logical_nt,
                n_slabs,
            )
            yield dev, q_start, plan


def _read_positions(sp, chunk_global, capacity, logical_n, dev, plan, k_chunk_t):
    """The (reconstructed_global, resident_global) token pairs a plan reads from the bounded cache.

    Resident positions come from PR1's production un-rotation helper (slab j holds the LARGEST
    written chunk group g with g mod m == j), evaluated AFTER this chunk's write (written = logical_n).
    """
    cap_local = capacity // sp
    resident = bounded_blockcyclic_positions(sp, chunk_global, capacity, logical_n)
    pairs = []
    for rng in plan.source_ranges:
        for chunk_off in range(rng.k_chunk_count):
            local_row0 = (rng.first_k_chunk + chunk_off) * k_chunk_t
            global_row0 = (rng.first_global_k_chunk + chunk_off) * k_chunk_t
            for tok in range(k_chunk_t * TILE):
                local_tok = local_row0 * TILE + tok
                assert local_tok < cap_local, "bounded plan row escapes the physical cache extent"
                res = int(resident[rng.source_ring_id * cap_local + local_tok])
                rec = global_row0 * TILE + tok
                pairs.append((rec, res))
    return pairs


@pytest.mark.parametrize("sp, chunk_local, m, n_chunks", CASES)
def test_bounded_plan_reads_resident_positions(sp, chunk_local, m, n_chunks):
    """(1) POSITION IDENTITY: what the plan thinks it reads (compute-mask coordinates) is what the
    circular cache actually holds at those rows — for every chunk, device, Q chunk, and K row,
    including across the wrap."""
    q_local_t, k_chunk_t, _, chunk_global, capacity, _ = _geometry(sp, chunk_local, m)
    for chunk_idx in range(n_chunks):  # ring cache-read runs on EVERY chunk (chunk 0 included, #53153)
        logical_n = (chunk_idx + 1) * chunk_global
        for dev, q_start, plan in _plans_for_chunk(sp, chunk_local, m, chunk_idx, bounded=True):
            assert plan.is_valid and plan.total_k_chunk_count > 0
            for rec, res in _read_positions(sp, chunk_global, capacity, logical_n, dev, plan, k_chunk_t):
                assert res >= 0, f"chunk {chunk_idx} dev {dev}: plan reads a never-written cache row"
                assert rec == res, (
                    f"chunk {chunk_idx} dev {dev} qs {q_start}: plan reconstructs global {rec} but the "
                    f"cache row holds {res} — reader/compute would mask the wrong positions"
                )


@pytest.mark.parametrize("sp, chunk_local, m, n_chunks", CASES)
def test_bounded_plan_window_coverage_and_mask_exactness(sp, chunk_local, m, n_chunks):
    """(2) For every Q token position p, the plan's resident positions filtered by the causal
    sliding-window predicate (on the RECONSTRUCTED positions, as the compute mask does) are exactly
    [max(0, p-window+1), p]."""
    q_local_t, k_chunk_t, q_chunk_t, chunk_global, capacity, _ = _geometry(sp, chunk_local, m)
    for chunk_idx in range(n_chunks):
        logical_n = (chunk_idx + 1) * chunk_global
        chunk_q_base = chunk_idx * chunk_global  # global position of this chunk's first token
        for dev, q_start, plan in _plans_for_chunk(sp, chunk_local, m, chunk_idx, bounded=True):
            positions = {rec for rec, _ in _read_positions(sp, chunk_global, capacity, logical_n, dev, plan, k_chunk_t)}
            # Q tokens of this (device, q chunk): block-cyclic — device dev owns chunk-local rows
            # [dev*chunk_local, (dev+1)*chunk_local), Q chunk covers q_start..q_start+q_chunk_t tiles.
            q_tok0 = chunk_q_base + dev * chunk_local + q_start * TILE
            for p in range(q_tok0, q_tok0 + q_chunk_t * TILE):
                window = set(range(max(0, p - WINDOW + 1), p + 1))
                attended = {t for t in positions if max(0, p - WINDOW + 1) <= t <= p}
                assert attended == window, (
                    f"chunk {chunk_idx} dev {dev} q token {p}: attended {sorted(attended)[:4]}... != "
                    f"window [{max(0, p - WINDOW + 1)}, {p}]"
                )


@pytest.mark.parametrize("sp, chunk_local, m, n_chunks", CASES)
def test_bounded_plan_matches_unbounded_up_to_slab_mod(sp, chunk_local, m, n_chunks):
    """(3) The bounded plan is the unbounded plan with ONLY the local slab base wrapped
    (group -> group mod m): same sources, counts, compact halo chunks, and global positions.
    Also pins n_slabs=0 to the pre-PR2 unbounded algorithm (identity slab map)."""
    q_local_t, k_chunk_t, _, _, _, _ = _geometry(sp, chunk_local, m)
    for chunk_idx in range(n_chunks):
        unbounded = list(_plans_for_chunk(sp, chunk_local, m, chunk_idx, bounded=False))
        bounded = list(_plans_for_chunk(sp, chunk_local, m, chunk_idx, bounded=True))
        for (dev_u, qs_u, pu), (dev_b, qs_b, pb) in zip(unbounded, bounded):
            assert (dev_u, qs_u) == (dev_b, qs_b)
            assert pu.is_valid and pb.is_valid
            assert len(pu.source_ranges) == len(pb.source_ranges)
            assert pu.total_k_chunk_count == pb.total_k_chunk_count
            for ru, rb in zip(pu.source_ranges, pb.source_ranges):
                assert ru.source_ring_id == rb.source_ring_id
                assert ru.k_chunk_count == rb.k_chunk_count
                assert ru.first_compact_k_chunk == rb.first_compact_k_chunk
                assert ru.first_global_k_chunk == rb.first_global_k_chunk
                # Local rows differ by exactly the slab wrap: group*q_local -> (group%m)*q_local.
                group = ru.first_global_k_chunk * k_chunk_t // (q_local_t * sp)
                slab_delta_chunks = (group - group % m) * q_local_t // k_chunk_t
                assert ru.first_k_chunk - rb.first_k_chunk == slab_delta_chunks
            if chunk_idx + 1 <= m:  # before the first wrap the two plans are identical
                assert pu.source_ranges == pb.source_ranges


@pytest.mark.parametrize("sp, chunk_local, m, n_chunks", CASES)
def test_bounded_halo_rows_inside_gather_tail(sp, chunk_local, m, n_chunks):
    """(4) Every halo-sourced K chunk lies inside the tail the one-hop gather sends (send origin =
    chunked_sliding_halo_source_start_tile with the same slab mod — the runtime-patched CCL window),
    and the compact indices address the gathered buffer consistently."""
    q_local_t, k_chunk_t, _, chunk_global, _, cap_local_t = _geometry(sp, chunk_local, m)
    halo_t = _halo_tile_rows(WINDOW, TILE, k_chunk_t)
    assert 0 < halo_t <= q_local_t, "geometry must satisfy the op's halo-fits-one-slab requirement"
    for chunk_idx in range(n_chunks):
        logical_nt = (chunk_idx + 1) * chunk_global // TILE
        for dev, q_start, plan in _plans_for_chunk(sp, chunk_local, m, chunk_idx, bounded=True):
            for rng in plan.source_ranges:
                if rng.source_ring_id == dev:
                    continue  # own-slab read, no halo
                send_start = _halo_source_start_tile(rng.source_ring_id, q_local_t, sp, logical_nt, halo_t, m)
                first_row = rng.first_k_chunk * k_chunk_t
                last_row = rng.last_k_chunk * k_chunk_t
                assert send_start <= first_row and last_row <= send_start + halo_t, (
                    f"chunk {chunk_idx} dev {dev}: halo read rows [{first_row}, {last_row}) outside the "
                    f"gathered tail [{send_start}, {send_start + halo_t})"
                )
                assert send_start + halo_t <= cap_local_t, "halo tail escapes the bounded cache"
                # Compact indexing: gathered-buffer chunk == source chunk offset within the tail.
                assert rng.first_compact_k_chunk == (first_row - send_start) // k_chunk_t
