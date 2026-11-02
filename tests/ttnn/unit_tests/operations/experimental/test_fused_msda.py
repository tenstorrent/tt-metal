# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for ttnn.experimental.fused_msda / fused_msda_from_offsets.

Everything here is checked against `_msda_reference` below, a self-contained
PyTorch implementation of the MSDA definition built on `F.grid_sample`. It
deliberately does not import the model-repo helpers and does not compare against
the existing `ttnn.experimental.multi_scale_deformable_attn` op: the point is to
pin the new op to the mathematical definition, not to another implementation.
`test_reference_matches_bevformer_reference` is what keeps the reference itself
honest.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------
def _msda_reference(
    value: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    spatial_shapes,
    align_corners: bool = False,
    locations_in_grid_space: bool = False,
) -> torch.Tensor:
    """Multi-scale deformable attention, straight from the definition.

    Args:
        value: (B, S, H, D) float32, S == sum_l H_l * W_l
        sampling_locations: (B, Q, H, L, P, 2) float32, normalized (x, y)
        attention_weights: (B, Q, H, L, P) float32
        spatial_shapes: sequence of (H_l, W_l)
        align_corners: bilinear pixel mapping
        locations_in_grid_space: locations are already in [-1, 1] rather than [0, 1]

    Returns:
        (B, Q, H*D) float32
    """
    B, S, H, D = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape
    assert S == sum(h * w for h, w in spatial_shapes)

    grids = sampling_locations if locations_in_grid_space else 2.0 * sampling_locations - 1.0
    value_levels = value.split([h * w for h, w in spatial_shapes], dim=1)

    out = value.new_zeros(B * H, D, Q)
    for lvl, (h_, w_) in enumerate(spatial_shapes):
        # (B, h*w, H, D) -> (B*H, D, h, w)
        v = value_levels[lvl].flatten(2).transpose(1, 2).reshape(B * H, D, h_, w_)
        # (B, Q, H, P, 2) -> (B*H, Q, P, 2)
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        sampled = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
        # (B, Q, H, P) -> (B*H, 1, Q, P)
        a = attention_weights[:, :, :, lvl].permute(0, 2, 1, 3).reshape(B * H, 1, Q, P)
        out = out + (sampled * a).sum(-1)

    # (B*H, D, Q) -> (B, Q, H*D)
    return out.reshape(B, H, D, Q).permute(0, 3, 1, 2).reshape(B, Q, H * D).contiguous()


def _offset_scale(extent: int, align_corners: bool) -> float:
    """Normalized delta of one feature-map pixel. 0 on a singleton align_corners axis."""
    if align_corners:
        return 0.0 if extent <= 1 else 1.0 / (extent - 1)
    return 1.0 / extent


def _locations_from_offsets(reference_points, sampling_offsets, spatial_shapes, reference_mode, align_corners=False):
    """The V2 frontend, in PyTorch. (B, Q, R, 2) + (B, Q, H, L, P, 2) -> (B, Q, H, L, P, 2).

    Offsets are feature-map pixels; align_corners selects the same spacing the SFPU applies.
    """
    B, Q, H, L, P, _ = sampling_offsets.shape
    R = reference_points.shape[2]
    loc = torch.empty_like(sampling_offsets)
    for lvl, (h_, w_) in enumerate(spatial_shapes):
        norm = torch.tensor(
            [_offset_scale(w_, align_corners), _offset_scale(h_, align_corners)], dtype=sampling_offsets.dtype
        )
        for p in range(P):
            r = lvl if reference_mode == "level" else (p % R)
            # reference_points is head-invariant: (B, Q, 2) broadcast over H
            loc[:, :, :, lvl, p, :] = (
                reference_points[:, :, r, :].unsqueeze(2) + sampling_offsets[:, :, :, lvl, p, :] * norm
            )
    return loc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Row at which the Blackhole split gather hands a block's rows from the reader
# to the writer, keyed by the value tensor's buffer type. Must equal
# fused_msda_gather::SPLIT_ROW_L1 / SPLIT_ROW_DRAM in msda_gather_rows.hpp.
SPLIT_ROW = {"l1": 16, "dram": 15}


def _pack_locations(loc: torch.Tensor) -> torch.Tensor:
    """(B, Q, H, L, P, 2) -> (B, Q, H, L*P*2), the layout the reader indexes as (l*P+p)*2."""
    B, Q, H = loc.shape[:3]
    return loc.reshape(B, Q, H, -1).contiguous()


def _pack_weights(w: torch.Tensor) -> torch.Tensor:
    """(B, Q, H, L, P) -> (B, Q, H, L*P), the layout the reader indexes as l*P+p."""
    B, Q, H = w.shape[:3]
    return w.reshape(B, Q, H, -1).contiguous()


def _pack_value(value: torch.Tensor) -> torch.Tensor:
    """(B, S, H, D) -> (B, S, H*D), heads concatenated on the last dim."""
    B, S, H, D = value.shape
    return value.reshape(B, S, H * D).contiguous()


def _packed_value_fits_dram(num_heads, head_dim):
    """Does the op accept packed value (B, S, H*D) with these heads in DRAM?

    Head h is read at byte offset h*D*2 inside one (b, s) page, so with more than one
    head that offset must be a multiple of the DRAM alignment: on Blackhole (64 B) every
    D % 32 == 16 is refused. test_fused_msda_rejects_unaligned_packed_value_heads pins
    the refusal; this helper only tells the other tests where value can go.
    """
    return num_heads == 1 or (head_dim * 2) % ttnn.device.get_dram_alignment() == 0


def _num_valid_queries_case(B, Q, H, seed, packed=True, D=32, ref_fp32=False):
    spatial_shapes = [(12, 20), (6, 10)]
    L, P, Z = 2, 4, 2
    S = sum(h * w for h, w in spatial_shapes)
    torch.manual_seed(seed)
    value = _bf16(torch.randn(B, S, H, D))
    ref_pts = torch.rand(B, Q, Z, 2)
    if not ref_fp32:
        ref_pts = _bf16(ref_pts)
    offsets = _bf16(torch.randn(B, Q, H, L, P, 2))
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1).reshape(B, Q, H, L, P))
    reference = _msda_reference(
        value, _locations_from_offsets(ref_pts, offsets, spatial_shapes, "pillar"), attn, spatial_shapes
    )
    if packed:
        offsets, attn = _pack_locations(offsets), _pack_weights(attn)
    return (_pack_value(value), ref_pts, offsets, attn), spatial_shapes, reference


# Packed inputs take the tiled geometry on Blackhole, unpacked ones the per-point geometry; their
# runtime args differ, so each is checked through a cache hit.
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize(
    "B,Q,H,counts",
    [
        # 18 tiles: fewer than any grid has cores, one slot per core.
        (3, 70, 2, ([70, 33, 1], [5, 64, 69])),
        # 160 tiles: more than any grid has cores and not a multiple of them, so the
        # cores split into two work groups and the slots are padded per core.
        (2, 300, 8, ([300, 129], [17, 288])),
    ],
)
def test_fused_msda_v2_num_valid_queries_matches_reference_and_reuses_program(device, B, Q, H, counts, packed):
    """Rows below each batch's count match the reference, and changing the counts is a cache hit.

    The counts cut blocks mid-way, at a block boundary and down to one row; Q itself leaves a
    partial trailing block. Calls alternate between no counts and two different count lists, so
    the tile lists in the cached runtime args change on every hit, and each call gets fresh
    tensors, so the buffer addresses change too. A stale list or a stale address computes the
    previous call's rows and misses this call's reference. Rows at or past a count are undefined
    by contract and are not compared.
    """
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    device.cache_entries_counter.reset()

    first, second = counts
    calls = []
    for seed, num_valid_queries in enumerate((None, first, None, second)):
        tensors, spatial_shapes, reference = _num_valid_queries_case(B, Q, H, seed=seed, packed=packed)
        # Kept alive until the end so the allocator cannot hand a later call the same addresses.
        calls.append(([_to_device(t, device) for t in tensors], spatial_shapes, reference, num_valid_queries))

    for (value_t, ref_t, off_t, attn_t), spatial_shapes, reference, num_valid_queries in calls:
        with device.cache_entries_counter.measure():
            out = ttnn.experimental.fused_msda_from_offsets(
                value_t,
                ref_t,
                off_t,
                attn_t,
                spatial_shapes,
                reference_mode="pillar",
                num_valid_queries=num_valid_queries,
            )
        got = ttnn.to_torch(out).to(torch.float32)
        if num_valid_queries is not None:
            with device.cache_entries_counter.measure():
                full_out = ttnn.experimental.fused_msda_from_offsets(
                    value_t,
                    ref_t,
                    off_t,
                    attn_t,
                    spatial_shapes,
                    reference_mode="pillar",
                )
            full_got = ttnn.to_torch(full_out).to(torch.float32)
        for b, n in enumerate(num_valid_queries or [Q] * B):
            _assert_close(reference[b, :n], got[b, :n])
            if num_valid_queries is not None:
                torch.testing.assert_close(full_got[b, :n], got[b, :n], rtol=0, atol=0)
    assert device.cache_entries_counter.total == 1


def _counts_tensor(counts, device=None, layout=ttnn.ROW_MAJOR_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(torch.tensor(counts, dtype=torch.int32), dtype=ttnn.uint32, layout=layout, device=device)


def _case_to_device(tensors, device, ref_fp32=False, memory="dram"):
    value, ref_pts, offsets, attn = tensors
    memory_config = ttnn.L1_MEMORY_CONFIG if memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
    ref_t = ttnn.from_torch(
        ref_pts,
        dtype=ttnn.float32 if ref_fp32 else ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )
    return [
        _to_device_in(value, device, memory),
        ref_t,
        _to_device_in(offsets, device, memory),
        _to_device_in(attn, device, memory),
    ]


def _host_list_msda(tensors, spatial_shapes, counts, Q, memory_config=None, value_layout="auto"):
    """The host-list path on the given device tensors, the tensor path's exact expectation."""
    value_t, ref_t, off_t, attn_t = tensors
    out = ttnn.experimental.fused_msda_from_offsets(
        value_t,
        ref_t,
        off_t,
        attn_t,
        spatial_shapes,
        reference_mode="pillar",
        memory_config=memory_config,
        num_valid_queries=[min(n, Q) for n in counts],
        value_layout=value_layout,
    )
    return ttnn.to_torch(out).to(torch.float32)


def _head_major_from_packed(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Packed (B, S, H*D) -> head-major (B, H, ceil(S/32), 32*D)."""
    B, S, HD = value.shape
    return _head_major_value(value.reshape(B, S, num_heads, HD // num_heads))


def _assert_rows_match(reference, expected, got, counts, Q):
    """Rows below each (clamped) count match the reference, and the host list's rows bit for bit."""
    for b, n in enumerate(counts):
        n = min(n, Q)
        if n == 0:
            continue
        _assert_close(reference[b, :n], got[b, :n])
        torch.testing.assert_close(expected[b, :n], got[b, :n], rtol=0, atol=0)


# The host-list test's shapes, plus counts above Q, which the host list rejects and the kernels
# clamp to Q, and a batch with no valid query. D = 48 takes the reader-only scatter, D = 32 the
# row-major staging (split between reader and writer on Blackhole); float32 reference points
# change the reader's defines and, packed, the tiled geometry's reference tile. D = 48's value
# runs in L1: its 96-byte head stick is not DRAM-aligned on every architecture.
SMALL_COUNTS = (3, 70, 2, ([70, 33, 1], [5, 64, 69], [0, 71, 40]))
LARGE_COUNTS = (2, 300, 8, ([300, 129], [17, 288], [1000, 0]))


@pytest.mark.parametrize(
    "B,Q,H,counts,packed,D,ref_fp32",
    [
        # Fewer tiles than cores, then more and not a multiple of them, each on both geometries.
        (*SMALL_COUNTS, True, 32, False),
        (*SMALL_COUNTS, False, 32, False),
        (*LARGE_COUNTS, True, 32, False),
        (*LARGE_COUNTS, False, 32, False),
        # The reader-only scatter, and float32 reference points.
        (*LARGE_COUNTS, True, 48, False),
        (*LARGE_COUNTS, True, 32, True),
    ],
    ids=["small-packed", "small-unpacked", "large-packed", "large-unpacked", "large-d48", "large-fp32-refs"],
)
def test_fused_msda_v2_num_valid_queries_tensor_matches_host_list(device, B, Q, H, counts, packed, D, ref_fp32):
    """Device counts derive the host list's tiles: rows below each count match it bit for bit.

    One tensor-path program serves every call, with different counts and buffer addresses each
    time; the host-list reference adds a second cache entry. Every call's tensors stay alive, so
    no call can reuse a previous call's addresses and pass on a stale runtime arg.
    """
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    # _num_valid_queries_case always packs value.
    memory = "dram" if _packed_value_fits_dram(H, D) else "l1"
    calls = []
    for seed, num_valid in enumerate(counts):
        tensors, spatial_shapes, reference = _num_valid_queries_case(
            B, Q, H, seed=seed, packed=packed, D=D, ref_fp32=ref_fp32
        )
        calls.append(
            (
                _case_to_device(tensors, device, ref_fp32, memory),
                _counts_tensor(num_valid, device),
                num_valid,
                reference,
            )
        )
    assert len({counts_t.buffer_address() for _, counts_t, _, _ in calls}) == len(calls)

    # The output goes to DRAM whatever value's placement: its half-row writes need only the NoC
    # write alignment, so large-d48 also writes a half with no partner into DRAM.
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    for inputs, counts_t, num_valid, reference in calls:
        out = ttnn.experimental.fused_msda_from_offsets(
            *inputs,
            spatial_shapes,
            reference_mode="pillar",
            memory_config=memory_config,
            num_valid_queries_tensor=counts_t,
        )
        got = ttnn.to_torch(out).to(torch.float32)
        expected = _host_list_msda(inputs, spatial_shapes, num_valid, Q, memory_config)
        _assert_rows_match(reference, expected, got, num_valid, Q)
    assert device.num_program_cache_entries() == 2


@pytest.mark.parametrize("memory", ["dram", "l1"])
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize("B,Q,H,counts", [SMALL_COUNTS, LARGE_COUNTS], ids=["small", "large"])
def test_fused_msda_v2_head_major_value_with_num_valid_queries_tensor(device, B, Q, H, counts, packed, memory):
    """Head-major value under device counts matches packed value under the same counts, bit for bit.

    Spatial cross-attention runs this combination: the counts pick each core's tiles at run
    time, including counts above Q and a batch with none, and the gather, split between reader
    and writer on Blackhole, pair-reads head-major value for exactly those rows.
    """
    for seed, num_valid in enumerate(counts):
        tensors, spatial_shapes, reference = _num_valid_queries_case(B, Q, H, seed=seed, packed=packed)
        packed_value_t, *rest = _case_to_device(tensors, device, memory=memory)
        head_major_t = _to_device_in(_head_major_from_packed(tensors[0], H), device, memory)
        counts_t = _counts_tensor(num_valid, device)

        def run(value_t, value_layout):
            out = ttnn.experimental.fused_msda_from_offsets(
                value_t,
                *rest,
                spatial_shapes,
                reference_mode="pillar",
                num_valid_queries_tensor=counts_t,
                value_layout=value_layout,
            )
            return ttnn.to_torch(out).to(torch.float32)

        _assert_rows_match(reference, run(packed_value_t, "auto"), run(head_major_t, "head_major"), num_valid, Q)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
@pytest.mark.parametrize("value_layout", ["auto", "head_major"])
@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
def test_fused_msda_v2_num_valid_queries_tensor_follows_counts_in_trace(device, packed, value_layout):
    """A trace captured with one set of counts computes the counts written before each replay.

    The first replays raise a count above the captured one, so a trace that kept the captured
    tile list would leave their extra rows uncomputed. The last computes nothing on any core,
    which the host list cannot express, and must still complete.
    """
    B, Q, H = 2, 300, 8

    def laid_out(host_tensors):
        value, *rest = host_tensors
        return (value if value_layout == "auto" else _head_major_from_packed(value, H), *rest)

    captured, *replayed = ([40, 129], [300, 129], [17, 288], [1000, 2], [0, 0])
    tensors, spatial_shapes, _ = _num_valid_queries_case(B, Q, H, seed=0, packed=packed)
    inputs = _case_to_device(laid_out(tensors), device)
    counts_t = _counts_tensor(captured, device)

    def run():
        return ttnn.experimental.fused_msda_from_offsets(
            *inputs,
            spatial_shapes,
            reference_mode="pillar",
            num_valid_queries_tensor=counts_t,
            value_layout=value_layout,
        )

    run()  # compile outside the capture
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        for seed, num_valid in enumerate(replayed, start=1):
            host_tensors, _, reference = _num_valid_queries_case(B, Q, H, seed=seed, packed=packed)
            for host, dev in zip(laid_out(host_tensors), inputs):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(host.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT), dev
                )
            ttnn.copy_host_to_device_tensor(_counts_tensor(num_valid), counts_t)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            if not any(num_valid):
                continue
            got = ttnn.to_torch(out).to(torch.float32)
            expected = _host_list_msda(inputs, spatial_shapes, num_valid, Q, value_layout=value_layout)
            _assert_rows_match(reference, expected, got, num_valid, Q)
    finally:
        ttnn.release_trace(device, trace_id)


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t.to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)


def _to_device_in(t: torch.Tensor, device, memory: str) -> ttnn.Tensor:
    memory_config = ttnn.L1_MEMORY_CONFIG if memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
    return ttnn.from_torch(
        t.to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config
    )


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """Round-trip through bf16 so the reference sees exactly what the device sees."""
    return t.to(torch.bfloat16).to(torch.float32)


def _random_case(B, Q, H, L, P, D, spatial_shapes, seed=0, loc_range=(0.0, 1.0)):
    torch.manual_seed(seed)
    S = sum(h * w for h, w in spatial_shapes)
    value = _bf16(torch.randn(B, S, H, D))
    lo, hi = loc_range
    loc = _bf16(torch.rand(B, Q, H, L, P, 2) * (hi - lo) + lo)
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1).reshape(B, Q, H, L, P))
    return value, loc, attn


def _oob_corner_fraction(loc, spatial_shapes):
    """Fraction of the 4 * L * P bilinear corners that fall outside their level."""
    total = 0
    outside = 0
    for lvl, (h_, w_) in enumerate(spatial_shapes):
        px = loc[:, :, :, lvl, :, 0] * w_ - 0.5
        py = loc[:, :, :, lvl, :, 1] * h_ - 0.5
        x0, y0 = torch.floor(px), torch.floor(py)
        for dx in (0, 1):
            for dy in (0, 1):
                cx, cy = x0 + dx, y0 + dy
                inside = (cx >= 0) & (cx < w_) & (cy >= 0) & (cy < h_)
                outside += int((~inside).sum())
                total += inside.numel()
    return outside / total


def _assert_close(ref: torch.Tensor, got: torch.Tensor, pcc=0.99, atol=None):
    """PCC plus an absolute-error gate.

    PCC alone is blind to a uniformly small output with large local error, which
    is exactly what a boundary or addressing bug in this op produces. The atol
    gate is scaled off the reference magnitude so it stays meaningful when the
    output is near zero.
    """
    assert_with_pcc(ref, got, pcc=pcc)
    if atol is None:
        atol = max(2e-2, 6e-2 * ref.abs().max().item())
    max_err = (ref - got).abs().max().item()
    assert max_err <= atol, f"max abs error {max_err} exceeds {atol} (ref max |x| = {ref.abs().max().item()})"


# ---------------------------------------------------------------------------
# The reference, checked against the model repo's reference (no device needed)
# ---------------------------------------------------------------------------
def test_reference_matches_bevformer_reference():
    """`_msda_reference` must agree with the repo's independent MSDA reference.

    Guards the rest of this file: every device assertion is only as good as the
    reference it compares to.
    """
    from models.experimental.bevformer.reference.ms_deformable_attention import (
        multi_scale_deformable_attn as bevformer_reference,
    )

    torch.manual_seed(0)
    spatial_shapes = [(7, 5), (4, 3)]
    B, Q, H, L, P, D = 2, 6, 3, 2, 4, 16
    S = sum(h * w for h, w in spatial_shapes)

    value = torch.randn(B, S, H, D)
    loc = torch.rand(B, Q, H, L, P, 2)
    attn = torch.softmax(torch.randn(B, Q, H, L * P), dim=-1).reshape(B, Q, H, L, P)

    expected = bevformer_reference(value, torch.tensor(spatial_shapes), loc, attn)
    got = _msda_reference(value, loc, attn, spatial_shapes)
    torch.testing.assert_close(expected, got, rtol=1e-5, atol=1e-5)


def test_locations_from_offsets_reference_matches_bevformer_pillar_layout():
    """The pillar-mode V2 frontend must match BEVFormer's (P // Z, Z) point grouping."""
    torch.manual_seed(0)
    spatial_shapes = [(7, 5), (4, 3)]
    B, Q, H, L, P, Z = 2, 6, 3, 2, 4, 2

    ref = torch.rand(B, Q, Z, 2)
    off = torch.randn(B, Q, H, L, P, 2)

    got = _locations_from_offsets(ref, off, spatial_shapes, "pillar")

    # BEVFormer's formulation, verbatim: offsets viewed as (P // Z, Z) with the
    # reference point broadcasting over the innermost (Z, 2) block.
    normalizer = torch.tensor([[w, h] for h, w in spatial_shapes], dtype=torch.float32)
    expected = ref[:, :, None, None, None, :, :] + (off / normalizer[None, None, None, :, None, :]).view(
        B, Q, H, L, P // Z, Z, 2
    )
    expected = expected.view(B, Q, H, L, P, 2)
    torch.testing.assert_close(expected, got, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# V1 correctness
# ---------------------------------------------------------------------------
_V1_ONE_LEVEL = [(10, 10)]
_V1_FOUR_LEVELS = [(16, 20), (8, 10), (4, 5), (2, 3)]


# Pairwise cover of the boundary values, not the full product. Value is in DRAM
# (from_torch's default) and the output follows it. H=8 with D=16 puts output heads 32 B apart,
# a multiple of every architecture's NoC write alignment; value is canonical,
# one page per head, so no read starts inside a page.
@pytest.mark.parametrize(
    "B,Q,H,spatial_shapes,P,D,align_corners",
    [
        (1, 33, 1, _V1_ONE_LEVEL, 1, 32, False),
        (1, 1, 1, _V1_FOUR_LEVELS, 4, 16, False),
        (1, 1, 1, _V1_ONE_LEVEL, 8, 64, True),
        (1, 31, 8, _V1_FOUR_LEVELS, 1, 32, True),
        (2, 31, 8, _V1_ONE_LEVEL, 4, 64, False),
        (2, 32, 1, _V1_FOUR_LEVELS, 8, 16, True),
        (2, 256, 8, _V1_ONE_LEVEL, 8, 32, False),
        (1, 256, 1, _V1_FOUR_LEVELS, 1, 64, True),
        (1, 32, 8, _V1_ONE_LEVEL, 4, 32, False),
        (2, 33, 8, _V1_FOUR_LEVELS, 4, 64, True),
        (2, 1, 1, _V1_ONE_LEVEL, 1, 16, False),
        (1, 31, 1, _V1_ONE_LEVEL, 8, 16, False),
        (1, 33, 1, _V1_ONE_LEVEL, 8, 16, False),
        (1, 256, 1, _V1_ONE_LEVEL, 4, 16, False),
        (1, 32, 1, _V1_ONE_LEVEL, 1, 64, False),
        (1, 1, 8, _V1_ONE_LEVEL, 1, 32, False),
        (1, 33, 8, _V1_FOUR_LEVELS, 4, 16, True),
        (2, 31, 8, _V1_ONE_LEVEL, 8, 16, False),
    ],
)
def test_fused_msda_v1(device, B, Q, H, spatial_shapes, P, D, align_corners):
    L = len(spatial_shapes)
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)
    ref = _msda_reference(value, loc, attn, spatial_shapes, align_corners=align_corners)

    out = ttnn.experimental.fused_msda(
        _to_device(value, device),
        _to_device(loc, device),
        _to_device(attn, device),
        spatial_shapes,
        align_corners=align_corners,
    )
    got = ttnn.to_torch(out).to(torch.float32)

    assert list(got.shape) == [B, Q, H * D]
    _assert_close(ref, got)


@pytest.mark.parametrize("value_memory", ["dram", "l1"])
@pytest.mark.parametrize(
    "Q,H,D",
    [(33, 1, 16), (33, 8, 16), (49, 2, 48), (17, 1, 48), (33, 8, 32), (17, 4, 64)],
    ids=["d16_h1", "d16_h8", "d48_h2", "d48_h1", "d32_h8", "d64_h4"],
)
def test_fused_msda_output_placement(device, value_memory, Q, H, D):
    """The writer sends every 16-value half of a tile row straight to its byte
    offset in the output page, so only the output buffer's NoC write alignment
    constrains where they land, whatever value's placement. D=16 and D=48 end on a half with no partner, D=64 spans two
    d-tiles, and Q leaves a tail block. L1 and DRAM output of the same inputs
    must be bit-identical: the writer only moves bytes.
    """
    spatial_shapes = _V1_FOUR_LEVELS
    L, P = len(spatial_shapes), 4
    value, loc, attn = _random_case(1, Q, H, L, P, D, spatial_shapes)
    ref = _msda_reference(value, loc, attn, spatial_shapes)

    value_t = _to_device_in(value, device, value_memory)
    loc_t = _to_device(loc, device)
    attn_t = _to_device(attn, device)
    outputs = {}
    for output_memory, memory_config in (("l1", ttnn.L1_MEMORY_CONFIG), ("dram", ttnn.DRAM_MEMORY_CONFIG)):
        out = ttnn.experimental.fused_msda(value_t, loc_t, attn_t, spatial_shapes, memory_config=memory_config)
        assert out.memory_config().buffer_type == memory_config.buffer_type
        outputs[output_memory] = ttnn.to_torch(out).to(torch.float32)
        assert list(outputs[output_memory].shape) == [1, Q, H * D]
        _assert_close(ref, outputs[output_memory])

    torch.testing.assert_close(outputs["l1"], outputs["dram"], rtol=0, atol=0)


@pytest.mark.parametrize("B,Q,H,L,P,D", [(1, 64, 2, 2, 4, 32)])
def test_fused_msda_v1_packed_equivalence(device, B, Q, H, L, P, D):
    """The packed rank-4 input forms must give the same answer as the canonical ones."""
    spatial_shapes = [(12, 9), (6, 5)]
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)

    value_t = _to_device(value, device)
    canonical = ttnn.to_torch(
        ttnn.experimental.fused_msda(value_t, _to_device(loc, device), _to_device(attn, device), spatial_shapes)
    ).to(torch.float32)
    packed = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            value_t,
            _to_device(_pack_locations(loc), device),
            _to_device(_pack_weights(attn), device),
            spatial_shapes,
        )
    ).to(torch.float32)

    # Same values, same order of accumulation — this must be exact, not merely close.
    torch.testing.assert_close(canonical, packed, rtol=0, atol=0)
    _assert_close(_msda_reference(value, loc, attn, spatial_shapes), packed)


@pytest.mark.parametrize("memory", ["dram", "l1"])
@pytest.mark.parametrize(
    "B,Q,H,L,P,D,spatial_shapes",
    [
        (1, 64, 2, 2, 4, 32, [(12, 9), (6, 5)]),
        # More (batch, head, query-block) units than cores, so every core runs
        # several blocks of the split gather back to back at D = 64.
        (2, 900, 8, 4, 4, 64, [(16, 20), (8, 10), (4, 5), (2, 3)]),
        # Heads 32 B apart inside a value page, read by the tile-scatter gather.
        (1, 33, 8, 2, 4, 16, [(12, 9), (6, 5)]),
    ],
    ids=["small_d32", "multiblock_d64", "small_d16"],
)
def test_fused_msda_v1_packed_value_equivalence(device, B, Q, H, L, P, D, spatial_shapes, memory):
    """Packed rank-3 value (B, S, H*D) must match canonical (B, S, H, D) exactly.

    The two layouts are the same bytes; only the page size and the (page,
    offset) addressing of every gathering RISC differ. A mismatch is an
    addressing bug, not bf16 rounding.
    """
    if memory == "dram" and not _packed_value_fits_dram(H, D):
        pytest.skip(
            f"packed value with {D * 2} B heads in DRAM is rejected on this device "
            f"(test_fused_msda_rejects_unaligned_packed_value_heads)"
        )
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)

    loc_t = _to_device(_pack_locations(loc), device)
    attn_t = _to_device(_pack_weights(attn), device)
    canonical = ttnn.to_torch(
        ttnn.experimental.fused_msda(_to_device_in(value, device, memory), loc_t, attn_t, spatial_shapes)
    ).to(torch.float32)
    packed = ttnn.to_torch(
        ttnn.experimental.fused_msda(_to_device_in(_pack_value(value), device, memory), loc_t, attn_t, spatial_shapes)
    ).to(torch.float32)

    torch.testing.assert_close(canonical, packed, rtol=0, atol=0)
    _assert_close(_msda_reference(value, loc, attn, spatial_shapes), packed)


@pytest.mark.parametrize("align_corners", [False, True])
def test_fused_msda_v1_grid_space(device, align_corners):
    """locations_in_grid_space=True on 2*loc-1 must equal the [0, 1] path on loc.

    Both align_corners settings are exercised: the two grid-space rows of the
    coordinate table are separate host-side constants, so one can be wrong
    while the other is right.
    """
    spatial_shapes = [(12, 9), (6, 5)]
    B, Q, H, L, P, D = 1, 40, 2, 2, 4, 32
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)
    grid = _bf16(2.0 * loc - 1.0)

    value_t, attn_t = _to_device(value, device), _to_device(attn, device)
    unit = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            value_t, _to_device(loc, device), attn_t, spatial_shapes, align_corners=align_corners
        )
    ).to(torch.float32)
    grid_space = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            value_t,
            _to_device(grid, device),
            attn_t,
            spatial_shapes,
            align_corners=align_corners,
            locations_in_grid_space=True,
        )
    ).to(torch.float32)

    # 2*loc-1 is not exact in bf16, so this is close-but-not-equal by construction.
    _assert_close(unit, grid_space, pcc=0.999)
    _assert_close(
        _msda_reference(value, grid, attn, spatial_shapes, align_corners=align_corners, locations_in_grid_space=True),
        grid_space,
    )


# ---------------------------------------------------------------------------
# V1 numerics: the cases where a boundary or addressing bug actually shows up
# ---------------------------------------------------------------------------
def test_fused_msda_v1_exact_pixel_centres(device):
    """Locations on pixel centres must return the stored value, not a blend."""
    spatial_shapes = [(8, 8)]
    B, Q, H, L, P, D = 1, 8, 1, 1, 1, 16
    torch.manual_seed(0)
    value = _bf16(torch.randn(B, 64, H, D))

    # align_corners=False: pixel centre i sits at normalized (i + 0.5) / size.
    xs = torch.tensor([(i % 8 + 0.5) / 8.0 for i in range(Q)])
    ys = torch.tensor([(i % 8 + 0.5) / 8.0 for i in range(Q)])
    loc = _bf16(torch.stack([xs, ys], dim=-1).reshape(1, Q, 1, 1, 1, 2))
    attn = torch.ones(B, Q, H, L, P)

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device(value, device), _to_device(loc, device), _to_device(attn, device), spatial_shapes
        )
    ).to(torch.float32)

    expected = torch.stack([value[0, (i % 8) * 8 + (i % 8), 0, :] for i in range(Q)]).unsqueeze(0)
    _assert_close(expected, out, pcc=0.999, atol=5e-2)


def test_fused_msda_v1_out_of_bounds_is_exactly_zero(device):
    """Fully out-of-range samples must contribute exactly 0 (padding_mode="zeros")."""
    spatial_shapes = [(8, 8)]
    B, Q, H, L, P, D = 1, 32, 2, 1, 4, 32
    torch.manual_seed(0)
    value = _bf16(torch.randn(B, 64, H, D))
    # Well outside [0, 1] in both directions, so all four corners are out of range.
    loc = _bf16(torch.full((B, Q, H, L, P, 2), 5.0))
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1).reshape(B, Q, H, L, P))

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device(value, device), _to_device(loc, device), _to_device(attn, device), spatial_shapes
        )
    ).to(torch.float32)

    assert torch.count_nonzero(out) == 0, f"{torch.count_nonzero(out)} nonzero entries in a fully OOB result"


@pytest.mark.parametrize("corner", [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_fused_msda_v1_corners_and_edges(device, corner):
    """Samples exactly on the feature-map corners straddle the border: the in-bounds
    corners must still contribute, the out-of-bounds ones must not."""
    spatial_shapes = [(6, 9)]
    B, Q, H, L, P, D = 1, 32, 1, 1, 1, 16
    torch.manual_seed(0)
    value = _bf16(torch.randn(B, 54, H, D))
    loc = _bf16(torch.full((B, Q, H, L, P, 2), 0.0))
    loc[..., 0] = corner[0]
    loc[..., 1] = corner[1]
    attn = torch.ones(B, Q, H, L, P)

    ref = _msda_reference(value, loc, attn, spatial_shapes)
    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device(value, device), _to_device(loc, device), _to_device(attn, device), spatial_shapes
        )
    ).to(torch.float32)
    _assert_close(ref, out, pcc=0.99, atol=5e-2)


def test_fused_msda_v1_zero_attention_weights(device):
    """Zero weights must zero the output exactly, regardless of the sampled values."""
    spatial_shapes = [(10, 10), (5, 5)]
    B, Q, H, L, P, D = 1, 33, 2, 2, 4, 32
    value, loc, _ = _random_case(B, Q, H, L, P, D, spatial_shapes)
    attn = torch.zeros(B, Q, H, L, P)

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device(value, device), _to_device(loc, device), _to_device(attn, device), spatial_shapes
        )
    ).to(torch.float32)
    assert torch.count_nonzero(out) == 0


def test_fused_msda_v1_attention_concentrated_on_one_point(device):
    """A one-hot weight must reproduce that single bilinear sample."""
    spatial_shapes = [(10, 10), (5, 5)]
    B, Q, H, L, P, D = 1, 32, 2, 2, 4, 32
    value, loc, _ = _random_case(B, Q, H, L, P, D, spatial_shapes)
    attn = torch.zeros(B, Q, H, L, P)
    attn[:, :, :, 1, 2] = 1.0

    ref = _msda_reference(value, loc, attn, spatial_shapes)
    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device(value, device), _to_device(loc, device), _to_device(attn, device), spatial_shapes
        )
    ).to(torch.float32)
    _assert_close(ref, out, pcc=0.999, atol=5e-2)


def test_fused_msda_v1_partially_out_of_bounds(device):
    """Locations spread well beyond [0, 1] exercise every mix of valid/invalid corners."""
    spatial_shapes = [(16, 20), (8, 10)]
    B, Q, H, L, P, D = 2, 64, 4, 2, 4, 32
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes, loc_range=(-0.3, 1.3))
    ref = _msda_reference(value, loc, attn, spatial_shapes)

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device(value, device), _to_device(loc, device), _to_device(attn, device), spatial_shapes
        )
    ).to(torch.float32)
    _assert_close(ref, out)


@pytest.mark.parametrize("D", [32, 64])
@pytest.mark.parametrize(
    "memory, Q", [(m, q) for m, s in SPLIT_ROW.items() for q in (129, s, s + 1, 32 + s, 32 + s + 1)]
)
def test_fused_msda_v1_masks_out_of_bounds_corners(device, Q, D, memory):
    """An out-of-bounds corner must contribute nothing, element by element.

    Regression gate for the out-of-bounds mask. The gathers skip an
    out-of-bounds corner and leave its input slot holding an earlier point's
    value stick, so the zero weight compute gives that corner
    (`msda_geometry.hpp::mask_corner_weights`) is the only thing standing
    between a stale slot and the sum.

    It asserts a per-element error ratio on top of `_assert_close` because a
    stale row is itself a plausible sampled value: correlation degrades more
    slowly than the individual elements do, so the ratio is what stays
    meaningful as shapes shrink and the output magnitude falls.

    The case is built so that both halves matter: enough corners are outside
    the map to exercise the mask, and enough are inside that a slot the reader
    skips has just been written with a real value stick by an earlier point.
    Every Q leaves a partial trailing block, whose rows >= v_rows the gathers
    skip as well. With the gather split between the two data-movement RISCs at
    ``SPLIT_ROW[memory]``, v_rows = SPLIT_ROW[memory] and SPLIT_ROW[memory] + 1
    sit on the split: the writer gathers nothing, or exactly one row.
    """
    spatial_shapes = [(16, 20), (8, 10), (4, 5), (2, 3)]
    B, H, L, P = 1, 4, 4, 4
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes, loc_range=(-0.25, 1.25))

    oob = _oob_corner_fraction(loc, spatial_shapes)
    assert 0.25 < oob < 0.75, f"test no longer exercises the mask: {oob:.2%} of corners are out of bounds"

    ref = _msda_reference(value, loc, attn, spatial_shapes)
    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device_in(value, device, memory),
            _to_device(_pack_locations(loc), device),
            _to_device(_pack_weights(attn), device),
            spatial_shapes,
        )
    ).to(torch.float32)

    _assert_close(ref, out)

    abs_tol = max(2e-2, 6e-2 * ref.abs().max().item())
    high_error_ratio = ((ref - out).abs() > abs_tol).float().mean().item()
    assert high_error_ratio <= 0.01, (
        f"{high_error_ratio:.2%} of output elements are off by more than {abs_tol:.4f}. "
        "An out-of-bounds corner kept a live weight and multiplied a stale input slot"
    )


@pytest.mark.parametrize("memory", ["l1", "dram"])
@pytest.mark.parametrize("path", ["v1", "v2_tiled", "v2_tiled_32"])
@pytest.mark.parametrize("bad", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_fused_msda_nonfinite_value_does_not_leak_into_out_of_bounds_corners(device, bad, path, memory):
    """A non-finite value stick gathered for one query must not reach a later one.

    On Blackhole the gathers leave an out-of-bounds corner's input slot holding
    whatever an earlier point gathered there, and only compute's zero weight
    cancels it. The contiguous work split gives the first core several query
    blocks, and its input CB holds two points' corners, so every later point on
    that core reuses the slots query 0's points wrote. Query 0 samples a
    non-finite pixel in bounds; every other query has all its corners out of
    bounds and must come out exactly 0. "v2_tiled" (L * P = 16) is the tiled
    geometry and, with D = 32, the split gather; "v2_tiled_32" (L * P = 32) adds
    the second group of 16 points.
    """
    grid = device.compute_with_storage_grid_size()
    spatial_shapes = [(8, 8)]
    B, H, D = 1, 1, 32
    L, P = {"v1": (1, 4), "v2_tiled": (1, 16), "v2_tiled_32": (1, 32)}[path]
    Q = 96 * grid.x * grid.y
    torch.manual_seed(0)
    value = _bf16(torch.randn(B, 64, H, D))
    value[0, 3 * 8 + 3] = bad
    attn = torch.full((B, Q, H, L, P), 1.0 / P)
    value_t = _to_device_in(_pack_value(value), device, memory)
    attn_t = _to_device(_pack_weights(attn), device)

    if path == "v1":
        loc = torch.full((B, Q, H, L, P, 2), 5.0)
        loc[0, 0] = 3.5 / 8
        out = ttnn.experimental.fused_msda(value_t, _to_device(_pack_locations(loc), device), attn_t, spatial_shapes)
    else:
        R = 4
        ref_pts = torch.full((B, Q, R, 2), 5.0)
        ref_pts[0, 0] = 3.5 / 8
        offsets = torch.zeros(B, Q, H, L, P, 2)
        out = ttnn.experimental.fused_msda_from_offsets(
            value_t,
            _to_device(ref_pts, device),
            _to_device(_pack_locations(offsets), device),
            attn_t,
            spatial_shapes,
            reference_mode="pillar",
        )
    out = ttnn.to_torch(out).to(torch.float32)

    assert not out[0, 0].isfinite().all(), "query 0 did not sample the non-finite pixel; the case no longer applies"
    assert torch.count_nonzero(out[0, 1:]) == 0, (
        f"{torch.count_nonzero(out[0, 1:])} nonzero outputs outside query 0: a skipped slot "
        "kept an earlier non-finite stick and its zero weight did not cancel it"
    )


@pytest.mark.parametrize("L, P", [(4, 4), (4, 8), (1, 8)], ids=["LP16", "LP32", "LP8"])
@pytest.mark.parametrize(
    "memory, Q, D",
    [(m, 32 + q, 32) for m, s in SPLIT_ROW.items() for q in (s, s + 1)] + [("dram", 32 + SPLIT_ROW["dram"], 16)],
)
def test_fused_msda_v2_tiled_masks_out_of_bounds_corners(device, memory, Q, D, L, P):
    """An out-of-bounds corner must contribute nothing on the tiled geometry too.

    V2 with packed inputs takes the tiled geometry on Blackhole, whose block
    solve and corner tiles are separate code from the per-point path; L * P are
    BEVFormer-base's (32, two groups of 16 points), exactly one group (16) and
    BEVFormer-small and -tiny's (8, half a group). Each Q is a full query block
    followed by a tail block with v_rows = SPLIT_ROW[memory] or
    SPLIT_ROW[memory] + 1 on the split, where the writer gathers nothing or one
    row. D = 16 takes the tile-scatter gather.
    """
    spatial_shapes = [(16, 20), (8, 10), (4, 5), (2, 3)][:L]
    B, H, R = 1, 1, 4
    value, ref_pts, offsets, attn = _v2_case(B, Q, H, L, P, D, R, spatial_shapes)
    ref_pts = _bf16(ref_pts * 1.5 - 0.25)
    loc = _locations_from_offsets(ref_pts, offsets, spatial_shapes, "pillar")

    oob = _oob_corner_fraction(loc, spatial_shapes)
    assert 0.25 < oob < 0.75, f"test no longer exercises the mask: {oob:.2%} of corners are out of bounds"

    ref = _msda_reference(value, loc, attn, spatial_shapes)
    out = ttnn.to_torch(
        ttnn.experimental.fused_msda_from_offsets(
            _to_device_in(_pack_value(value), device, memory),
            _to_device(ref_pts, device),
            _to_device(_pack_locations(offsets), device),
            _to_device(_pack_weights(attn), device),
            spatial_shapes,
            reference_mode="pillar",
        )
    ).to(torch.float32)

    _assert_close(ref, out)

    abs_tol = max(2e-2, 6e-2 * ref.abs().max().item())
    high_error_ratio = ((ref - out).abs() > abs_tol).float().mean().item()
    assert high_error_ratio <= 0.01, (
        f"{high_error_ratio:.2%} of output elements are off by more than {abs_tol:.4f}. "
        "An out-of-bounds corner kept a live weight and multiplied a stale input slot"
    )


# Raw bf16 bit patterns: converting a float32 NaN to bf16 drops its sign.
@pytest.mark.parametrize("bad", [0x7FC0, 0xFFC0, 0x7F80, 0xFF80], ids=["nan", "neg_nan", "inf", "neg_inf"])
@pytest.mark.parametrize("path", ["v1", "v2", "v2_tiled", "v2_tiled_32"])
def test_fused_msda_nonfinite_sampling_location_contributes_nothing(device, path, bad):
    """A sample whose x, y or both is NaN or infinite must add nothing.

    `msda_geometry.hpp::corner_word` clamps the floored corner before its bounds
    tests, so such a sample's corner word must have every flag clear: the
    gathers issue no read for it and compute zeroes its weights, although the
    attention weight handed to the op is live. The reference drops the poisoned
    samples instead, and the other points of the same queries must be
    untouched: on the tiled path a poisoned offset is one column of the block
    matmul. "v2" (unpacked inputs) takes the per-point geometry, "v2_tiled"
    (L * P = 16) and "v2_tiled_32" (L * P = 32) the tiled one on Blackhole.
    """
    B, Q, H, D = 1, 64, 2, 32
    if path == "v2_tiled":
        spatial_shapes, P, R = [(16, 20), (8, 10), (4, 5), (2, 3)], 4, 4
    elif path == "v2_tiled_32":
        spatial_shapes, P, R = [(16, 20), (8, 10), (4, 5), (2, 3)], 8, 4
    else:
        spatial_shapes, P, R = [(16, 20), (8, 10)], 4, 2
    L = len(spatial_shapes)

    torch.manual_seed(0)
    poisoned = torch.rand(B, Q, H, L, P) < 0.1
    # 0: x only, 1: y only, 2: both
    which = torch.randint(0, 3, (B, Q, H, L, P))
    bad_x = poisoned & (which != 1)
    bad_y = poisoned & (which != 0)

    if path == "v1":
        value, coords, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)
        loc = coords
    else:
        value, ref_pts, coords, attn = _v2_case(B, Q, H, L, P, D, R, spatial_shapes)
        loc = _locations_from_offsets(ref_pts, coords, spatial_shapes, "pillar")
    sent = coords.to(torch.bfloat16)
    bits = sent.view(torch.int16)
    bad_int16 = bad - 0x10000 if bad >= 0x8000 else bad
    bits[..., 0][bad_x] = bad_int16
    bits[..., 1][bad_y] = bad_int16

    ref = _msda_reference(value, loc, attn.masked_fill(poisoned, 0.0), spatial_shapes)
    if path == "v1":
        out = ttnn.experimental.fused_msda(
            _to_device(value, device),
            _to_device(_pack_locations(sent), device),
            _to_device(_pack_weights(attn), device),
            spatial_shapes,
        )
    else:
        packed = path != "v2"
        out = ttnn.experimental.fused_msda_from_offsets(
            _to_device(_pack_value(value), device),
            _to_device(ref_pts, device),
            _to_device(_pack_locations(sent) if packed else sent, device),
            _to_device(_pack_weights(attn) if packed else attn, device),
            spatial_shapes,
            reference_mode="pillar",
        )
    out = ttnn.to_torch(out).to(torch.float32)

    assert out.isfinite().all(), f"{(~out.isfinite()).sum()} non-finite outputs: a poisoned sample kept a live weight"
    _assert_close(ref, out)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_fused_msda_v1_repeated_calls_hit_program_cache(device, memory):
    """Repeated calls reuse one program and stay exact while value moves.

    With the split gather (Blackhole, D % 32 == 0) the writer RISC also reads
    value, through its own address argument. Alternating two value tensors, with
    an allocation in between so the second lands elsewhere, catches a stale
    address on either RISC after a cache hit, and identical calls must stay
    bit-exact. The handshake semaphores are not under test here: dispatch writes
    their initial values with every launch.
    """
    spatial_shapes = [(16, 20), (8, 10), (4, 5), (2, 3)]
    B, Q, H, L, P, D = 2, 900, 8, 4, 4, 32
    value_a, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes, seed=1, loc_range=(-0.25, 1.25))
    value_b = _bf16(torch.randn_like(value_a))

    loc_t = _to_device(_pack_locations(loc), device)
    attn_t = _to_device(_pack_weights(attn), device)
    value_a_t = _to_device_in(_pack_value(value_a), device, memory)
    spacer = _to_device_in(torch.zeros(1, 4096), device, memory)
    value_b_t = _to_device_in(_pack_value(value_b), device, memory)

    def run(value_t):
        return ttnn.to_torch(ttnn.experimental.fused_msda(value_t, loc_t, attn_t, spatial_shapes)).to(torch.float32)

    entries_before = device.num_program_cache_entries()
    first = {"a": run(value_a_t), "b": run(value_b_t)}
    _assert_close(_msda_reference(value_a, loc, attn, spatial_shapes), first["a"])
    _assert_close(_msda_reference(value_b, loc, attn, spatial_shapes), first["b"])
    for i in range(10):
        key = "a" if i % 2 == 0 else "b"
        out = run(value_a_t if key == "a" else value_b_t)
        torch.testing.assert_close(out, first[key], rtol=0, atol=0)
    assert device.num_program_cache_entries() - entries_before == 1
    ttnn.deallocate(spacer)


# ---------------------------------------------------------------------------
# Realistic shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,B,Q,H,L,P,D,spatial_shapes",
    [
        # BEVFormer spatial cross-attention: 4 FPN levels at nuScenes resolution,
        # 8 heads x 32 head_dim = 256 embed dims, 4 points per (head, level).
        ("bevformer_sca", 1, 256, 8, 4, 4, 32, [(116, 200), (58, 100), (29, 50), (15, 25)]),
        # DINO/UniAD-style deformable decoder: 900 object queries over the same pyramid.
        ("dino_decoder", 1, 900, 8, 4, 4, 32, [(29, 50), (15, 25), (8, 13), (4, 7)]),
    ],
)
def test_fused_msda_v1_realistic_shapes(device, name, B, Q, H, L, P, D, spatial_shapes):
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)
    ref = _msda_reference(value, loc, attn, spatial_shapes)

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device(value, device),
            _to_device(_pack_locations(loc), device),
            _to_device(_pack_weights(attn), device),
            spatial_shapes,
        )
    ).to(torch.float32)
    _assert_close(ref, out)


# ---------------------------------------------------------------------------
# V2 equivalence — the test that says "V2 is a frontend, not a new operator"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reference_mode,R", [("level", 4), ("pillar", 4), ("pillar", 2), ("pillar", 1)])
@pytest.mark.parametrize("Q", [32, 33])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("align_corners", [False, True])
def test_fused_msda_v1_v2_equivalence(device, reference_mode, R, Q, packed, align_corners):
    spatial_shapes = [(16, 20), (8, 10), (4, 5), (2, 3)]
    B, H, L, P, D = 2, 4, 4, 4, 32
    S = sum(h * w for h, w in spatial_shapes)

    torch.manual_seed(0)
    value = _bf16(torch.randn(B, S, H, D))
    ref_pts = _bf16(torch.rand(B, Q, R, 2))
    # Offsets are raw pixel units; a few pixels of jitter is the realistic range.
    offsets = _bf16(torch.randn(B, Q, H, L, P, 2) * 2.0)
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1).reshape(B, Q, H, L, P))

    loc = _bf16(
        _locations_from_offsets(
            ref_pts,
            offsets,
            spatial_shapes,
            reference_mode,
            align_corners=align_corners,
        )
    )

    value_t = _to_device(value, device)
    attn_t = _to_device(_pack_weights(attn) if packed else attn, device)

    v1 = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            value_t,
            _to_device(_pack_locations(loc) if packed else loc, device),
            attn_t,
            spatial_shapes,
            align_corners=align_corners,
        )
    ).to(torch.float32)
    v2 = ttnn.to_torch(
        ttnn.experimental.fused_msda_from_offsets(
            value_t,
            _to_device(ref_pts, device),
            _to_device(_pack_locations(offsets) if packed else offsets, device),
            attn_t,
            spatial_shapes,
            reference_mode=reference_mode,
            align_corners=align_corners,
        )
    ).to(torch.float32)

    # V1 sees locations already rounded to bf16; V2 forms the position on the
    # SFPU from the unrounded reference point and offset, so the two differ by
    # the rounding of the intermediate location only.
    _assert_close(v1, v2, pcc=0.999)
    _assert_close(_msda_reference(value, loc, attn, spatial_shapes, align_corners=align_corners), v2, pcc=0.99)


# BEVFormer's spatial cross-attention (L*P*2 = 64 values, 128 B per head) and temporal
# self-attention with its two BEV maps as two levels in "level" mode (L*P*2 = 16 values, 32 B per
# head: below Blackhole's 64 B DRAM alignment, so L1).
@pytest.mark.parametrize(
    "H, L, P, R, spatial_shapes, reference_mode, memory",
    [
        (8, 4, 8, 4, [(29, 50), (15, 25), (8, 13), (4, 7)], "pillar", "dram"),
        (8, 4, 8, 4, [(29, 50), (15, 25), (8, 13), (4, 7)], "pillar", "l1"),
        (8, 2, 4, 2, [(50, 50), (50, 50)], "level", "l1"),
    ],
    ids=["sca_dram", "sca_l1", "tsa_l1"],
)
def test_fused_msda_v2_heads_packed_offsets(device, H, L, P, R, spatial_shapes, reference_mode, memory):
    """(B, Q, H*L*P*2) offsets, a Linear's output as is, must match (B, Q, H, L*P*2) exactly."""
    B, Q, D = 1, 96, 32
    S = sum(h * w for h, w in spatial_shapes)
    torch.manual_seed(0)
    value = _bf16(torch.randn(B, S, H * D))
    ref_pts = _bf16(torch.rand(B, Q, R, 2))
    offsets = _bf16(torch.randn(B, Q, H, L * P * 2) * 2.0)
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1))

    def run(offsets_t):
        return ttnn.to_torch(
            ttnn.experimental.fused_msda_from_offsets(
                _to_device(value, device),
                _to_device(ref_pts, device),
                offsets_t,
                _to_device(attn, device),
                spatial_shapes,
                reference_mode=reference_mode,
            )
        )

    per_head = run(_to_device_in(offsets, device, memory))
    heads_packed = run(_to_device_in(offsets.reshape(B, Q, H * L * P * 2), device, memory))
    torch.testing.assert_close(per_head, heads_packed, rtol=0, atol=0)


@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_fused_msda_v1_heads_packed_locations(device, memory):
    """V1 shares the staging: (B, Q, H*L*P*2) locations must match (B, Q, H, L*P*2) exactly, and each
    form keeps a program cache entry of its own."""
    spatial_shapes = [(29, 50), (15, 25), (8, 13), (4, 7)]
    B, Q, H, L, P, D = 2, 64, 8, 4, 8, 32
    S = sum(h * w for h, w in spatial_shapes)
    torch.manual_seed(0)
    value = _bf16(torch.randn(B, S, H * D))
    loc = _bf16(torch.rand(B, Q, H, L * P * 2))
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1))

    def run(loc_t):
        return ttnn.to_torch(
            ttnn.experimental.fused_msda(_to_device(value, device), loc_t, _to_device(attn, device), spatial_shapes)
        )

    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    per_head = run(_to_device_in(loc, device, memory))
    entries = device.num_program_cache_entries()
    heads_packed = run(_to_device_in(loc.reshape(B, Q, H * L * P * 2), device, memory))
    torch.testing.assert_close(per_head, heads_packed, rtol=0, atol=0)
    assert device.num_program_cache_entries() == entries + 1
    entries = device.num_program_cache_entries()
    torch.testing.assert_close(run(_to_device_in(loc.reshape(B, Q, H * L * P * 2), device, memory)), heads_packed)
    assert device.num_program_cache_entries() == entries


def test_fused_msda_v2_heads_packed_offsets_rejects_unaligned_heads(device, expect_error):
    """16 B per head is below DRAM's alignment: the reader could not address a head's stick."""
    B, Q, H, L, P, D = 1, 32, 16, 1, 4, 32
    spatial_shapes = [(50, 50)]
    with expect_error(RuntimeError, "heads-packed sampling_offsets"):
        ttnn.experimental.fused_msda_from_offsets(
            _to_device(torch.zeros(B, 2500, H * D), device),
            _to_device(torch.zeros(B, Q, 1, 2), device),
            _to_device_in(torch.zeros(B, Q, H * L * P * 2), device, "dram"),
            _to_device(torch.zeros(B, Q, H, L * P), device),
            spatial_shapes,
            reference_mode="pillar",
        )


def test_fused_msda_v2_bevformer_pillar_shape(device):
    """BEVFormer spatial cross-attention in pillar mode: 4 z-anchors per BEV query."""
    spatial_shapes = [(29, 50), (15, 25), (8, 13), (4, 7)]
    B, Q, H, L, P, D, Z = 6, 128, 8, 4, 4, 32, 4
    S = sum(h * w for h, w in spatial_shapes)

    torch.manual_seed(0)
    value = _bf16(torch.randn(B, S, H, D))
    ref_pts = _bf16(torch.rand(B, Q, Z, 2))
    offsets = _bf16(torch.randn(B, Q, H, L, P, 2) * 2.0)
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1).reshape(B, Q, H, L, P))
    # NOT _bf16(): the V2 op reads bf16 reference points and offsets but forms the
    # location on the SFPU without rounding it and samples at that position, so rounding it
    # here would compare the kernel against a coarser sampler than it is. The
    # difference is not cosmetic at these shapes -- bf16 quantization of a
    # reference point near 0.5 is ~0.002, and W=50 turns that into ~0.1 px of
    # sampling offset. V1 is the case that must be compared against a rounded
    # location, because a rounded location is literally its input.
    loc = _locations_from_offsets(ref_pts, offsets, spatial_shapes, "pillar")

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda_from_offsets(
            _to_device(_pack_value(value), device),
            _to_device(ref_pts, device),
            _to_device(_pack_locations(offsets), device),
            _to_device(_pack_weights(attn), device),
            spatial_shapes,
            reference_mode="pillar",
        )
    ).to(torch.float32)
    _assert_close(_msda_reference(value, loc, attn, spatial_shapes), out)


@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize(
    "reference_mode, spatial_shapes, Q, P, R",
    [
        # BEVFormer's decoder and temporal self-attention: L = R = 1, where level and pillar
        # mode coincide.
        ("level", [(200, 200)], 256, 4, 1),
        # The encoder's spatial cross-attention: four z-anchors, a partial query tile.
        ("pillar", [(29, 50), (15, 25), (8, 13), (4, 7)], 33, 8, 4),
        ("pillar", [(29, 50), (15, 25), (8, 13), (4, 7)], 33, 4, 4),
    ],
    ids=["decoder-200x200", "pillar-R4-LP32", "pillar-R4-LP16"],
)
def test_fused_msda_v2_float32_reference_points(device, reference_mode, spatial_shapes, Q, P, R, packed):
    """float32 reference points, kept unrounded into px, next to the same points in bf16. Each run
    is checked against a reference sampling at its own points, so a program built for one
    reference dtype and reused for the other fails. Packed inputs take the tiled geometry on
    Blackhole, where the FPU reads a float32 point as three bf16 parts; unpacked ones the
    per-point geometry, where the SFPU reads it whole."""
    B, H, D = 1, 8, 32
    L = len(spatial_shapes)
    S = sum(h * w for h, w in spatial_shapes)

    torch.manual_seed(0)
    value = _bf16(torch.randn(B, S, H, D))
    ref_pts = torch.rand(B, Q, R, 2)
    offsets = _bf16(torch.randn(B, Q, H, L, P, 2) * 2.0)
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1).reshape(B, Q, H, L, P))

    def expected(points):
        return _msda_reference(
            value, _locations_from_offsets(points, offsets, spatial_shapes, reference_mode), attn, spatial_shapes
        )

    def run(ref_dtype):
        return ttnn.to_torch(
            ttnn.experimental.fused_msda_from_offsets(
                _to_device(_pack_value(value), device),
                ttnn.from_torch(ref_pts, dtype=ref_dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT),
                _to_device(_pack_locations(offsets) if packed else offsets, device),
                _to_device(_pack_weights(attn) if packed else attn, device),
                spatial_shapes,
                reference_mode=reference_mode,
            )
        ).to(torch.float32)

    fp32_out = run(ttnn.float32)
    bf16_out = run(ttnn.bfloat16)
    _assert_close(expected(ref_pts), fp32_out, pcc=0.9999)
    _assert_close(expected(_bf16(ref_pts)), bf16_out, pcc=0.9999)
    # Against the unrounded points, the bf16 run carries the points' rounding as well.
    fp32_error = (fp32_out - expected(ref_pts)).abs().mean()
    bf16_error = (bf16_out - expected(ref_pts)).abs().mean()
    assert fp32_error < 0.5 * bf16_error, f"float32 points {fp32_error:.5f}, bf16 points {bf16_error:.5f}"


def test_fused_msda_v2_one_pixel_offset(device):
    """An offset of +1 feature-map pixel must sample the next pixel exactly.

    Width 5 with align_corners spaces pixels by 1/4, so a scale of 1/5 lands
    between pixels and this check fails.
    """
    extent = 5
    spatial_shapes = [(extent, extent)]
    B, Q, heads, L, P, D = 1, 2, 1, 1, 1, 16

    value = torch.zeros(B, extent * extent, heads, D)
    for y in range(extent):
        for x in range(extent):
            value[0, y * extent + x, 0, :] = float(y * extent + x)
    value = _bf16(value)

    # align_corners places pixel i at i/(extent-1).
    ref = _bf16(torch.full((B, Q, 1, 2), 1.0 / (extent - 1)))
    offsets = torch.zeros(B, Q, heads, L, P, 2)
    offsets[0, 0, 0, 0, 0, 0] = 1.0
    offsets[0, 1, 0, 0, 0, 1] = 1.0

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda_from_offsets(
            _to_device(value, device),
            _to_device(ref, device),
            _to_device(_bf16(offsets), device),
            _to_device(torch.ones(B, Q, heads, L, P), device),
            spatial_shapes,
            reference_mode="level",
            align_corners=True,
        )
    ).to(torch.float32)

    expected = torch.zeros(B, Q, heads * D)
    expected[0, 0, :] = float(1 * extent + 2)  # (x=2, y=1)
    expected[0, 1, :] = float(2 * extent + 1)  # (x=1, y=2)
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-2)


def test_fused_msda_program_cache_hit_readdresses(device):
    """Same-spec calls share one cached program, so the second call's buffers are not the ones compiled in.

    A hit replays that program; only override_runtime_arguments rewrites the addresses. Both
    allocations stay live and carry different data: a stale address still points at the first call
    and cannot match the second reference.
    """
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    device.cache_entries_counter.reset()

    spatial_shapes = [(8, 8)]
    B, Q, H, L, P, D = 1, 32, 1, 1, 4, 32

    def allocate(seed):
        value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes, seed=seed)
        tensors = (_to_device(value, device), _to_device(loc, device), _to_device(attn, device))
        return tensors, _msda_reference(value, loc, attn, spatial_shapes)

    (value1, loc1, attn1), ref1 = allocate(0)
    (value2, loc2, attn2), ref2 = allocate(1)
    assert value1.buffer_address() != value2.buffer_address()
    assert loc1.buffer_address() != loc2.buffer_address()
    assert attn1.buffer_address() != attn2.buffer_address()
    assert (ref1 - ref2).abs().max().item() > 1e-2

    def dispatch(value_t, loc_t, attn_t):
        with device.cache_entries_counter.measure():
            out_t = ttnn.experimental.fused_msda(value_t, loc_t, attn_t, spatial_shapes)
        return out_t, ttnn.to_torch(out_t).to(torch.float32)

    out1, got1 = dispatch(value1, loc1, attn1)
    _assert_close(ref1, got1)
    entries_after_miss = device.cache_entries_counter.total
    assert entries_after_miss >= 1

    out2, got2 = dispatch(value2, loc2, attn2)
    _assert_close(ref2, got2)
    assert device.cache_entries_counter.total == entries_after_miss
    assert out1.buffer_address() != out2.buffer_address()


def _v2_case(B, Q, H, L, P, D, R, spatial_shapes, seed=0):
    torch.manual_seed(seed)
    S = sum(h * w for h, w in spatial_shapes)
    value = _bf16(torch.randn(B, S, H, D))
    ref_pts = _bf16(torch.rand(B, Q, R, 2))
    offsets = _bf16(torch.randn(B, Q, H, L, P, 2) * 2.0)
    attn = _bf16(torch.softmax(torch.randn(B, Q, H, L * P), dim=-1).reshape(B, Q, H, L, P))
    return value, ref_pts, offsets, attn


def _check_reference_points_reused(device, Q, reference_mode, R, D, ref_dtype, L, P, spatial_shapes, packed=True):
    H = 3
    grid = device.compute_with_storage_grid_size()
    B = 2 * grid.x * grid.y + 1

    value, ref_pts, offsets, attn = _v2_case(B, Q, H, L, P, D, R, spatial_shapes)
    loc = _locations_from_offsets(ref_pts, offsets, spatial_shapes, reference_mode)

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda_from_offsets(
            _to_device(_pack_value(value), device),
            ttnn.from_torch(ref_pts, dtype=ref_dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT),
            _to_device(_pack_locations(offsets) if packed else offsets, device),
            _to_device(_pack_weights(attn) if packed else attn, device),
            spatial_shapes,
            reference_mode=reference_mode,
        )
    ).to(torch.float32)
    _assert_close(_msda_reference(value, loc, attn, spatial_shapes), out)


@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked"])
@pytest.mark.parametrize("ref_dtype", [ttnn.bfloat16, ttnn.float32], ids=["ref_bf16", "ref_fp32"])
@pytest.mark.parametrize("D", [32, 64])
@pytest.mark.parametrize("reference_mode,R", [("level", 2), ("pillar", 2)])
@pytest.mark.parametrize("Q", [17, 33])
def test_fused_msda_v2_reference_points_reused_across_heads(device, Q, reference_mode, R, D, ref_dtype, packed):
    """reference_points staged once per query block must be re-staged whenever the block changes.

    Tiles are ordered batch, query block, head, and each core takes a contiguous
    run, so the V2 reader keeps the head-invariant reference points of the
    previous tile when (b, q_start) repeats. B = 2 * cores + 1 gives every core
    at least two batches' worth of tiles, and the first tiles_per_batch cores
    one more, so their runs start part-way through a batch (for Q = 33 also
    part-way through a query block) and every run reuses the points across
    heads and re-stages them inside the run:
      * Q = 17, one query block per batch: consecutive tiles across a batch
        boundary share q_start = 0, so a cache that ignores the batch reuses
        the wrong batch's points.
      * Q = 33: the run crosses a query-block boundary, and goes from a full
        block into a v_rows = 1 tail block.
    float32 reference points use a 4-byte-per-coordinate arena, reused the same way.
    Packed inputs (L * P = 8) take the tiled geometry on Blackhole, where the reference tile is
    rewritten only when the query block changes and re-pushed unchanged otherwise, and block
    t+1's points are staged while block t is still being gathered; unpacked ones the per-point
    geometry.
    """
    _check_reference_points_reused(device, Q, reference_mode, R, D, ref_dtype, 2, 4, [(8, 10), (4, 5)], packed)


@pytest.mark.parametrize("ref_dtype", [ttnn.bfloat16, ttnn.float32], ids=["ref_bf16", "ref_fp32"])
@pytest.mark.parametrize("P", [4, 8], ids=["LP16", "LP32"])
@pytest.mark.parametrize("Q", [17, 33])
def test_fused_msda_v2_tiled_reference_points_reused_across_heads(device, Q, P, ref_dtype):
    """The same reuse and re-staging on the tiled geometry with four levels, one and two groups
    of 16 points."""
    _check_reference_points_reused(device, Q, "pillar", 2, 32, ref_dtype, 4, P, [(8, 10), (4, 5), (2, 3), (1, 2)])


@pytest.mark.parametrize(
    "L, P, R, reference_mode",
    [
        (1, 2, 1, "level"),
        (1, 4, 1, "level"),
        (1, 8, 4, "pillar"),
        (2, 6, 2, "level"),
        (2, 8, 4, "pillar"),
        (3, 3, 3, "level"),
        (1, 15, 15, "pillar"),
        (3, 8, 4, "pillar"),
        (2, 12, 4, "pillar"),
        (4, 8, 4, "pillar"),
        (4, 8, 4, "level"),
        (2, 16, 4, "pillar"),
        (1, 32, 4, "pillar"),
    ],
)
@pytest.mark.parametrize("ref_dtype", [ttnn.bfloat16, ttnn.float32], ids=["ref_bf16", "ref_fp32"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_fused_msda_v2_tiled_points_per_block(device, L, P, R, reference_mode, ref_dtype, align_corners):
    """Every point of a block lands in its own lane and group on the tiled geometry (packed V2,
    2 <= L * P <= 32, Blackhole).

    Point J is lane J % 16 of group J // 16, and its level is J // P. L * P = 2, 4, 8 and 12
    leave lanes of the only group unused, 9 and 15 (with R = 15, the most the reference tile
    holds) are odd, 24 leaves the second group part-filled, (2, 12) puts level 1 across both
    groups and (4, 8) changes level inside the second. A point read from the wrong lane, group,
    level or reference point samples elsewhere. Offsets that end short of a 64 B boundary bring
    padding into columns no point uses. B = cores + 1 gives every core several blocks, so each
    block's geometry is solved while the previous one is gathered, and Q = 33 adds a v_rows = 1
    tail block.
    """
    spatial_shapes = [(29, 50), (15, 25), (8, 13), (4, 7)][:L]
    grid = device.compute_with_storage_grid_size()
    B, Q, H, D = grid.x * grid.y + 1, 33, 2, 32
    torch.manual_seed(0)
    value, ref_pts, offsets, attn = _v2_case(B, Q, H, L, P, D, R, spatial_shapes)
    ref_pts = torch.rand(B, Q, R, 2)
    if ref_dtype == ttnn.bfloat16:
        ref_pts = _bf16(ref_pts)
    loc = _locations_from_offsets(ref_pts, offsets, spatial_shapes, reference_mode, align_corners=align_corners)

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda_from_offsets(
            _to_device(_pack_value(value), device),
            ttnn.from_torch(ref_pts, dtype=ref_dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT),
            _to_device(_pack_locations(offsets), device),
            _to_device(_pack_weights(attn), device),
            spatial_shapes,
            reference_mode=reference_mode,
            align_corners=align_corners,
        )
    ).to(torch.float32)
    _assert_close(_msda_reference(value, loc, attn, spatial_shapes, align_corners=align_corners), out, pcc=0.999)


@pytest.mark.parametrize(
    "L, P, R, reference_mode",
    [
        (1, 1, 1, "level"),
        (3, 11, 3, "level"),
        (4, 12, 4, "pillar"),
        (1, 16, 16, "pillar"),
    ],
    ids=["LP1", "LP33", "LP48", "R16"],
)
@pytest.mark.parametrize("ref_dtype", [ttnn.bfloat16, ttnn.float32], ids=["ref_bf16", "ref_fp32"])
def test_fused_msda_v2_packed_outside_tiled_range(device, L, P, R, reference_mode, ref_dtype):
    """Packed V2 the tiled geometry does not take runs the per-point geometry.

    L * P = 1 is below the block-ahead solve's minimum, 33 and 48 are past the 32 a block's
    attention row holds, and R = 16 is past the 15 reference points a reference tile holds.
    """
    spatial_shapes = [(29, 50), (15, 25), (8, 13), (4, 7)][:L]
    B, Q, H, D = 2, 33, 2, 32
    value, ref_pts, offsets, attn = _v2_case(B, Q, H, L, P, D, R, spatial_shapes)
    ref_pts = torch.rand(B, Q, R, 2)
    if ref_dtype == ttnn.bfloat16:
        ref_pts = _bf16(ref_pts)
    loc = _locations_from_offsets(ref_pts, offsets, spatial_shapes, reference_mode)

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda_from_offsets(
            _to_device(_pack_value(value), device),
            ttnn.from_torch(ref_pts, dtype=ref_dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT),
            _to_device(_pack_locations(offsets), device),
            _to_device(_pack_weights(attn), device),
            spatial_shapes,
            reference_mode=reference_mode,
        )
    ).to(torch.float32)
    _assert_close(_msda_reference(value, loc, attn, spatial_shapes), out, pcc=0.999)


@pytest.mark.parametrize("packed", [True, False], ids=["tiled", "per_point"])
def test_fused_msda_v2_float32_reference_point_sub_ulp(device, packed):
    """A float32 reference point keeps all 24 significant bits into the sampling position.

    Pixel x of a 256-wide level holds K * (x % 2), and every point lies a fraction f in
    [1/64, 1/16) right of an even pixel centre, so each output is K * f. A point that lost its
    last 8 significant bits would move by up to 256 * 2^-16 = 2^-8 px, K / 2^8 at the output,
    against output roundings of at most K / 2^12. The tiled geometry reads the point as three
    bf16 tiles, so this fails if any part is dropped. B = cores + 1 runs several blocks per core,
    with the reference tile reused across heads.
    """
    grid = device.compute_with_storage_grid_size()
    B, Q, H, L, P, D, R = grid.x * grid.y + 1, 33, 2, 1, 4, 32, 1
    W, K = 256, 16.0
    spatial_shapes = [(2, W)]
    x = torch.arange(2 * W) % W
    value = (K * (x % 2)).to(torch.float32).view(1, 2 * W, 1, 1).expand(B, 2 * W, H, D).contiguous()

    torch.manual_seed(0)
    x0 = 2 * torch.randint(64, 127, (B, Q), dtype=torch.float64)
    f = 1 / 64 + torch.rand(B, Q, dtype=torch.float64) * (1 / 16 - 1 / 64)
    ref_pts = torch.empty(B, Q, R, 2, dtype=torch.float32)
    ref_pts[..., 0, 0] = ((x0 + 0.5 + f) / W).to(torch.float32)
    ref_pts[..., 0, 1] = 0.25  # py = 0: row 0, dy = 0
    # The sampling position of the float32 point, exactly.
    expected = K * (ref_pts[..., 0, 0].to(torch.float64) * W - 0.5 - x0)

    offsets = torch.zeros(B, Q, H, L, P, 2)
    attn = torch.full((B, Q, H, L, P), 1.0 / P)
    out = ttnn.to_torch(
        ttnn.experimental.fused_msda_from_offsets(
            _to_device(_pack_value(value), device),
            ttnn.from_torch(ref_pts, dtype=ttnn.float32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT),
            _to_device(_pack_locations(offsets) if packed else offsets, device),
            _to_device(_pack_weights(attn) if packed else attn, device),
            spatial_shapes,
            reference_mode="level",
        )
    ).to(torch.float64)

    err = (out - expected.unsqueeze(-1)).abs().max().item()
    assert err <= K / 2**11, f"max error {err} exceeds {K / 2**11}: the reference point lost bits"


@pytest.mark.timeout(300)
@pytest.mark.parametrize("P", [4, 8], ids=["LP16", "LP32"])
def test_fused_msda_v2_repeated_calls_hit_program_cache(device, P):
    """A cached V2 program re-reads reference_points on every launch.

    The reader skips re-staging reference points for consecutive tiles of one
    query block; that state must not survive into the next launch. Alternating
    two reference_points tensors on a shape where cores run several tiles
    catches both a stale buffer binding and a cache that outlives the launch.
    """
    spatial_shapes = [(16, 20), (8, 10), (4, 5), (2, 3)]
    B, Q, H, L, D, R = 2, 900, 8, 4, 32, 4
    value, ref_a, offsets, attn = _v2_case(B, Q, H, L, P, D, R, spatial_shapes, seed=1)
    ref_b = _bf16(torch.rand_like(ref_a))

    value_t = _to_device(_pack_value(value), device)
    off_t = _to_device(_pack_locations(offsets), device)
    attn_t = _to_device(_pack_weights(attn), device)
    ref_a_t = _to_device(ref_a, device)
    ref_b_t = _to_device(ref_b, device)

    def run(ref_t):
        return ttnn.to_torch(
            ttnn.experimental.fused_msda_from_offsets(
                value_t, ref_t, off_t, attn_t, spatial_shapes, reference_mode="level"
            )
        ).to(torch.float32)

    entries_before = device.num_program_cache_entries()
    first = {"a": run(ref_a_t), "b": run(ref_b_t)}
    for key, ref in (("a", ref_a), ("b", ref_b)):
        loc = _locations_from_offsets(ref, offsets, spatial_shapes, "level")
        _assert_close(_msda_reference(value, loc, attn, spatial_shapes), first[key])
    for i in range(10):
        key = "a" if i % 2 == 0 else "b"
        out = run(ref_a_t if key == "a" else ref_b_t)
        torch.testing.assert_close(out, first[key], rtol=0, atol=0)
    assert device.num_program_cache_entries() - entries_before == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _valid_inputs(device, B=1, Q=8, H=2, L=2, P=4, D=32, spatial_shapes=((6, 5), (3, 2))):
    spatial_shapes = [tuple(s) for s in spatial_shapes]
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)
    return (
        _to_device(value, device),
        _to_device(loc, device),
        _to_device(attn, device),
        spatial_shapes,
    )


@pytest.mark.parametrize("D", [8, 24, 40])
def test_fused_msda_rejects_head_dim_not_multiple_of_16(device, expect_error, D):
    value_t, loc_t, attn_t, shapes = _valid_inputs(device, D=D)
    with expect_error(RuntimeError, "multiple of 16"):
        ttnn.experimental.fused_msda(value_t, loc_t, attn_t, shapes)


def test_fused_msda_rejects_packed_value_not_divisible_by_heads(device, expect_error):
    value_t, loc_t, attn_t, shapes = _valid_inputs(device, H=2, D=32)
    # Last dim 49 is odd, so it cannot be split into the 2 heads attention_weights
    # declares and head_dim is not recoverable. (An even non-H*D last dim such as 48
    # would divide cleanly by 2 and be rejected by the head_dim % 16 check instead.)
    packed = _to_device(torch.randn(1, 36, 49), device)  # S = 6*5 + 3*2 = 36
    with expect_error(RuntimeError, "divisible"):
        ttnn.experimental.fused_msda(packed, loc_t, attn_t, shapes)


def test_fused_msda_rejects_spatial_shape_key_mismatch(device, expect_error):
    value_t, loc_t, attn_t, _ = _valid_inputs(device)
    with expect_error(RuntimeError, "must equal sum of"):
        # Same level count, different total key count than `value` was built for.
        ttnn.experimental.fused_msda(value_t, loc_t, attn_t, [(6, 5), (3, 3)])


def test_fused_msda_rejects_too_many_levels(device, expect_error):
    shapes = [(2, 2)] * 9
    value_t, loc_t, attn_t, _ = _valid_inputs(device, L=9, spatial_shapes=shapes)
    with expect_error(RuntimeError, "feature levels"):
        ttnn.experimental.fused_msda(value_t, loc_t, attn_t, shapes)


@pytest.mark.parametrize("shapes", [[(257, 4)], [(4, 257)]], ids=["h", "w"])
def test_fused_msda_rejects_spatial_shape_beyond_extent_limit(device, expect_error, shapes):
    """A feature map past the tested extent must be refused, not run.

    256 per axis is the largest extent the op is tested at; past it nothing pins
    the corner word or the geometry, so the op rejects the shape.

    Both axes are parametrized because one `TT_FATAL` tests both, so a refactor
    that drops the `w` half would otherwise go unnoticed.
    """
    value_t, loc_t, attn_t, _ = _valid_inputs(device, L=1, spatial_shapes=shapes)
    with expect_error(RuntimeError, "per-axis limit"):
        ttnn.experimental.fused_msda(value_t, loc_t, attn_t, shapes)


def test_fused_msda_accepts_spatial_shape_at_extent_limit(device):
    """A 256-pixel axis, the largest the op accepts, must sample correctly.

    The companion to the rejection test above: it pins the boundary itself, so a
    `<` where the code means `<=` fails here rather than silently narrowing the op.
    """
    spatial_shapes = [(256, 4)]
    B, Q, H, L, P, D = 1, 8, 1, 1, 4, 32
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)
    ref = _msda_reference(value, loc, attn, spatial_shapes)

    out = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device(value, device), _to_device(loc, device), _to_device(attn, device), spatial_shapes
        )
    ).to(torch.float32)
    _assert_close(ref, out)


def test_fused_msda_rejects_level_count_mismatch(device, expect_error):
    value_t, loc_t, attn_t, _ = _valid_inputs(device)
    with expect_error(RuntimeError, "levels"):
        # attention_weights says L=2, spatial_shapes says L=1 (and S still matches).
        ttnn.experimental.fused_msda(value_t, loc_t, attn_t, [(6, 6)])


def test_fused_msda_rejects_tile_layout(device, expect_error):
    value_t, loc_t, attn_t, shapes = _valid_inputs(device)
    tiled = ttnn.to_layout(attn_t, ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "ROW_MAJOR"):
        ttnn.experimental.fused_msda(value_t, loc_t, tiled, shapes)


def test_fused_msda_rejects_unaligned_packed_value_heads(device, expect_error):
    """Packed value reads head h at byte offset h*D*2 inside one page. Where that is
    not a multiple of the DRAM alignment the op must refuse rather than read across
    heads; the same value in L1 is accepted (test_fused_msda_v1_packed_value_equivalence).
    """
    H, D = 8, 16
    if _packed_value_fits_dram(H, D):
        pytest.skip(f"this device's {ttnn.device.get_dram_alignment()} B DRAM alignment permits {D * 2} B heads")
    spatial_shapes = [(6, 6), (3, 3)]
    value, loc, attn = _random_case(1, 8, H, len(spatial_shapes), 2, D, spatial_shapes)
    # Output in L1, so the only placement the op can refuse is value's.
    with expect_error(RuntimeError, "packed value"):
        ttnn.experimental.fused_msda(
            _to_device_in(_pack_value(value), device, "dram"),
            _to_device(loc, device),
            _to_device(attn, device),
            spatial_shapes,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


def test_fused_msda_from_offsets_rejects_box_reference_points(device, expect_error):
    value_t, off_t, attn_t, shapes = _valid_inputs(device)
    ref_t = _to_device(torch.rand(1, 8, 2, 4), device)
    with expect_error(RuntimeError, "box form"):
        ttnn.experimental.fused_msda_from_offsets(value_t, ref_t, off_t, attn_t, shapes, reference_mode="level")


def test_fused_msda_from_offsets_rejects_level_mode_rank_mismatch(device, expect_error):
    value_t, off_t, attn_t, shapes = _valid_inputs(device)  # L = 2
    ref_t = _to_device(torch.rand(1, 8, 3, 2), device)  # R = 3
    with expect_error(RuntimeError, "one reference point per level"):
        ttnn.experimental.fused_msda_from_offsets(value_t, ref_t, off_t, attn_t, shapes, reference_mode="level")


def test_fused_msda_from_offsets_rejects_pillar_mode_indivisible_points(device, expect_error):
    value_t, off_t, attn_t, shapes = _valid_inputs(device, P=4)
    ref_t = _to_device(torch.rand(1, 8, 3, 2), device)  # P=4 is not divisible by R=3
    with expect_error(RuntimeError, "divisible"):
        ttnn.experimental.fused_msda_from_offsets(value_t, ref_t, off_t, attn_t, shapes, reference_mode="pillar")


@pytest.mark.parametrize("tensor", ["value", "sampling_offsets", "attention_weights"])
def test_fused_msda_from_offsets_rejects_float32_inputs(device, expect_error, tensor):
    """Only reference_points may be float32; the readers read value, sampling_offsets and
    attention_weights as bf16."""
    value_t, off_t, attn_t, shapes = _valid_inputs(device)
    ref_t = _to_device(torch.rand(1, 8, 2, 2), device)
    inputs = {"value": value_t, "sampling_offsets": off_t, "attention_weights": attn_t}
    inputs[tensor] = ttnn.from_torch(
        ttnn.to_torch(inputs[tensor]).float(), dtype=ttnn.float32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    with expect_error(RuntimeError, "must be BFLOAT16,"):
        ttnn.experimental.fused_msda_from_offsets(
            inputs["value"],
            ref_t,
            inputs["sampling_offsets"],
            inputs["attention_weights"],
            shapes,
            reference_mode="level",
        )


@pytest.mark.parametrize(
    "num_valid_queries,message",
    [
        ([], "empty list"),
        ([8], "one count per batch"),
        ([8, 9], "exceeds num_queries"),
        ([0, 0], "no query"),
    ],
)
def test_fused_msda_from_offsets_rejects_bad_num_valid_queries(device, expect_error, num_valid_queries, message):
    value_t, off_t, attn_t, shapes = _valid_inputs(device, B=2)
    ref_t = _to_device(torch.rand(2, 8, 2, 2), device)
    with expect_error(RuntimeError, message):
        ttnn.experimental.fused_msda_from_offsets(
            value_t, ref_t, off_t, attn_t, shapes, reference_mode="pillar", num_valid_queries=num_valid_queries
        )


@pytest.mark.parametrize(
    "counts,dtype,layout,on_device,message",
    [
        ([8, 8], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, True, "must be UINT32"),
        ([8, 8], ttnn.uint32, ttnn.TILE_LAYOUT, True, "must be ROW_MAJOR"),
        ([8, 8], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, False, "expect device tensors"),
        ([8], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, True, "counts in the last dim"),
        ([[8], [8]], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, True, "counts in the last dim"),
    ],
    ids=["int32", "tile", "host", "too-few", "not-last-dim"],
)
def test_fused_msda_from_offsets_rejects_bad_num_valid_queries_tensor(
    device, expect_error, counts, dtype, layout, on_device, message
):
    value_t, off_t, attn_t, shapes = _valid_inputs(device, B=2)
    ref_t = _to_device(torch.rand(2, 8, 2, 2), device)
    counts_t = ttnn.from_torch(
        torch.tensor(counts, dtype=torch.int32), dtype=dtype, layout=layout, device=device if on_device else None
    )
    with expect_error(RuntimeError, message):
        ttnn.experimental.fused_msda_from_offsets(
            value_t, ref_t, off_t, attn_t, shapes, reference_mode="pillar", num_valid_queries_tensor=counts_t
        )


def test_fused_msda_from_offsets_rejects_both_count_forms(device, expect_error):
    value_t, off_t, attn_t, shapes = _valid_inputs(device, B=2)
    ref_t = _to_device(torch.rand(2, 8, 2, 2), device)
    with expect_error(RuntimeError, "not both"):
        ttnn.experimental.fused_msda_from_offsets(
            value_t,
            ref_t,
            off_t,
            attn_t,
            shapes,
            reference_mode="pillar",
            num_valid_queries=[8, 8],
            num_valid_queries_tensor=_counts_tensor([8, 8], device),
        )


def test_fused_msda_from_offsets_rejects_unknown_reference_mode(device, expect_error):
    value_t, off_t, attn_t, shapes = _valid_inputs(device)
    ref_t = _to_device(torch.rand(1, 8, 2, 2), device)
    with expect_error(RuntimeError, "reference_mode"):
        ttnn.experimental.fused_msda_from_offsets(value_t, ref_t, off_t, attn_t, shapes, reference_mode="boxes")


# ---------------------------------------------------------------------------
# Head-major value (B, H, ceil(S/32), 32*D)
# ---------------------------------------------------------------------------
# Must equal fused_msda_value_layout::HEAD_MAJOR_KEYS_PER_PAGE in
# device/kernels/msda_value_layout.hpp (the tile height).
HEAD_MAJOR_KEYS_PER_PAGE = 32

# W = 33 > 32, so every row of level 0 crosses a page boundary and some
# horizontal corner pairs straddle two pages; S % 32 != 0 leaves a padded last
# page per head.
HEAD_MAJOR_SHAPES = [(9, 33), (5, 7)]


def _head_major_value(value: torch.Tensor) -> torch.Tensor:
    """(B, S, H, D) -> (B, H, ceil(S/32), 32*D), the last page of each head zero-padded."""
    B, S, H, D = value.shape
    pages = -(-S // HEAD_MAJOR_KEYS_PER_PAGE)
    hm = torch.zeros(B, H, pages * HEAD_MAJOR_KEYS_PER_PAGE, D, dtype=value.dtype)
    hm[:, :, :S] = value.permute(0, 2, 1, 3)
    return hm.reshape(B, H, pages, HEAD_MAJOR_KEYS_PER_PAGE * D)


def _value_stick_aligned(head_dim: int, memory: str) -> bool:
    """Do D-wide sticks sit on legal NoC source addresses in this buffer type?

    Head-major value stores keys back to back, so key k of a page starts at
    k * D * 2 bytes; the op refuses value whose stick is not a multiple of the
    buffer's alignment (see test_fused_msda_rejects_unaligned_head_major_stick).
    """
    alignment = ttnn.device.get_dram_alignment() if memory == "dram" else ttnn.device.get_l1_alignment()
    return (head_dim * 2) % alignment == 0


def _check_head_major_matches_packed(device, B, Q, H, D, memory):
    spatial_shapes = HEAD_MAJOR_SHAPES
    value, loc, attn = _random_case(B, Q, H, 2, 4, D, spatial_shapes, loc_range=(-0.2, 1.2))

    loc_t = _to_device(_pack_locations(loc), device)
    attn_t = _to_device(_pack_weights(attn), device)
    packed = ttnn.to_torch(
        ttnn.experimental.fused_msda(_to_device_in(_pack_value(value), device, memory), loc_t, attn_t, spatial_shapes)
    ).to(torch.float32)
    head_major = ttnn.to_torch(
        ttnn.experimental.fused_msda(
            _to_device_in(_head_major_value(value), device, memory),
            loc_t,
            attn_t,
            spatial_shapes,
            value_layout="head_major",
        )
    ).to(torch.float32)

    torch.testing.assert_close(packed, head_major, rtol=0, atol=0)
    _assert_close(_msda_reference(value, loc, attn, spatial_shapes), head_major)


@pytest.mark.parametrize("D", [32, 64])
# Q = 32 + SPLIT_ROW + 1 leaves a tail block of SPLIT_ROW + 1 rows: the writer
# gathers, and pair-reads, exactly one row of it.
@pytest.mark.parametrize(
    "memory, B, Q, H",
    [(m, B, Q, H) for m, s in SPLIT_ROW.items() for B, Q, H in [(1, 33, 2), (2, 129, 8), (1, 32 + s + 1, 2)]],
)
def test_fused_msda_v1_head_major_value_equivalence(device, B, Q, H, D, memory):
    """Head-major value must match packed value exactly, pair reads and their fallbacks included.

    Locations spill past [0, 1] so every mix of in- and out-of-bounds corners
    occurs: a pair is one read only when both corners are in bounds and in the
    same page; otherwise each in-bounds corner is one read and an out-of-bounds slot is
    skipped (zeroed off Blackhole).
    """
    _check_head_major_matches_packed(device, B, Q, H, D, memory)


@pytest.mark.parametrize("memory", ["dram", "l1"])
@pytest.mark.parametrize("D", [16, 48])
def test_fused_msda_v1_head_major_value_scatter_path(device, D, memory):
    """D % 32 != 0 takes the tile-scatter reader, which reads head-major value one corner at a time.

    One head keeps the packed comparison legal at every D: packed value with H > 1
    needs D*2 to be a multiple of the buffer's alignment. The head-major stick
    still has to meet that alignment at any H.
    """
    if not _value_stick_aligned(D, memory):
        pytest.skip(f"a {D * 2}-byte stick is not aligned for {memory} on this device")
    _check_head_major_matches_packed(device, 2, 65, 1, D, memory)


def test_fused_msda_rejects_unaligned_head_major_stick(device, expect_error):
    head_dim = 16
    if _value_stick_aligned(head_dim, "dram"):
        pytest.skip(f"this device's {ttnn.device.get_dram_alignment()}-byte DRAM alignment permits head_dim {head_dim}")
    _, loc_t, attn_t, shapes = _valid_inputs(device, H=1, D=head_dim)
    value, _, _ = _random_case(1, 8, 1, 2, 4, head_dim, shapes)
    with expect_error(RuntimeError, "must be a multiple of value's"):
        ttnn.experimental.fused_msda(
            _to_device(_head_major_value(value), device), loc_t, attn_t, shapes, value_layout="head_major"
        )


def test_fused_msda_rejects_unknown_value_layout(device, expect_error):
    value_t, loc_t, attn_t, shapes = _valid_inputs(device, H=2, D=32)
    with expect_error(RuntimeError, 'value_layout must be "auto"'):
        ttnn.experimental.fused_msda(value_t, loc_t, attn_t, shapes, value_layout="packed")


@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_fused_msda_v2_head_major_value_equivalence(device, memory):
    """The V2 entry point takes head-major value too, with the same answer as packed."""
    spatial_shapes = HEAD_MAJOR_SHAPES
    value, ref_pts, offsets, attn = _v2_case(2, 65, 8, 2, 4, 32, 4, spatial_shapes)

    def run(value_t, value_layout):
        return ttnn.to_torch(
            ttnn.experimental.fused_msda_from_offsets(
                value_t,
                _to_device(ref_pts, device),
                _to_device(_pack_locations(offsets), device),
                _to_device(_pack_weights(attn), device),
                spatial_shapes,
                reference_mode="pillar",
                value_layout=value_layout,
            )
        ).to(torch.float32)

    packed = run(_to_device_in(_pack_value(value), device, memory), "auto")
    head_major = run(_to_device_in(_head_major_value(value), device, memory), "head_major")
    torch.testing.assert_close(packed, head_major, rtol=0, atol=0)


@pytest.mark.parametrize("in_memory", ["dram", "l1"])
@pytest.mark.parametrize("out_memory", ["dram", "l1"])
@pytest.mark.parametrize("D", [32, 64])
@pytest.mark.parametrize("H", [1, 8])
@pytest.mark.parametrize("B,S", [(1, 1), (1, 32), (2, 33), (2, 332)])
def test_fused_msda_head_major_value_op(device, B, S, H, D, out_memory, in_memory):
    """The producer must place every key of every head where head-major says it lives."""
    torch.manual_seed(0)
    value = _bf16(torch.randn(B, S, H, D))
    memory_config = ttnn.L1_MEMORY_CONFIG if out_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
    tiled = ttnn.from_torch(
        _pack_value(value).to(torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG if in_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG,
    )

    out_t = ttnn.experimental.fused_msda_head_major_value(tiled, H, memory_config=memory_config)
    assert out_t.layout == ttnn.ROW_MAJOR_LAYOUT
    assert out_t.memory_config().buffer_type == memory_config.buffer_type
    out = ttnn.to_torch(out_t).to(torch.float32)

    want = _head_major_value(value)
    assert list(out.shape) == list(want.shape)
    # Rows past S hold the input's tile padding, which the op never reads.
    keys = out.reshape(B, H, -1, D)[:, :, :S]
    torch.testing.assert_close(keys, want.reshape(B, H, -1, D)[:, :, :S], rtol=0, atol=0)


def test_fused_msda_head_major_value_op_feeds_fused_msda(device):
    """value_proj's tiled output -> producer -> op must equal the packed ROW_MAJOR path exactly."""
    spatial_shapes = HEAD_MAJOR_SHAPES
    B, Q, H, D = 2, 65, 8, 32
    value, ref_pts, offsets, attn = _v2_case(B, Q, H, 2, 4, D, 4, spatial_shapes)
    tiled = ttnn.from_torch(_pack_value(value).to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

    def run(value_t, value_layout):
        return ttnn.to_torch(
            ttnn.experimental.fused_msda_from_offsets(
                value_t,
                _to_device(ref_pts, device),
                _to_device(_pack_locations(offsets), device),
                _to_device(_pack_weights(attn), device),
                spatial_shapes,
                reference_mode="pillar",
                value_layout=value_layout,
            )
        ).to(torch.float32)

    packed = run(ttnn.to_layout(tiled, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG), "auto")
    head_major = run(
        ttnn.experimental.fused_msda_head_major_value(tiled, H, memory_config=ttnn.L1_MEMORY_CONFIG), "head_major"
    )
    torch.testing.assert_close(packed, head_major, rtol=0, atol=0)


def test_fused_msda_rejects_head_major_value_with_wrong_page_count(device, expect_error):
    _, loc_t, attn_t, shapes = _valid_inputs(device, H=2, D=32)
    # S = 36 needs ceil(36/32) = 2 pages per head, not 3.
    bad = _to_device(torch.randn(1, 2, 3, 32 * 32), device)
    with expect_error(RuntimeError, "is not head-major"):
        ttnn.experimental.fused_msda(bad, loc_t, attn_t, shapes, value_layout="head_major")


def test_fused_msda_head_major_value_needs_value_layout(device, expect_error):
    """Without value_layout="head_major" the op reads rank-4 value as canonical (B, S, H, D) and rejects the key count."""
    _, loc_t, attn_t, shapes = _valid_inputs(device, H=2, D=32)
    value, _, _ = _random_case(1, 8, 2, 2, 4, 32, shapes)
    with expect_error(RuntimeError, 'pass value_layout="head_major"'):
        ttnn.experimental.fused_msda(_to_device(_head_major_value(value), device), loc_t, attn_t, shapes)


def test_fused_msda_head_major_layout_rejects_packed_value(device, expect_error):
    value, _, _ = _random_case(1, 8, 2, 2, 4, 32, [(6, 5), (3, 2)])
    _, loc_t, attn_t, shapes = _valid_inputs(device, H=2, D=32)
    with expect_error(RuntimeError, "is not head-major"):
        ttnn.experimental.fused_msda(
            _to_device(_pack_value(value), device), loc_t, attn_t, shapes, value_layout="head_major"
        )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("memory", ["dram", "l1"])
def test_fused_msda_head_major_value_repeated_calls_hit_program_cache(device, memory):
    """Producer and op each reuse one program across calls whose buffers sit at different addresses.

    The model calls both once per layer on fresh buffers, so a stale address
    binding on a cache hit would only show from the second layer on. Within a
    round both tensors of the a/b pair stay alive, so consecutive calls of each
    op alternate between two addresses (asserted) and every cache hit rebinds.
    """
    spatial_shapes = HEAD_MAJOR_SHAPES
    B, Q, H, D = 2, 65, 8, 32
    memory_config = ttnn.L1_MEMORY_CONFIG if memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
    value_a, ref_pts, offsets, attn = _v2_case(B, Q, H, 2, 4, D, 4, spatial_shapes, seed=1)
    value_b = _bf16(torch.randn_like(value_a))
    loc = _locations_from_offsets(ref_pts, offsets, spatial_shapes, "pillar")
    ref_t = _to_device(ref_pts, device)
    off_t = _to_device(_pack_locations(offsets), device)
    attn_t = _to_device(_pack_weights(attn), device)

    def tiled(value):
        return ttnn.from_torch(
            _pack_value(value).to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
        )

    entries_before = device.num_program_cache_entries()
    outputs = {}
    for i in range(8):
        tiled_a, tiled_b = tiled(value_a), tiled(value_b)
        hm_a = ttnn.experimental.fused_msda_head_major_value(tiled_a, H, memory_config=memory_config)
        hm_b = ttnn.experimental.fused_msda_head_major_value(tiled_b, H, memory_config=memory_config)
        assert tiled_a.buffer_address() != tiled_b.buffer_address()
        assert hm_a.buffer_address() != hm_b.buffer_address()
        for key, hm in (("a", hm_a), ("b", hm_b)):
            out = ttnn.to_torch(
                ttnn.experimental.fused_msda_from_offsets(
                    hm, ref_t, off_t, attn_t, spatial_shapes, reference_mode="pillar", value_layout="head_major"
                )
            ).to(torch.float32)
            if key not in outputs:
                outputs[key] = out
            else:
                torch.testing.assert_close(out, outputs[key], rtol=0, atol=0)
        for t in (hm_a, tiled_a, hm_b, tiled_b):
            ttnn.deallocate(t)
    _assert_close(_msda_reference(value_a, loc, attn, spatial_shapes), outputs["a"])
    _assert_close(_msda_reference(value_b, loc, attn, spatial_shapes), outputs["b"])
    # One program for the producer, one for the op.
    assert device.num_program_cache_entries() - entries_before == 2


def test_fused_msda_head_major_value_op_default_memory_config(device):
    """Without memory_config the output lands where the input lives."""
    value = ttnn.from_torch(
        torch.randn(1, 40, 64).to(torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = ttnn.experimental.fused_msda_head_major_value(value, 2)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1


def test_fused_msda_head_major_value_op_rejects_row_major(device, expect_error):
    value = _to_device(torch.randn(1, 64, 64), device)
    with expect_error(RuntimeError, "must be TILE"):
        ttnn.experimental.fused_msda_head_major_value(value, 2)


def test_fused_msda_head_major_value_op_rejects_float32(device, expect_error):
    value = ttnn.from_torch(torch.randn(1, 64, 64), dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "BFLOAT16"):
        ttnn.experimental.fused_msda_head_major_value(value, 2)


def test_fused_msda_head_major_value_op_rejects_non_32x32_tiles(device, expect_error):
    value = ttnn.from_torch(
        torch.randn(1, 64, 64).to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((16, 32))
    )
    with expect_error(RuntimeError, "32x32"):
        ttnn.experimental.fused_msda_head_major_value(value, 2)


def test_fused_msda_head_major_value_op_rejects_rank4_input(device, expect_error):
    value = ttnn.from_torch(torch.randn(1, 2, 64, 64).to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, r"must be \(B, S, H\*D\)"):
        ttnn.experimental.fused_msda_head_major_value(value, 2)


def test_fused_msda_head_major_value_op_rejects_zero_heads(device, expect_error):
    value = ttnn.from_torch(torch.randn(1, 64, 64).to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "num_heads must be > 0"):
        ttnn.experimental.fused_msda_head_major_value(value, 0)


@pytest.mark.parametrize("padded_shape", [[2, 96, 64], [2, 64, 96]], ids=["keys", "channels"])
def test_fused_msda_head_major_value_op_rejects_extra_tile_padding(device, expect_error, padded_shape):
    """Tile ids come from the logical shape, so padding past the last tile, in S or in H*D, would read other tiles."""
    row_major = ttnn.from_torch(torch.randn(2, 40, 64).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    padded = ttnn.tilize_with_val_padding(row_major, padded_shape, 0.0)
    assert list(padded.shape) == [2, 40, 64] and list(padded.padded_shape) == padded_shape
    with expect_error(RuntimeError, "only padding S up to a whole tile"):
        ttnn.experimental.fused_msda_head_major_value(padded, 2)


def test_fused_msda_head_major_value_op_rejects_heads_narrower_than_a_tile(device, expect_error):
    # H*D = 128 over 8 heads is D = 16: a head would share its tiles with the next.
    value = ttnn.from_torch(torch.randn(1, 64, 128).to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "whole number of tile widths"):
        ttnn.experimental.fused_msda_head_major_value(value, 8)
