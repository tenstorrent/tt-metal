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


def _locations_from_offsets(reference_points, sampling_offsets, spatial_shapes, reference_mode):
    """The V2 frontend, in PyTorch. (B, Q, R, 2) + (B, Q, H, L, P, 2) -> (B, Q, H, L, P, 2)."""
    B, Q, H, L, P, _ = sampling_offsets.shape
    R = reference_points.shape[2]
    loc = torch.empty_like(sampling_offsets)
    for lvl, (h_, w_) in enumerate(spatial_shapes):
        norm = torch.tensor([1.0 / w_, 1.0 / h_], dtype=sampling_offsets.dtype)
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


def _per_head_stride_supported(num_heads: int, head_dim: int) -> bool:
    """Can the writer place this many heads side by side in one output page?

    The output is (B, Q, H*D) and the writer emits head h as a D*2-byte NoC
    write at byte offset h*D*2 inside query (b, q)'s page, so that stride must
    satisfy the device's DRAM alignment as soon as there is more than one head.
    D=16 gives a 32-byte stride: legal on a 32-byte-aligned target, rejected on
    a 64-byte one (Blackhole). D=32 and up are unconstrained everywhere.

    See test_fused_msda_rejects_unaligned_per_head_stride, which pins the
    rejection itself; this helper only tells the sweep what to expect.
    """
    if num_heads == 1:
        return True
    return (head_dim * 2) % ttnn.device.get_dram_alignment() == 0


def _to_device(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t.to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)


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
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("Q", [1, 31, 32, 33, 256])
@pytest.mark.parametrize("H", [1, 8])
@pytest.mark.parametrize("L,spatial_shapes", [(1, [(10, 10)]), (4, [(16, 20), (8, 10), (4, 5), (2, 3)])])
@pytest.mark.parametrize("P", [1, 4, 8])
@pytest.mark.parametrize("D", [16, 32, 64])
@pytest.mark.parametrize("align_corners", [False, True])
def test_fused_msda_v1(device, B, Q, H, L, spatial_shapes, P, D, align_corners):
    if not _per_head_stride_supported(H, D):
        pytest.skip(
            f"head_dim {D} gives a {D * 2}-byte per-head output stride, which this device's "
            f"{ttnn.device.get_dram_alignment()}-byte DRAM alignment does not allow with H={H}"
        )

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


@pytest.mark.parametrize("B,Q,H,L,P,D", [(1, 64, 2, 2, 4, 32)])
def test_fused_msda_v1_packed_value_equivalence(device, B, Q, H, L, P, D):
    """Packed rank-3 value (B, S, H*D) must match canonical (B, S, H, D) exactly.

    The two layouts are the same bytes; only the DRAM page size and the reader's
    (page, offset) addressing differ. A mismatch is an addressing bug, not bf16
    rounding.
    """
    spatial_shapes = [(12, 9), (6, 5)]
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes)

    loc_t = _to_device(_pack_locations(loc), device)
    attn_t = _to_device(_pack_weights(attn), device)
    canonical = ttnn.to_torch(
        ttnn.experimental.fused_msda(_to_device(value, device), loc_t, attn_t, spatial_shapes)
    ).to(torch.float32)
    packed = ttnn.to_torch(
        ttnn.experimental.fused_msda(_to_device(_pack_value(value), device), loc_t, attn_t, spatial_shapes)
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
        _msda_reference(
            value, grid, attn, spatial_shapes, align_corners=align_corners, locations_in_grid_space=True
        ),
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


def test_fused_msda_v1_masks_out_of_bounds_corners(device):
    """An out-of-bounds corner must contribute nothing, element by element.

    Regression gate for the input-tile mask. The compute kernel builds the
    reduction's scalar from the fractions the SFPU produced and has no bounds
    information, so the zeroing of un-gathered input rows in
    `fused_msda_reader_common.hpp` is the only thing standing between a stale
    circular-buffer row and a live weight.

    It asserts a per-element error ratio on top of `_assert_close` because a
    stale row is itself a plausible sampled value: correlation degrades more
    slowly than the individual elements do, so the ratio is what stays
    meaningful as shapes shrink and the output magnitude falls.

    The case is built so that both halves matter: enough corners are outside
    the map to exercise the mask, and enough are inside that a slot the reader
    skips has just been written with a real value stick by an earlier point.
    `Q = 129` also gives every (batch, head) a partial trailing block, so the
    `r >= v_rows` half of the zeroing predicate is taken as well.
    """
    spatial_shapes = [(16, 20), (8, 10), (4, 5), (2, 3)]
    B, Q, H, L, P, D = 1, 129, 4, 4, 4, 32
    value, loc, attn = _random_case(B, Q, H, L, P, D, spatial_shapes, loc_range=(-0.25, 1.25))

    oob = _oob_corner_fraction(loc, spatial_shapes)
    assert 0.25 < oob < 0.75, f"test no longer exercises the mask: {oob:.2%} of corners are out of bounds"

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

    abs_tol = max(2e-2, 6e-2 * ref.abs().max().item())
    high_error_ratio = ((ref - out).abs() > abs_tol).float().mean().item()
    assert high_error_ratio <= 0.01, (
        f"{high_error_ratio:.2%} of output elements are off by more than {abs_tol:.4f}. "
        "An unmasked out-of-bounds corner reads a stale input row and multiplies it by a live weight"
    )


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

    loc = _bf16(_locations_from_offsets(ref_pts, offsets, spatial_shapes, reference_mode))

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
    _assert_close(
        _msda_reference(value, loc, attn, spatial_shapes, align_corners=align_corners), v2, pcc=0.99
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
    # NOT _bf16(): the V2 reader reads bf16 reference points and offsets but forms
    # the location in fp32 and samples at that unrounded position, so rounding it
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
def test_fused_msda_rejects_spatial_shape_beyond_bf16_exact_integers(device, expect_error, shapes):
    """A feature map larger than bf16 counts exactly must be refused, not mis-sampled.

    The SFPU geometry hands the reader floor(px) as bf16, which represents every
    integer up to 256 and only some beyond it. At 257 the decoded corner would
    round to a *different, still in-bounds* pixel -- a silently wrong sample
    rather than a failure -- so the op rejects the shape instead.

    Both axes are parametrized because one `TT_FATAL` tests both, so a refactor
    that drops the `w` half would otherwise go unnoticed.
    """
    value_t, loc_t, attn_t, _ = _valid_inputs(device, L=1, spatial_shapes=shapes)
    with expect_error(RuntimeError, "per-axis limit"):
        ttnn.experimental.fused_msda(value_t, loc_t, attn_t, shapes)


def test_fused_msda_accepts_spatial_shape_at_the_bf16_exact_limit(device):
    """256 is the largest exactly-representable corner index, so it must be accepted.

    The companion to the rejection test above: it pins the boundary itself, so a
    `<` where the code means `<=` fails here rather than silently narrowing the
    op. It is also the only case that exercises the bf16 corner decode at its
    stated limit.
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


def test_fused_msda_rejects_unaligned_per_head_stride(device, expect_error):
    """A per-head output stride the device cannot address must be refused, not silently
    mis-written into the neighbouring head's bytes.

    On a 32-byte-aligned target every supported head_dim is addressable and there is
    nothing to reject, so the assertion only applies where the alignment is coarser.
    """
    head_dim = 16
    if _per_head_stride_supported(2, head_dim):
        pytest.skip(f"this device's {ttnn.device.get_dram_alignment()}-byte alignment permits head_dim {head_dim}")

    value_t, loc_t, attn_t, shapes = _valid_inputs(device, H=8, D=head_dim)
    with expect_error(RuntimeError, "per-head output stride"):
        ttnn.experimental.fused_msda(value_t, loc_t, attn_t, shapes)


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


def test_fused_msda_from_offsets_rejects_unknown_reference_mode(device, expect_error):
    value_t, off_t, attn_t, shapes = _valid_inputs(device)
    ref_t = _to_device(torch.rand(1, 8, 2, 2), device)
    with expect_error(RuntimeError, "reference_mode"):
        ttnn.experimental.fused_msda_from_offsets(value_t, ref_t, off_t, attn_t, shapes, reference_mode="boxes")
