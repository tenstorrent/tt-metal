# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Step 1 of the 3D-neighborhood windowed-SDPA generalization: validate the algorithm
on the host, in the *kernel's* coordinate conventions, before any C++/rebuild.

Two things the eventual kernel must get right, both error-prone and both cheap to pin here:

1. The 3D mask predicate: decode a flat T-outer index to ``(t,h,w)`` the way the kernel will
   (``t = idx // (H*W)``, ``h = (idx % (H*W)) // W``, ``w = idx % W``), and test membership in the
   NATTEN inward-shifted ``(kt,kh,kw)`` box. Verified by running neighborhood attention two ways
   -- dense-with-my-mask vs the validated :func:`na3d_torch` -- and requiring bit-agreement.

2. The T-only k-range: with the T-outer layout, whole frames are contiguous, so a Q chunk needs
   exactly the frames its queries' T-windows span. Verified for completeness against the full mask.

When this is green, Step 2 transcribes ``in_window_3d`` into ``windowed_mask_gen.hpp`` and Step 3
transcribes ``frame_k_range`` into ``windowed_loop_geometry.hpp``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.tt_dit.layers.na3d import na3d_torch, window_bounds

# (dims, kernel): interior + both border regimes, kernel==axis, and kernel>axis (whole-axis clamp).
CASES = [
    ((5, 4, 4), (3, 3, 3)),
    ((4, 3, 5), (3, 3, 3)),
    ((6, 4, 4), (3, 3, 3)),
    ((7, 4, 4), (5, 3, 3)),
    ((5, 5, 5), (5, 5, 5)),
    ((3, 4, 4), (5, 3, 3)),  # kt=5 > T=3 -> clamps to whole T axis
]


def decode(idx: int, dims: tuple[int, int, int]) -> tuple[int, int, int]:
    """Flat T-outer index -> (t,h,w), exactly as the kernel will decode it."""
    _, h_, w_ = dims
    t = idx // (h_ * w_)
    rem = idx % (h_ * w_)
    return t, rem // w_, rem % w_


def in_axis(q: int, k: int, length: int, ker: int, stride: int = 1) -> bool:
    """NATTEN inward-shifted 1D window membership (mirrors na3d.window_bounds).

    ``stride`` snaps the query to its GNA group leader first. Written out longhand rather than calling
    ``window_bounds`` so this stays an INDEPENDENT ground truth for it.
    """
    ker = min(ker, length)
    leader = min((q // stride) * stride + stride // 2, length - 1)
    start = min(max(leader - ker // 2, 0), length - ker)
    return start <= k < start + ker


def in_window_3d(
    q: int,
    k: int,
    dims: tuple[int, int, int],
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
) -> bool:
    qt, qh, qw = decode(q, dims)
    kt_, kh_, kw_ = decode(k, dims)
    (t_, h_, w_), (kt, kh, kw) = dims, kernel
    st, sh, sw = stride
    return in_axis(qt, kt_, t_, kt, st) and in_axis(qh, kh_, h_, kh, sh) and in_axis(qw, kw_, w_, kw, sw)


def neighborhood_3d_mask(
    dims: tuple[int, int, int],
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    s = dims[0] * dims[1] * dims[2]
    m = torch.zeros(s, s, dtype=torch.bool)
    for q in range(s):
        for k in range(s):
            if in_window_3d(q, k, dims, kernel, stride):
                m[q, k] = True
    return m


def frame_k_range(q_frames: range, t_len: int, kt: int, st: int = 1) -> tuple[int, int]:
    """Frames a Q chunk (spanning ``q_frames``) needs: union of its queries' T-windows.

    Contiguous in the T-outer layout, so this is exactly [min start, max end)."""
    starts, ends = window_bounds(t_len, kt, st)
    return min(starts[q] for q in q_frames), max(ends[q] for q in q_frames)


@pytest.mark.parametrize("dims,kernel", CASES)
def test_mask_predicate_matches_na3d_torch(dims, kernel):
    """Dense attention with my 3D mask == the validated na3d_torch reference."""
    torch.manual_seed(0)
    t_, h_, w_ = dims
    s, head_dim = t_ * h_ * w_, 8
    q = torch.randn(1, t_, h_, w_, 1, head_dim)
    k = torch.randn(1, t_, h_, w_, 1, head_dim)
    v = torch.randn(1, t_, h_, w_, 1, head_dim)

    ref = na3d_torch(q, k, v, kernel_size=kernel).reshape(s, head_dim)  # validated NA3D

    mask = neighborhood_3d_mask(dims, kernel)
    attn_mask = torch.where(mask, 0.0, float("-inf"))
    mine = F.scaled_dot_product_attention(
        q.reshape(s, head_dim), k.reshape(s, head_dim), v.reshape(s, head_dim), attn_mask=attn_mask
    )
    torch.testing.assert_close(mine, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dims,kernel", CASES)
@pytest.mark.parametrize("chunk_frames", [1, 2, 3])
def test_frame_k_range_covers_every_in_window_key(dims, kernel, chunk_frames):
    """Every in-window key of a Q chunk lands in the chunk's computed frame range."""
    t_, h_, w_ = dims
    hw = h_ * w_
    mask = neighborhood_3d_mask(dims, kernel)
    for lo in range(0, t_, chunk_frames):
        q_frames = range(lo, min(lo + chunk_frames, t_))
        f_lo, f_hi = frame_k_range(q_frames, t_, kernel[0])
        for qf in q_frames:
            for local in range(hw):  # every query in these frames
                q = qf * hw + local
                for k in torch.nonzero(mask[q], as_tuple=False).flatten().tolist():
                    kf = k // hw
                    assert f_lo <= kf < f_hi, f"key frame {kf} for q={q} outside [{f_lo},{f_hi})"


# --- Generalized Neighborhood Attention (stride > 1) -----------------------------------------------
# (dims, kernel, stride): each stride divides its axis and is <= its kernel, matching what the op
# validates. Includes stride == kernel (maximally coarse, the block-sparse setting) and mixed strides
# where only some axes are grouped.
STRIDE_CASES = [
    ((4, 4, 4), (3, 3, 3), (2, 2, 2)),
    ((6, 4, 4), (3, 3, 3), (3, 1, 1)),
    ((4, 4, 4), (3, 3, 3), (1, 2, 1)),
    ((8, 4, 6), (5, 3, 3), (4, 2, 3)),
    ((5, 5, 5), (5, 5, 5), (5, 5, 5)),
    ((6, 4, 4), (3, 3, 3), (3, 2, 2)),
]


@pytest.mark.parametrize("dims,kernel,stride", STRIDE_CASES)
def test_window_bounds_matches_independent_leader_snap(dims, kernel, stride):
    """``window_bounds`` (the shared host/device primitive) == the longhand leader-snap above.

    ``in_axis`` recomputes the leader from scratch, so this pins the primitive itself rather than
    comparing it to a copy of itself.
    """
    for length, ker, s in zip(dims, kernel, stride):
        starts, ends = window_bounds(length, ker, s)
        for q in range(length):
            for k in range(length):
                assert (starts[q] <= k < ends[q]) == in_axis(
                    q, k, length, ker, s
                ), f"length={length} ker={ker} stride={s} q={q} k={k}"


@pytest.mark.parametrize("dims,kernel", CASES)
def test_stride_one_is_exactly_standard_neighborhood_attention(dims, kernel):
    """Stride 1 must be the IDENTITY, not merely close: gna_leader(q, 1) == q by construction.

    This is the regression guard for the whole feature -- every existing NA caller runs this path.
    """
    for length, ker in zip(dims, kernel):
        assert window_bounds(length, ker, 1) == window_bounds(
            length, ker
        ), f"stride 1 diverged from no-stride at length={length} ker={ker}"
    assert torch.equal(neighborhood_3d_mask(dims, kernel, (1, 1, 1)), neighborhood_3d_mask(dims, kernel))


@pytest.mark.parametrize("dims,kernel,stride", STRIDE_CASES)
def test_gna_mask_predicate_matches_na3d_torch(dims, kernel, stride):
    """Dense attention with the strided 3D mask == na3d_torch planned at the same stride."""
    torch.manual_seed(0)
    t_, h_, w_ = dims
    s, head_dim = t_ * h_ * w_, 8
    q = torch.randn(1, t_, h_, w_, 1, head_dim)
    k = torch.randn(1, t_, h_, w_, 1, head_dim)
    v = torch.randn(1, t_, h_, w_, 1, head_dim)

    ref = na3d_torch(q, k, v, kernel_size=kernel, stride=stride).reshape(s, head_dim)

    attn_mask = torch.where(neighborhood_3d_mask(dims, kernel, stride), 0.0, float("-inf"))
    mine = F.scaled_dot_product_attention(
        q.reshape(s, head_dim), k.reshape(s, head_dim), v.reshape(s, head_dim), attn_mask=attn_mask
    )
    torch.testing.assert_close(mine, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dims,kernel,stride", STRIDE_CASES)
def test_group_members_share_one_window(dims, kernel, stride):
    """The property the block-sparse fast path rests on: within a group every query has the SAME
    window, so the union of a group's windows (the kernel's 'box') is that one window and there is
    nothing left for a fine-grained mask to remove."""
    for length, ker, s in zip(dims, kernel, stride):
        starts, ends = window_bounds(length, ker, s)
        for lo in range(0, length, s):
            group = range(lo, min(lo + s, length))
            windows = {(starts[q], ends[q]) for q in group}
            assert len(windows) == 1, f"group {list(group)} of length={length} ker={ker} s={s} -> {windows}"
            (w_lo, w_hi) = windows.pop()
            assert w_hi - w_lo == min(ker, length)


@pytest.mark.parametrize("dims,kernel,stride", STRIDE_CASES)
def test_every_query_attends_itself_under_stride(dims, kernel, stride):
    """stride <= kernel (what the op validates) must keep every query inside the window it shares.

    A group wider than its window would push edge members out of their own receptive field.
    """
    for length, ker, s in zip(dims, kernel, stride):
        starts, ends = window_bounds(length, ker, s)
        for q in range(length):
            assert starts[q] <= q < ends[q], f"q={q} outside its own window at length={length} ker={ker} s={s}"
