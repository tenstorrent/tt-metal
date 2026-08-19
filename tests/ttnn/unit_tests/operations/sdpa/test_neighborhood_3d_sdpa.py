# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Step 2 of the 3D-neighborhood generalization: the windowed-SDPA op's 3D-neighborhood mode
(``neighborhood_3d=(T,H,W,kt,kh,kw)``) vs the validated host reference ``na3d_torch``.

Step 2 builds the per-element 3D mask on-device with the K-range left FULL (no narrowing yet), so
this isolates mask correctness. Small grids keep it cheap; larger grids / the k-range narrowing come
in step 3, and wiring it into DiffVAE NA3D in step 4.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.layers.na3d import na3d_torch
from models.tt_dit.utils.check import assert_quality

# interior + both border regimes + a larger temporal kernel; S = T*H*W kept to a few tiles.
CASES = [
    ((4, 4, 4), (3, 3, 3)),
    ((5, 4, 3), (3, 3, 3)),
    ((3, 4, 4), (3, 3, 3)),
    ((4, 4, 4), (5, 3, 3)),
    ((6, 3, 3), (3, 3, 3)),
]


@pytest.mark.parametrize("dims,kernel", CASES)
def test_neighborhood_3d_matches_na3d_torch(device, dims, kernel):
    torch.manual_seed(0)
    t_, h_, w_ = dims
    s, head_dim = t_ * h_ * w_, 64

    q = torch.randn(1, t_, h_, w_, 1, head_dim)
    k = torch.randn(1, t_, h_, w_, 1, head_dim)
    v = torch.randn(1, t_, h_, w_, 1, head_dim)

    # Reference: validated host NA3D (both default to scale = 1/sqrt(head_dim)).
    ref = na3d_torch(q, k, v, kernel_size=kernel).reshape(s, head_dim)

    def to_tt(x):
        return ttnn.from_torch(
            x.reshape(1, 1, s, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    out = ttnn.transformer.scaled_dot_product_attention(
        to_tt(q), to_tt(k), to_tt(v), is_causal=False, neighborhood_3d=(t_, h_, w_, *kernel)
    )
    got = ttnn.to_torch(out).reshape(s, head_dim)

    assert_quality(ref, got, pcc=0.99)


# T-shards (frame ranges) whose token offset t_lo*H*W lands on a TILE_HEIGHT boundary, so the shard
# is expressible as a windowed_q_token_offset. HW = 64 here (a multiple of 32).
@pytest.mark.parametrize("t_lo, t_hi", [(0, 4), (4, 8), (2, 6)])
@pytest.mark.parametrize("offset_as_tensor", [False, True], ids=["scalar", "tensor"])
def test_neighborhood_3d_q_offset_shard(device, t_lo, t_hi, offset_as_tensor):
    """SP-over-T capability: a globally-offset Q shard against full K/V reproduces the same slice of
    the full result. This is what lets the host split Q over T across a mesh (K/V replicated). The
    offset is supplied as a baked scalar or, for a real mesh where one program serves differently-
    offset chips, as a per-device tensor read on device."""
    torch.manual_seed(0)
    dims, kernel, head_dim = (8, 8, 8), (3, 3, 3), 64
    t_, h_, w_ = dims
    s, hw = t_ * h_ * w_, h_ * w_

    q = torch.randn(1, t_, h_, w_, 1, head_dim)
    k = torch.randn(1, t_, h_, w_, 1, head_dim)
    v = torch.randn(1, t_, h_, w_, 1, head_dim)

    tk = ttnn.from_torch(k.reshape(1, 1, s, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tv = ttnn.from_torch(v.reshape(1, 1, s, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    full = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            ttnn.from_torch(q.reshape(1, 1, s, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            tk,
            tv,
            is_causal=False,
            neighborhood_3d=(t_, h_, w_, *kernel),
        )
    ).reshape(s, head_dim)

    lo, hi = t_lo * hw, t_hi * hw
    q_shard = q[:, t_lo:t_hi].reshape(1, 1, hi - lo, head_dim)
    off_tensor = (
        ttnn.from_torch(
            torch.tensor([lo], dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        if offset_as_tensor
        else None
    )
    got = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            ttnn.from_torch(q_shard, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            tk,
            tv,
            is_causal=False,
            neighborhood_3d=(t_, h_, w_, *kernel),
            windowed_q_token_offset=0 if offset_as_tensor else lo,
            windowed_q_token_offset_tensor=off_tensor,
        )
    ).reshape(hi - lo, head_dim)

    assert_quality(full[lo:hi], got, pcc=0.99)


# W-bands whose padded shard (interior + halo) reproduces the full-result W-columns. Covers an
# interior band and both true edges, where the inward-shift must win over the fake replicate halo.
@pytest.mark.parametrize("w_lo, w_hi", [(2, 6), (0, 4), (4, 8)], ids=["interior", "left_edge", "right_edge"])
def test_neighborhood_3d_w_shard(device, w_lo, w_hi):
    """Spatial-SP over W: neighborhood_w_shard=(W_full, w_origin) makes a padded W-band shard attend
    at global W. This is what lets the host split the volume over W with a neighbor-pad halo."""
    torch.manual_seed(0)
    t_, h_, w_full, kt, kh, kw, head_dim = 4, 4, 8, 3, 3, 3, 64
    halo = kw // 2
    qv, kv, vv = (torch.randn(1, t_, h_, w_full, 1, head_dim) for _ in range(3))

    def to_tt(x):
        s = x.shape[1] * x.shape[2] * x.shape[3]
        return ttnn.from_torch(
            x.reshape(1, 1, s, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    full = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            to_tt(qv), to_tt(kv), to_tt(vv), is_causal=False, neighborhood_3d=(t_, h_, w_full, kt, kh, kw)
        )
    ).reshape(t_, h_, w_full, head_dim)

    pad_lo, pad_hi = w_lo - halo, w_hi + halo
    cols = [min(max(w, 0), w_full - 1) for w in range(pad_lo, pad_hi)]  # replicate fake halo at true edges
    w_pad = len(cols)
    qs, ks, vs = (x[:, :, :, cols, :, :] for x in (qv, kv, vv))
    out = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            to_tt(qs),
            to_tt(ks),
            to_tt(vs),
            is_causal=False,
            neighborhood_3d=(t_, h_, w_pad, kt, kh, kw),
            neighborhood_w_shard=(w_full, pad_lo & 0xFFFFFFFF),
        )
    ).reshape(t_, h_, w_pad, head_dim)
    interior = out[:, :, halo : halo + (w_hi - w_lo), :]
    assert_quality(full[:, :, w_lo:w_hi, :], interior, pcc=0.99)


# --- Generalized Neighborhood Attention (neighborhood_stride) ---------------------------------------
# (dims, kernel, stride): each stride divides its axis and is <= its kernel. Mixed strides check that
# the axes are independent; stride == kernel is the coarsest (block-sparse) setting.
STRIDE_CASES = [
    ((4, 4, 4), (3, 3, 3), (2, 2, 2)),
    ((4, 4, 4), (3, 3, 3), (1, 2, 1)),
    ((6, 4, 4), (3, 3, 3), (3, 1, 2)),
    ((4, 4, 4), (3, 3, 3), (2, 1, 1)),
    ((6, 3, 3), (3, 3, 3), (3, 3, 3)),
]


def _nbr_flat(device, q, k, v, dims, kernel, stride=None):
    t_, h_, w_ = dims
    s, head_dim = t_ * h_ * w_, q.shape[-1]

    def to_tt(x):
        return ttnn.from_torch(
            x.reshape(1, 1, s, head_dim), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    out = ttnn.transformer.scaled_dot_product_attention(
        to_tt(q),
        to_tt(k),
        to_tt(v),
        is_causal=False,
        neighborhood_3d=(t_, h_, w_, *kernel),
        neighborhood_stride=stride,
    )
    return ttnn.to_torch(out).reshape(s, head_dim)


@pytest.mark.parametrize("dims,kernel,stride", STRIDE_CASES)
def test_gna_stride_matches_na3d_torch(device, dims, kernel, stride):
    """The op's GNA mode vs the host reference planned at the same stride."""
    torch.manual_seed(0)
    t_, h_, w_ = dims
    s, head_dim = t_ * h_ * w_, 64
    q, k, v = (torch.randn(1, t_, h_, w_, 1, head_dim) for _ in range(3))

    ref = na3d_torch(q, k, v, kernel_size=kernel, stride=stride).reshape(s, head_dim)
    got = _nbr_flat(device, q, k, v, dims, kernel, stride)
    assert_quality(ref, got, pcc=0.99)


@pytest.mark.parametrize("dims,kernel", CASES)
def test_gna_stride_one_is_bit_identical_to_no_stride(device, dims, kernel):
    """Stride (1,1,1) must be BIT-identical to omitting the argument, not merely within PCC.

    gna_leader(q, 1) == q makes stride 1 the identity by construction, so any difference at all means
    the stride plumbing perturbed the standard NA path -- the one regression that would affect every
    existing caller. Exact equality is the only assertion that can catch a subtle drift here.
    """
    torch.manual_seed(0)
    t_, h_, w_ = dims
    q, k, v = (torch.randn(1, t_, h_, w_, 1, 64) for _ in range(3))

    base = _nbr_flat(device, q, k, v, dims, kernel, None)
    strided = _nbr_flat(device, q, k, v, dims, kernel, (1, 1, 1))
    assert torch.equal(
        base, strided
    ), f"stride (1,1,1) diverged from the no-stride path: max|delta| = {(base - strided).abs().max()}"


@pytest.mark.parametrize(
    "stride,message",
    # Matching the TT_FATAL text pins WHICH rule fired: a stride of 0 must not be caught by the
    # divisibility check, or a future reordering of the validation could hide one rule behind another.
    [
        ((0, 1, 1), "must be >= 1"),
        ((5, 1, 1), "must not exceed the effective kernel"),
        ((3, 1, 1), "must divide the t axis length"),
    ],
)
def test_gna_stride_invalid_is_rejected(device, expect_error, stride, message):
    """Host validation must reject these rather than producing a silently wrong window."""
    torch.manual_seed(0)
    dims, kernel = (4, 4, 4), (3, 3, 3)
    q, k, v = (torch.randn(1, 4, 4, 4, 1, 64) for _ in range(3))
    with expect_error(RuntimeError, message):
        _nbr_flat(device, q, k, v, dims, kernel, stride)
