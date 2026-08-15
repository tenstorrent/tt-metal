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
