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
