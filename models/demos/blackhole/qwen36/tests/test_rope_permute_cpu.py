# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exactness proof for the QWEN36_ROPE_PERMUTE transform (no device).

The claim behind the flag: rotating a head-dim-permuted Q/K with a full-dim
rotate-half and the expanded cos/sin tables equals permuting the partial-rope
result — exactly, not approximately — and the permutation cancels in QK^T and
in the per-head RMS norm. These tests pin that algebra in fp32 so a device
regression can be blamed on kernels/layout, never on the math.

Run: pytest models/demos/blackhole/qwen36/tests/test_rope_permute_cpu.py -v
"""
import pytest
import torch

from models.demos.blackhole.qwen36.tt.attention.rope_tp import (
    permute_rope_tables_full_dim,
    rope_head_permutation,
)

HEAD_DIM = 256
ROPE_DIM = 64  # partial_rotary_factor = 0.25


def _tables(positions, rope_dim, theta=10_000_000.0):
    """HF split-halves cos/sin [L, rope_dim]."""
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    emb = torch.cat([torch.outer(positions.float(), inv)] * 2, dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _partial_rope(x, cos, sin, rope_dim):
    """HF partial rope: rotate the first rope_dim dims, pass the rest. x [..., D], cos/sin [..., R]."""
    xr, xp = x[..., :rope_dim], x[..., rope_dim:]
    return torch.cat([xr * cos + _rotate_half(xr) * sin, xp], dim=-1)


@pytest.mark.parametrize("head_dim,rope_dim", [(256, 64), (128, 64), (64, 64), (256, 128)])
def test_permutation_is_bijective(head_dim, rope_dim):
    perm = rope_head_permutation(head_dim, rope_dim)
    assert perm.shape == (head_dim,)
    assert torch.equal(perm.sort().values, torch.arange(head_dim))
    if head_dim == rope_dim:
        assert torch.equal(perm, torch.arange(head_dim)), "R == D must degenerate to the identity"


def test_tables_pass_through_slots_are_exact_identity():
    cos, sin = _tables(torch.arange(7) * 13, ROPE_DIM)
    cos_f, sin_f = permute_rope_tables_full_dim(cos, sin, HEAD_DIM)
    assert cos_f.shape[-1] == HEAD_DIM and sin_f.shape[-1] == HEAD_DIM
    h, hd = ROPE_DIM // 2, HEAD_DIM // 2
    pass_slots = torch.cat([torch.arange(h, hd), torch.arange(hd + h, HEAD_DIM)])
    assert torch.equal(cos_f[..., pass_slots], torch.ones_like(cos_f[..., pass_slots]))
    assert torch.equal(sin_f[..., pass_slots], torch.zeros_like(sin_f[..., pass_slots]))
    # The rotary halves land at [0, h) and [hd, hd+h), pairing (i, i+D/2) like the device op.
    assert torch.equal(cos_f[..., :h], cos[..., :h])
    assert torch.equal(cos_f[..., hd : hd + h], cos[..., h:])
    assert torch.equal(sin_f[..., :h], sin[..., :h])
    assert torch.equal(sin_f[..., hd : hd + h], sin[..., h:])


@pytest.mark.parametrize("head_dim,rope_dim", [(256, 64), (128, 64), (64, 64)])
def test_full_dim_rope_on_permuted_equals_permuted_partial_rope(head_dim, rope_dim):
    """The exactness core: rope_full(Px) == P(rope_partial(x)), bitwise in fp32."""
    torch.manual_seed(0)
    L, H = 16, 3
    positions = torch.arange(L) * 7 + 3
    cos, sin = _tables(positions, rope_dim)  # [L, R]
    cos_f, sin_f = permute_rope_tables_full_dim(cos, sin, head_dim)  # [L, D]
    perm = rope_head_permutation(head_dim, rope_dim)

    x = torch.randn(L, H, head_dim)
    ref = _partial_rope(x, cos[:, None, :], sin[:, None, :], rope_dim)[..., perm]
    # Full-dim rotate-half on the permuted input — exactly what rotary_embedding_hf computes.
    got = x[..., perm] * cos_f[:, None, :] + _rotate_half(x[..., perm]) * sin_f[:, None, :]
    torch.testing.assert_close(got, ref, rtol=0.0, atol=0.0)


def test_qk_dot_product_invariant_under_permutation():
    torch.manual_seed(1)
    perm = rope_head_permutation(HEAD_DIM, ROPE_DIM)
    q = torch.randn(8, 3, HEAD_DIM)
    k = torch.randn(8, 3, HEAD_DIM)
    ref = (q * k).sum(-1)
    got = (q[..., perm] * k[..., perm]).sum(-1)
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-5)


def test_rms_norm_commutes_with_permutation():
    """RMSNorm with permuted weight on permuted input == permuted RMSNorm output."""
    torch.manual_seed(2)
    perm = rope_head_permutation(HEAD_DIM, ROPE_DIM)
    x = torch.randn(8, 3, HEAD_DIM)
    w = torch.randn(HEAD_DIM) + 1.0

    def rms(x, w):
        return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    ref = rms(x, w)[..., perm]
    got = rms(x[..., perm], w[perm])
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)
