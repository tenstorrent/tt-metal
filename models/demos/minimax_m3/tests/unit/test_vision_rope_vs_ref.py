# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks of the 3D vision RoPE in tt/vision/rope.py vs the Torch reference. No device.

rope.py computes the cos and sin on host; the device applies them as x * cos + (x @ R) * sin, with the
head dim padded from 80 to 96 (tt/vision/attention.py). The tests check, for square, non-square,
two-frame and two-image grids:
  - rope_cos_sin equals the reference's cos and sin bit for bit;
  - the padded form (rope_cos_sin_padded with rope_rotate_matrix) gives the reference rotation on the
    80 real dims and zeros on the 16 pad dims;
  - apply_rope_ref equals the reference's apply_rope.

Without a device these tests have no mesh id, so the 1-card leg's -k "1x1 ..." does not select them.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_rope_vs_ref.py
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.demos.minimax_m3.reference import vision as ref
from models.demos.minimax_m3.tt.vision.rope import apply_rope_ref, rope_cos_sin, rope_cos_sin_padded, rope_rotate_matrix

HEAD_DIM, PADDED_HEAD_DIM, NUM_HEADS, THETA, MERGE = 80, 96, 16, 10000.0, 2

# (t, h, w) patch grid of each image in the sequence.
ROPE_GRIDS = {
    "h16w16": [(1, 16, 16)],
    "h32w32": [(1, 32, 32)],
    "h48w48": [(1, 48, 48)],
    "h34w46": [(1, 34, 46)],
    "h46w34": [(1, 46, 34)],
    "t2h16w16": [(2, 16, 16)],
    "two_images": [(1, 16, 16), (1, 34, 46)],
}
parametrize_rope_grids = pytest.mark.parametrize("grid_thw", list(ROPE_GRIDS.values()), ids=list(ROPE_GRIDS))


@torch.no_grad()
@parametrize_rope_grids
def test_vision_rope_cos_sin_vs_ref(grid_thw):
    """rope_cos_sin equals the reference's cos and sin, bit for bit."""
    grid = torch.tensor(grid_thw)
    cos, sin = rope_cos_sin(grid, head_dim=HEAD_DIM, theta=THETA, spatial_merge_size=MERGE)
    ref_cos, ref_sin = ref.rope_cos_sin(grid, HEAD_DIM, THETA, MERGE)
    assert cos.shape == (int(grid.prod(dim=-1).sum()), 3 * ref.rope_axis_dim(HEAD_DIM))
    assert torch.equal(cos, ref_cos), (cos - ref_cos).abs().max()
    assert torch.equal(sin, ref_sin), (sin - ref_sin).abs().max()


@torch.no_grad()
@parametrize_rope_grids
def test_vision_rope_padded_vs_ref(grid_thw):
    """x * cos + (x @ R) * sin over the padded head dim equals the reference rotation, with zero pad dims."""
    grid = torch.tensor(grid_thw)
    num_patches = int(grid.prod(dim=-1).sum())
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, generator=generator)
    k = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, generator=generator)
    ref_q, ref_k = ref.apply_rope(q, k, *ref.rope_cos_sin(grid, HEAD_DIM, THETA, MERGE))

    cos, sin = rope_cos_sin_padded(
        grid, head_dim=HEAD_DIM, padded_head_dim=PADDED_HEAD_DIM, theta=THETA, spatial_merge_size=MERGE
    )
    rotate = rope_rotate_matrix(HEAD_DIM, PADDED_HEAD_DIM)
    for name, x, x_ref in (("q", q, ref_q), ("k", k, ref_k)):
        heads = F.pad(x, (0, PADDED_HEAD_DIM - HEAD_DIM)).transpose(1, 2)  # (1, heads, L, 96), as on device
        out = (heads * cos + (heads @ rotate) * sin).transpose(1, 2)
        torch.testing.assert_close(out[..., :HEAD_DIM], x_ref, rtol=1e-6, atol=1e-6, msg=lambda m: f"{name}: {m}")
        assert torch.all(out[..., HEAD_DIM:] == 0), f"{name}: nonzero pad dims"


@torch.no_grad()
def test_vision_apply_rope_ref_vs_ref():
    """apply_rope_ref equals the reference's apply_rope."""
    grid = torch.tensor(ROPE_GRIDS["h34w46"])
    num_patches = int(grid.prod(dim=-1).sum())
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, generator=generator)
    k = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, generator=generator)
    cos, sin = ref.rope_cos_sin(grid, HEAD_DIM, THETA, MERGE)
    for ours, theirs in zip(apply_rope_ref(q, k, cos, sin), ref.apply_rope(q, k, cos, sin)):
        assert torch.equal(ours, theirs), (ours - theirs).abs().max()
