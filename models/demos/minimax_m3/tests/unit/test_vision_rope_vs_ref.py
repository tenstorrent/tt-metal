# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks of the 3D vision RoPE's layout on device (tt/vision/rope.py and the q and k packing in
tt/vision/attention.py) vs the Torch reference, and of the model's RoPE tables per grid. No device.

The device rotates q and k with ttnn.experimental.rotary_embedding_llama, which pairs lanes (2i, 2i+1) of a head, so
the q and k rows of wqkv are packed in the head order of attention._rope_head_order, padded from 80 to 96, and
rope.py lays out the reference's cos and sin in that order. The tests check, for square, non-square, two-frame and
two-image grids:
  - the op's rotation, run on host, of q and k in that order with the cos and sin of rope_cos_sin_padded equals
    the reference rotation bit for bit, keeps the 16 pad dims at zero and leaves q k^T as the reference's;
  - the attention computed on host from M3VLAttention's packed weights (q and k rows reordered) equals the
    reference attention;
and that M3VLVisionModel keeps the cos and sin of a grid on device for the next images of that grid.

These tests have no mesh id; the 1-card CI leg selects them by name (its -k includes "vision_rope").

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_rope_vs_ref.py
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.tensor_utils import get_rot_transformation_mat
from models.demos.minimax_m3.reference import vision as ref
from models.demos.minimax_m3.tt.vision.attention import (
    _pack_wqkv_bias,
    _pack_wqkv_weight,
    _pad_wo_weight,
    _rope_head_order,
)
from models.demos.minimax_m3.tt.vision.model import ROPE_CACHE_GRIDS, M3VLVisionModel
from models.demos.minimax_m3.tt.vision.model_config import MiniMaxM3VLModelArgs
from models.demos.minimax_m3.tt.vision.rope import rope_cos_sin_padded

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


def rotary_embedding_llama(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ttnn.experimental.rotary_embedding_llama in prefill mode, on host: x * cos + (x @ trans_mat) * sin, with the
    op's one-tile transformation matrix applied to each 32-lane tile of the head dim."""
    trans_mat = get_rot_transformation_mat()[0, 0].to(x.dtype)  # (32, 32)
    rotated = (x.unflatten(-1, (-1, trans_mat.shape[0])) @ trans_mat).flatten(-2)
    return x * cos + rotated * sin


@torch.no_grad()
@parametrize_rope_grids
def test_vision_rope_padded_vs_ref(grid_thw):
    """The RoPE op's rotation of q and k in the device head order equals the reference rotation, bit for bit, with
    zero pad dims and the reference's q k^T."""
    grid = torch.tensor(grid_thw)
    num_patches = int(grid.prod(dim=-1).sum())
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, generator=generator)
    k = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, generator=generator)
    ref_q, ref_k = ref.apply_rope(q, k, *ref.rope_cos_sin(grid, HEAD_DIM, THETA, MERGE))

    cos, sin = rope_cos_sin_padded(
        grid, head_dim=HEAD_DIM, padded_head_dim=PADDED_HEAD_DIM, theta=THETA, spatial_merge_size=MERGE
    )
    perm = _rope_head_order(torch.arange(HEAD_DIM), 1, HEAD_DIM)  # lane j of a q or k head holds dim perm[j]
    rotated = {}
    for name, x, x_ref in (("q", q, ref_q), ("k", k, ref_k)):
        # (1, heads, L, 96), as the QKV linear with the reordered weights and the head split give them
        heads = F.pad(x[..., perm], (0, PADDED_HEAD_DIM - HEAD_DIM)).transpose(1, 2)
        rotated[name] = rotary_embedding_llama(heads, cos, sin)
        out = rotated[name].transpose(1, 2)
        torch.testing.assert_close(out[..., :HEAD_DIM], x_ref[..., perm], rtol=0, atol=0, msg=lambda m: f"{name}: {m}")
        assert torch.all(out[..., HEAD_DIM:] == 0), f"{name}: nonzero pad dims"

    # q k^T over the 96 lanes against the reference's over its 80 dims, for the first 128 queries, in fp64: the
    # products are exact there, so only the summation order differs.
    scores = rotated["q"][:, :, :128].double() @ rotated["k"].double().transpose(-2, -1)
    ref_scores = ref_q.transpose(1, 2)[:, :, :128].double() @ ref_k.transpose(1, 2).double().transpose(-2, -1)
    torch.testing.assert_close(scores, ref_scores, rtol=1e-12, atol=1e-12)


@torch.no_grad()
@pytest.mark.parametrize("grid_thw", [ROPE_GRIDS["h16w16"], ROPE_GRIDS["two_images"]], ids=["h16w16", "two_images"])
def test_vision_attention_layout_vs_ref(grid_thw):
    """The attention computed from M3VLAttention's packed weights equals the reference attention.

    Runs what the device runs, on host in fp64: the QKV linear with the packed weight and bias (q and k rows in the
    RoPE op's head order, heads padded to 96), the head split, the RoPE op on q and k, SDPA over the 96 lanes with
    scale 1/sqrt(80), the head concat and the output projection with its padded weight. The reference layer has
    bf16-valued weights, which packing keeps exactly; its softmax runs in fp32, hence the tolerance.
    """
    num_heads = 2
    config = ref.MiniMaxM3VisionConfig(
        hidden_size=num_heads * HEAD_DIM,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        projector_hidden_size=32,
        text_hidden_size=32,
    )
    attn = ref.MiniMaxM3VisionEncoder(config).init_random_weights(seed=0).layers[0].self_attn
    attn = attn.to(torch.bfloat16).double()
    grid = torch.tensor(grid_thw)
    num_patches = int(grid.prod(dim=-1).sum())
    x = torch.randn(1, num_patches, config.hidden_size, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    cos, sin = ref.rope_cos_sin(grid, HEAD_DIM, THETA, MERGE)
    expected = attn(x, cos.double(), sin.double())[0]

    q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
    wqkv = _pack_wqkv_weight(q_proj.weight, k_proj.weight, v_proj.weight, num_heads, HEAD_DIM, PADDED_HEAD_DIM)
    bqkv = _pack_wqkv_bias(q_proj.bias, k_proj.bias, v_proj.bias, num_heads, HEAD_DIM, PADDED_HEAD_DIM)
    wo = _pad_wo_weight(attn.out_proj.weight, num_heads, HEAD_DIM, PADDED_HEAD_DIM)
    xqkv = x[0] @ wqkv.double() + bqkv.double()
    # nlp_create_qkv_heads: [q heads | k heads | v heads] -> 3 x (heads, L, 96)
    q, k, v = xqkv.view(num_patches, 3, num_heads, PADDED_HEAD_DIM).permute(1, 2, 0, 3)
    cos_p, sin_p = rope_cos_sin_padded(
        grid, head_dim=HEAD_DIM, padded_head_dim=PADDED_HEAD_DIM, theta=THETA, spatial_merge_size=MERGE
    )
    q, k = (rotary_embedding_llama(t, cos_p.double(), sin_p.double()) for t in (q, k))
    attn_heads = F.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)  # (heads, L, 96)
    # nlp_concat_heads, then the output projection
    out = attn_heads.transpose(0, 1).reshape(num_patches, -1) @ wo.double() + attn.out_proj.bias
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_vision_rope_tables_kept_per_grid(monkeypatch):
    """M3VLVisionModel stages the cos and sin of a grid with its first image and gives the blocks the same tables for
    the next images of that grid. Each image gets the tables of its own grid, also right after an image of the same
    patch count. The model keeps the ROPE_CACHE_GRIDS grids used last, frees the two tables of the grid it drops and
    nothing else, and stages that grid again for its next image; _stage_rope gives its caller new tables. No device:
    ttnn.from_torch, ttnn.deallocate and the modules are stand-ins."""
    tables, freed = [], []  # the (cos, sin) each forward gives the blocks; what ttnn.deallocate gets

    def block(x, cos, sin):
        tables.append((cos, sin))
        return x

    monkeypatch.setattr(ttnn, "from_torch", lambda tensor, **kwargs: tensor.clone())
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    monkeypatch.setattr("models.demos.minimax_m3.tt.vision.model.mesh_mapper", lambda mesh_device: None)
    monkeypatch.setattr("models.demos.minimax_m3.tt.vision.rope.mesh_mapper", lambda mesh_device: None)
    model = M3VLVisionModel(
        None, MiniMaxM3VLModelArgs(), patch_embed=lambda x: x, pre_layrnorm=lambda x: x, blocks=[block]
    )

    def forward(grid):
        """The tables the blocks get for an image of grid, which must be the grid's own and not freed."""
        model.forward_tower(torch.zeros(grid[1] * grid[2], 1176), torch.tensor([grid]))
        expected = rope_cos_sin_padded(torch.tensor([grid]), HEAD_DIM, PADDED_HEAD_DIM, THETA, MERGE)
        for table, table_ref in zip(tables[-1], expected):
            assert torch.equal(table, table_ref.to(torch.bfloat16)[None, None]), f"{grid} got another grid's tables"
            assert all(table is not f for f in freed), f"{grid} got freed tables"
        return tables[-1]

    def same(a, b):
        return len(a) == len(b) and all(x is y for x, y in zip(a, b))

    # One grid more than the model keeps, each next to its transpose: the same patch count, other tables.
    sides = [2 * n for n in range(2, ROPE_CACHE_GRIDS + 2)]
    grids = [grid for side in sides for grid in ((1, 2, side), (1, side, 2))][: ROPE_CACHE_GRIDS + 1]
    first = forward(grids[0])
    for grid in grids[1:-1]:
        forward(grid)
    assert same(forward(grids[0]), first) and not freed
    forward(grids[-1])  # drops grids[1], the least recently used
    assert same(freed, tables[1]), "freed other tables than those of the grid used least recently"
    assert same(forward(grids[0]), first)
    forward(grids[1])  # stages grids[1] again and drops grids[2]
    assert same(freed[2:], tables[2]), "freed other tables than those of the grid used least recently"
    staged = model._stage_rope(torch.tensor([grids[0]]))
    assert all(table is not kept and torch.equal(table, kept) for table, kept in zip(staged, first))
    assert len(freed) == 4
