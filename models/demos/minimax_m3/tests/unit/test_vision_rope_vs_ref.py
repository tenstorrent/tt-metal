# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side checks of the 3D vision RoPE's layout on device (tt/vision/rope.py and the q and k packing in
tt/vision/attention.py) vs Hugging Face's minimax_m3_vl, and of the model's RoPE tables per grid. No device.

The device rotates q and k with ttnn.experimental.rotary_embedding_llama, which pairs lanes (2i, 2i+1) of a head, so
the q and k rows of wqkv are packed in the head order of attention._rope_head_order, padded from 80 to 96, and
rope.py lays out the cos and sin in that order. The tests check, for square, non-square, two-frame and two-image grids
and one with a side of 2 patches:
  - rope_cos_sin, from which rope.py makes the device's tables, equals Hugging Face's MiniMaxM3VL3DRotaryEmbedding bit
    for bit, for the checkpoint's head_dim 80 and for 64 with another theta;
  - the op's rotation, run on host, of q and k in that order with the cos and sin of rope_cos_sin_padded equals
    Hugging Face's rotation (apply_rotary_pos_emb_vision with MiniMaxM3VL3DRotaryEmbedding's cos and sin) bit for
    bit, keeps the 16 pad dims at zero and leaves q k^T as Hugging Face's;
  - the attention computed on host from M3VLAttention's packed weights (q and k rows reordered) equals Hugging Face's
    MiniMaxM3VLVisionAttention with eager attention;
and that M3VLVisionModel keeps the cos and sin of a grid in DRAM for the next images of that grid and copies them to L1
once per forward for its blocks.

Needs transformers with minimax_m3_vl (tt-metal pins 5.12.1) for the checks against Hugging Face. These tests have no
mesh id; the 1-card CI leg selects them by name (its -k includes "test_vision_").

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_rope_vs_ref.py
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.tensor_utils import get_rot_transformation_mat
from models.demos.minimax_m3.tt.vision.attention import (
    _pack_wqkv_bias,
    _pack_wqkv_weight,
    _pad_wo_weight,
    _rope_head_order,
)
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.model import ROPE_CACHE_GRIDS, M3VLVisionModel
from models.demos.minimax_m3.tt.vision.rope import rope_cos_sin, rope_cos_sin_padded

from .vision_checkpoint import random_vision_weights
from .vision_helpers import hf_vision_modules

HEAD_DIM, PADDED_HEAD_DIM, NUM_HEADS, THETA, MERGE = 80, 96, 16, 10000.0, 2
# (head_dim, theta) of the check against Hugging Face's cos and sin: the checkpoint's head_dim, which rotates 78 dims
# and passes 2 through, and 64, which splits the rotary dims over T, H and W otherwise (60 rotated, 4 passed through).
ROPE_DIMS = {"hd80": (HEAD_DIM, THETA), "hd64": (64, 5000.0)}

# (t, h, w) patch grid of each image in the sequence.
ROPE_GRIDS = {
    "h16w16": [(1, 16, 16)],
    "h32w32": [(1, 32, 32)],
    "h48w48": [(1, 48, 48)],
    "h34w46": [(1, 34, 46)],
    "h46w34": [(1, 46, 34)],
    "t2h16w16": [(2, 16, 16)],
    "two_images": [(1, 16, 16), (1, 34, 46)],
    "h2w8": [(1, 2, 8)],  # 28x112, the thinnest image of the processor's smallest size: one row of 2x2 blocks
}
parametrize_rope_grids = pytest.mark.parametrize("grid_thw", list(ROPE_GRIDS.values()), ids=list(ROPE_GRIDS))


@pytest.fixture(scope="module")
def hf_vision():
    """Hugging Face's minimax_m3_vl modeling module."""
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl

    return modeling_minimax_m3_vl


def hf_rope(hf_vision, grid: torch.Tensor, head_dim: int = HEAD_DIM, theta: float = THETA) -> tuple:
    """Hugging Face's cos and sin of the 3D RoPE for grid, each (L, rotated dims) in fp32: 78 dims for head_dim 80."""
    rotary = hf_vision.MiniMaxM3VL3DRotaryEmbedding(head_dim, theta=theta, spatial_merge_size=MERGE)
    return rotary(grid, device=torch.device("cpu"), dtype=torch.float32)


def rotary_embedding_llama(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ttnn.experimental.rotary_embedding_llama in prefill mode, on host: x * cos + (x @ trans_mat) * sin, with the
    op's one-tile transformation matrix applied to each 32-lane tile of the head dim."""
    trans_mat = get_rot_transformation_mat()[0, 0].to(x.dtype)  # (32, 32)
    rotated = (x.unflatten(-1, (-1, trans_mat.shape[0])) @ trans_mat).flatten(-2)
    return x * cos + rotated * sin


@torch.no_grad()
@parametrize_rope_grids
@pytest.mark.parametrize("head_dim, theta", list(ROPE_DIMS.values()), ids=list(ROPE_DIMS))
def test_vision_rope_vs_hf(head_dim, theta, grid_thw, hf_vision):
    """rope_cos_sin equals Hugging Face's MiniMaxM3VL3DRotaryEmbedding, bit for bit."""
    grid = torch.tensor(grid_thw)
    hf_cos, hf_sin = hf_rope(hf_vision, grid, head_dim, theta)
    cos, sin = rope_cos_sin(grid, head_dim, theta, MERGE)
    torch.testing.assert_close(cos, hf_cos, rtol=0, atol=0, msg=lambda m: f"cos: {m}")
    torch.testing.assert_close(sin, hf_sin, rtol=0, atol=0, msg=lambda m: f"sin: {m}")


@torch.no_grad()
@parametrize_rope_grids
def test_vision_rope_padded_vs_ref(grid_thw, hf_vision):
    """The RoPE op's rotation of q and k in the device head order equals Hugging Face's rotation, bit for bit, with
    zero pad dims and Hugging Face's q k^T."""
    grid = torch.tensor(grid_thw)
    num_patches = int(grid.prod(dim=-1).sum())
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, generator=generator)
    k = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, generator=generator)
    ref_q, ref_k = hf_vision.apply_rotary_pos_emb_vision(q, k, *hf_rope(hf_vision, grid))

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

    # q k^T over the 96 lanes against Hugging Face's over its 80 dims, for the first 128 queries, in fp64: the
    # products are exact there, so only the summation order differs.
    scores = rotated["q"][:, :, :128].double() @ rotated["k"].double().transpose(-2, -1)
    ref_scores = ref_q.transpose(1, 2)[:, :, :128].double() @ ref_k.transpose(1, 2).double().transpose(-2, -1)
    torch.testing.assert_close(scores, ref_scores, rtol=1e-12, atol=1e-12)


@torch.no_grad()
@pytest.mark.parametrize("grid_thw", [ROPE_GRIDS["h16w16"], ROPE_GRIDS["two_images"]], ids=["h16w16", "two_images"])
def test_vision_attention_layout_vs_ref(grid_thw, hf_vision):
    """The attention computed from M3VLAttention's packed weights equals Hugging Face's attention.

    Runs what the device runs, on host in fp64: the QKV linear with the packed weight and bias (q and k rows in the
    RoPE op's head order, heads padded to 96), the head split, the RoPE op on q and k, SDPA over the 96 lanes with
    scale 1/sqrt(80), the head concat and the output projection with its padded weight. Hugging Face's layer has
    bf16-valued weights, which packing keeps exactly; its softmax runs in fp32, hence the tolerance.
    """
    num_heads = 2
    config = MiniMaxM3VisionConfig(
        hidden_size=num_heads * HEAD_DIM,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        projector_hidden_size=32,
        text_hidden_size=32,
    )
    tower, _ = hf_vision_modules(config, random_vision_weights(config))
    attn = tower.layers[0].self_attn.to(torch.bfloat16).double()
    grid = torch.tensor(grid_thw)
    num_patches = int(grid.prod(dim=-1).sum())
    x = torch.randn(1, num_patches, config.hidden_size, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    cos, sin = hf_rope(hf_vision, grid)
    expected, _ = attn(x, position_embeddings=(cos.double(), sin.double()))  # the output and the attention weights
    expected = expected[0]

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
    # the heads side by side, as SDPA writes them, then the output projection
    out = attn_heads.transpose(0, 1).reshape(num_patches, -1) @ wo.double() + attn.out_proj.bias
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_vision_rope_tables_kept_per_grid(monkeypatch):
    """M3VLVisionModel stages the cos and sin of a grid with its first image and gives the blocks copies of the same
    tables for the next images of that grid. Each image gets the tables of its own grid, also right after an image of
    the same patch count. The model keeps the ROPE_CACHE_GRIDS grids used last, frees the two tables of the grid it
    drops and no other cached table, and stages that grid again for its next image; _stage_rope gives its caller new
    tables. No device: ttnn.from_torch, ttnn.to_memory_config, ttnn.deallocate and the modules are stand-ins."""
    tables, freed = [], []  # the cached (cos, sin) whose copies each forward gives the blocks; the cached tables freed
    copies = {}  # id of each L1 copy not freed yet -> the table it copies

    def block(x, cos, sin):
        tables.append((copies[id(cos)], copies[id(sin)]))
        return x

    def to_memory_config(tensor, memory_config):
        copy = tensor.clone()
        copies[id(copy)] = tensor
        return copy

    def deallocate(tensor):
        if id(tensor) in copies:
            del copies[id(tensor)]
        else:
            freed.append(tensor)

    monkeypatch.setattr(ttnn, "from_torch", lambda tensor, **kwargs: tensor.clone())
    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
    monkeypatch.setattr(ttnn, "deallocate", deallocate)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: None)
    config = MiniMaxM3VisionConfig()
    model = M3VLVisionModel(
        None, config, PADDED_HEAD_DIM, patch_embed=lambda x: x, pre_layrnorm=lambda x: x, blocks=[block]
    )

    def forward(grid):
        """The tables the blocks get for an image of grid, which must be the grid's own and not freed."""
        model.forward_tower(torch.zeros(grid[1] * grid[2], config.patch_dim), torch.tensor([grid]))
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


def test_vision_rope_tables_copied_to_l1(monkeypatch):
    """One forward copies the cached cos and sin of its grid to L1 once, before the first block, gives the two copies
    to every block and frees them after the last block. The cache keeps the two tables it uploaded to DRAM, not freed,
    and the next image of the grid copies the same tables. No device: ttnn.from_torch, ttnn.to_memory_config,
    ttnn.deallocate and the modules are stand-ins."""
    uploads, calls = [], []  # (tensor, memory config) of each upload; the copies, blocks and frees in call order

    def from_torch(tensor, **kwargs):
        uploads.append((tensor.clone(), kwargs["memory_config"]))
        return uploads[-1][0]

    def to_memory_config(tensor, memory_config):
        calls.append(("copy", tensor, memory_config, tensor.clone()))
        return calls[-1][3]

    def block(index):
        def forward(x, cos, sin):
            calls.append(("block", index, cos, sin))
            return x

        return forward

    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: calls.append(("free", tensor)))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: None)
    config = MiniMaxM3VisionConfig()
    blocks = [block(index) for index in range(3)]
    model = M3VLVisionModel(
        None, config, PADDED_HEAD_DIM, patch_embed=lambda x: x, pre_layrnorm=lambda x: x, blocks=blocks
    )
    grid, copied = (1, 4, 6), None
    for image in (1, 2):
        calls.clear()
        model.forward_tower(torch.zeros(grid[1] * grid[2], config.patch_dim), torch.tensor([grid]))
        kinds = [call[0] for call in calls]
        assert kinds == ["copy"] * 2 + ["block"] * len(blocks) + ["free"] * 2, f"image {image}: {kinds}"
        (_, cos, cos_memory_config, cos_l1), (_, sin, sin_memory_config, sin_l1) = calls[:2]
        assert cos_memory_config == ttnn.L1_MEMORY_CONFIG and sin_memory_config == ttnn.L1_MEMORY_CONFIG
        assert all(
            call[1] == index and call[2] is cos_l1 and call[3] is sin_l1 for index, call in enumerate(calls[2:-2])
        ), f"image {image}: a block got other tables than the L1 copies"
        freed = [call[1] for call in calls[-2:]]
        assert any(t is cos_l1 for t in freed) and any(t is sin_l1 for t in freed), f"image {image}: freed {freed}"
        cached = model._rope_cache[grid]
        assert cos is cached[0] and sin is cached[1], f"image {image}: copied other tables than the cached ones"
        assert [memory_config for t, memory_config in uploads if t is cos or t is sin] == [ttnn.DRAM_MEMORY_CONFIG] * 2
        assert copied is None or (cos is copied[0] and sin is copied[1]), "the second image copied other tables"
        copied = cached
