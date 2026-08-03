# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL vision tower.

The tower turns raw patches into `out_hidden_size` tokens for the text decoder, and additionally
exports intermediate features from `deepstack_visual_indexes` that the decoder adds into its first few
layers. MiniMax-H3 needs it for the `fl2va` and `ref2va` tasks; `t2va` passes no pixels and never
reaches it.

Currently ported: the input stage (patch embedding and interpolated position embedding). The blocks,
attention and mergers are still to come.

Two things about this tower differ from the text encoder in `model_qwen3vl.py`:

- `head_dim` is `1152 // 16 == 72`, which is not tile-aligned. `ttnn` SDPA rejects it outright
  (`TT_FATAL logical_shape[3] == legacy_shape[3]`), so the projections must be padded to 96 and the
  softmax `scale` passed explicitly as `72 ** -0.5` -- the padding zeros must not change the
  temperature.
- Norms are `LayerNorm`, attention is plain MHA with biases, and there is no QK-norm.
"""

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module
from ...layers.normalization import LayerNorm

# `ttnn` SDPA requires a tile-aligned head dimension.
_TILE = 32


def vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    *,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(4, total_patches)` corner indices into the position table, and their bilinear weights.

    The table is a fixed `num_grid_per_side ** 2` grid (48x48 for MiniMax-H3's conditioner), so any
    canvas that is not exactly that shape is interpolated onto it. That is the common case rather than
    an edge case: a 768x1344 keyframe is 48x84 patches against a 48x48 table.

    The `reorder` step is what puts patches into `spatial_merge_size` blocks, matching the order the
    merger later consumes them in.
    """
    side = num_grid_per_side
    merge_size = spatial_merge_size
    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)

        h_grid = torch.linspace(0, side - 1, h)
        w_grid = torch.linspace(0, side - 1, w)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]

        h_idx = torch.arange(h).view(h // merge_size, merge_size)
        w_idx = torch.arange(w).view(w // merge_size, merge_size)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)

        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    return (
        torch.stack([torch.cat(p) for p in idx_parts]),
        torch.stack([torch.cat(p) for p in weight_parts]),
    )


def vision_pos_embeds(
    pos_embed_weight: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """The `(total_patches, hidden_size)` position embedding to add after the patch embedding.

    Computed on the host and uploaded, the way [`create_rope_tensors`] supplies the decoder's rotary
    tensors: this is a pure function of `grid_thw` and the position table -- no pixels enter -- so
    there is nothing for the device to do that host arithmetic does not already settle.
    """
    indices, weights = vision_bilinear_indices_and_weights(
        grid_thw, num_grid_per_side=num_grid_per_side, spatial_merge_size=spatial_merge_size
    )
    return (pos_embed_weight[indices] * weights[:, :, None]).sum(0)


class Qwen3VlVisionPatchEmbed(Module):
    """Patch embedding of the vision tower.

    The reference is a `Conv3d` whose kernel equals its stride over patches that the processor has
    already flattened, which makes it a plain matmul: each `(in_channels, temporal_patch_size,
    patch_size, patch_size)` patch is one independent row. Verified equal to the reference `Conv3d` to
    3.8e-6 in fp32 (accumulation order only), which is far below bf16 resolution. So this is a
    `Linear` and `_prepare_torch_state` flattens the convolution weight into it.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.hidden_size = hidden_size
        self.proj = Linear(self.patch_dim, hidden_size, bias=True, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # `proj.weight` arrives as the Conv3d kernel (hidden, in_ch, t, p, p); flatten the spatial and
        # channel axes so it multiplies the already-flattened patch rows.
        weight = state.get("proj.weight")
        if weight is not None and weight.ndim > 2:
            state["proj.weight"] = weight.reshape(self.hidden_size, self.patch_dim)

    def forward(self, patches: ttnn.Tensor) -> ttnn.Tensor:
        """`(total_patches, patch_dim)` -> `(total_patches, hidden_size)`."""
        return self.proj.forward(patches)


def vision_rope_position_ids(grid_thw: torch.Tensor, *, spatial_merge_size: int) -> torch.Tensor:
    """`(total_patches, 2)` `(row, col)` positions for the tower's 2-D rotary embedding.

    Note this is a *different* grid from the decoder's 3-axis M-RoPE in `model_qwen3vl.py`: the tower
    rotates over spatial position within one image only, and has no temporal axis -- frames of a video
    repeat the same `(row, col)` grid, since each frame is its own attention block.

    The reshape/transpose puts positions into `spatial_merge_size` blocks, matching the patch order the
    merger consumes.
    """
    out = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        m = spatial_merge_size
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos = hpos.reshape(h // m, m, w // m, m).transpose(1, 2).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos = wpos.reshape(h // m, m, w // m, m).transpose(1, 2).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def vision_rope_tensors(
    grid_thw: torch.Tensor,
    *,
    head_dim: int,
    spatial_merge_size: int,
    rope_theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` of shape `(total_patches, head_dim)` for the tower's rotary embedding.

    The two position axes each contribute `head_dim // 4` frequencies, giving `head_dim // 2` before
    the `rotate_half` duplication. Built on the host and uploaded, as elsewhere in this port.
    """
    position_ids = vision_rope_position_ids(grid_thw, spatial_merge_size=spatial_merge_size)
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = (position_ids.unsqueeze(-1) * inv_freq).flatten(1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _pad_head_dim(weight: torch.Tensor, *, num_heads: int, head_dim: int, padded: int, axis: int) -> torch.Tensor:
    """Zero-pad a per-head weight or bias from `head_dim` to `padded` along `axis`.

    `axis=0` for the output rows of q/k/v and for biases, `axis=1` for the input columns of the output
    projection. The zeros contribute nothing to the dot products, which is why the softmax `scale` must
    still be computed from the *true* head_dim -- see [`Qwen3VlVisionAttention`].
    """
    if head_dim == padded:
        return weight
    if axis == 0:
        shaped = weight.reshape(num_heads, head_dim, *weight.shape[1:])
        pad = [0] * (2 * (shaped.ndim - 2)) + [0, padded - head_dim]
        return torch.nn.functional.pad(shaped, pad).reshape(num_heads * padded, *weight.shape[1:])
    shaped = weight.reshape(weight.shape[0], num_heads, head_dim)
    return torch.nn.functional.pad(shaped, (0, padded - head_dim)).reshape(weight.shape[0], num_heads * padded)


class Qwen3VlVisionMLP(Module):
    """Two biased linears with a GELU between. `hidden_act` is `gelu_pytorch_tanh` for this config."""

    def __init__(self, *, hidden_size: int, intermediate_size: int, hidden_act: str, mesh_device) -> None:
        super().__init__()
        self.linear_fc1 = Linear(hidden_size, intermediate_size, bias=True, mesh_device=mesh_device)
        self.linear_fc2 = Linear(intermediate_size, hidden_size, bias=True, mesh_device=mesh_device)
        self._act = hidden_act

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.linear_fc1.forward(x)
        x = ttnn.gelu(x) if self._act.startswith("gelu") else ttnn.silu(x)
        return self.linear_fc2.forward(x)


class Qwen3VlVisionAttention(Module):
    """Plain MHA over one attention block, with biases and no QK-norm.

    `head_dim` is `hidden_size // num_heads == 72` for this config, which is not tile-aligned. `ttnn`
    SDPA rejects it outright (`TT_FATAL logical_shape[3] == legacy_shape[3]`), so q/k/v/o are padded to
    96 at load time and `scale` is passed explicitly as `72 ** -0.5`. Leaving `scale` to SDPA would
    make it `96 ** -0.5` and silently change the softmax temperature -- wrong output, not a crash.

    One image is one attention block: `cu_seqlens = repeat_interleave(h*w, t).cumsum()`, so a single
    image gives `[0, h*w]` and plain full attention is exact. Multiple images or video frames need
    block-diagonal masking, which is not implemented here yet -- `forward` takes one block.
    """

    def __init__(self, *, hidden_size: int, num_heads: int, mesh_device) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.padded_head_dim = math.ceil(self.head_dim / _TILE) * _TILE
        self.scale = self.head_dim**-0.5

        self.inner = num_heads * self.padded_head_dim
        self.qkv = Linear(hidden_size, 3 * self.inner, bias=True, mesh_device=mesh_device)
        self.proj = Linear(self.inner, hidden_size, bias=True, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        kw = dict(num_heads=self.num_heads, head_dim=self.head_dim, padded=self.padded_head_dim)
        if (w := state.get("qkv.weight")) is not None:
            # the reference packs [q|k|v] on the output axis; pad each third's heads independently
            state["qkv.weight"] = torch.cat([_pad_head_dim(part, axis=0, **kw) for part in w.chunk(3, dim=0)])
        if (b := state.get("qkv.bias")) is not None:
            state["qkv.bias"] = torch.cat([_pad_head_dim(part, axis=0, **kw) for part in b.chunk(3, dim=0)])
        if (w := state.get("proj.weight")) is not None:
            state["proj.weight"] = _pad_head_dim(w, axis=1, **kw)

    def forward(self, hidden_states: ttnn.Tensor, *, pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        qkv = self.qkv.forward(hidden_states)

        q, k, v = (
            ttnn.permute(
                ttnn.reshape(part, (1, seq_len, self.num_heads, self.padded_head_dim)),
                (0, 2, 1, 3),
            )
            # `ttnn.split` takes a chunk *size*, as torch does -- passing 3 would give `inner`
            # slices of width 3 rather than the three [q | k | v] blocks.
            for part in ttnn.split(qkv, self.inner, dim=-1)
        )

        cos, sin = pos_embeds
        q = _apply_vision_rope(q, cos, sin, self.head_dim)
        k = _apply_vision_rope(k, cos, sin, self.head_dim)

        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), (seq_len, self.num_heads * self.padded_head_dim))
        return self.proj.forward(attn)


def _apply_vision_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, head_dim: int) -> ttnn.Tensor:
    """Rotate the leading `head_dim` channels of each head, leaving the padding tail untouched.

    The padded channels carry no position, so rotating them would mix zeros into the rotation and is
    simply skipped.
    """
    rot, tail = x[..., :head_dim], x[..., head_dim:]
    half = head_dim // 2
    rotated = ttnn.concat([ttnn.neg(rot[..., half:]), rot[..., :half]], dim=-1)
    out = ttnn.add(ttnn.mul(rot, cos), ttnn.mul(rotated, sin))
    return ttnn.concat([out, tail], dim=-1) if tail.shape[-1] else out


class Qwen3VlVisionBlock(Module):
    """Pre-norm attention and MLP with `LayerNorm` (eps 1e-6), not the decoder's RMSNorm."""

    def __init__(
        self, *, hidden_size: int, num_heads: int, intermediate_size: int, hidden_act: str, norm_eps: float, mesh_device
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, norm_eps=norm_eps, mesh_device=mesh_device)
        self.attn = Qwen3VlVisionAttention(hidden_size=hidden_size, num_heads=num_heads, mesh_device=mesh_device)
        self.norm2 = LayerNorm(hidden_size, norm_eps=norm_eps, mesh_device=mesh_device)
        self.mlp = Qwen3VlVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            mesh_device=mesh_device,
        )

    def forward(self, hidden_states: ttnn.Tensor, *, pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        hidden_states = hidden_states + self.attn.forward(self.norm1.forward(hidden_states), pos_embeds=pos_embeds)
        return hidden_states + self.mlp.forward(self.norm2.forward(hidden_states))
