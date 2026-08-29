# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL vision tower.

The tower turns raw patches into `out_hidden_size` tokens for the text decoder, and additionally
exports intermediate features from `deepstack_visual_indexes` that the decoder adds into its first few
layers. MiniMax-H3 needs it for the `fl2va` and `ref2va` tasks; `t2va` passes no pixels and never
reaches it.

The whole tower is ported. `forward` returns the merged output tokens plus one deepstack feature per
entry of `deepstack_visual_indexes`, matching `Qwen3VLVisionModel`'s `pooler_output` and
`deepstack_features`.

Two things about this tower differ from the text encoder in `model_qwen3vl.py`:

- `head_dim` is `1152 // 16 == 72`, which is not tile-aligned. `ttnn` SDPA rejects it outright
  (`TT_FATAL logical_shape[3] == legacy_shape[3]`), so the projections must be padded to 96 and the
  softmax `scale` passed explicitly as `72 ** -0.5` -- the padding zeros must not change the
  temperature.
- Norms are `LayerNorm`, attention is plain MHA with biases, and there is no QK-norm.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import torch
from loguru import logger

import ttnn

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import LayerNorm
from ...utils.mochi import get_rot_transformation_mat
from ...utils.tensor import bf16_tensor, typed_tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...parallel.config import EncoderParallelConfig
    from ...parallel.manager import CCLManager

# `ttnn` SDPA requires a tile-aligned head dimension.
_TILE = 32


class VisionParallel(NamedTuple):
    """Resolved tensor- and sequence-parallel placement for the tower.

    Separate from `EncoderParallelConfig` because that carries mesh axes while every module here needs
    the *factors* too, and because the replicated case (both factors 1) must stay a plain `Linear`
    path: gating on the factor keeps an unparallelized run bit-identical to the single-device module.
    """

    tp_axis: int | None = None
    tp_factor: int = 1
    sp_axis: int | None = None
    sp_factor: int = 1
    ccl_manager: CCLManager | None = None

    @property
    def tp(self) -> bool:
        return self.tp_factor > 1

    @property
    def sp(self) -> bool:
        return self.sp_factor > 1


def resolve_vision_parallel(
    mesh_device: ttnn.MeshDevice,
    parallel_config: EncoderParallelConfig | None,
    ccl_manager: CCLManager | None,
) -> VisionParallel:
    """`VisionParallel` from the pipeline's `EncoderParallelConfig`.

    `sequence_parallel` is optional on that config; the tower is its only encoder-side consumer. A
    factor above 1 on either axis requires a `ccl_manager`, since both paths reduce or gather across
    devices.
    """
    if parallel_config is None:
        return VisionParallel(ccl_manager=ccl_manager)

    tp, sp = parallel_config.tensor_parallel, parallel_config.sequence_parallel
    resolved = VisionParallel(
        tp_axis=tp.mesh_axis if tp is not None else None,
        tp_factor=tp.factor if tp is not None else 1,
        sp_axis=sp.mesh_axis if sp is not None else None,
        sp_factor=sp.factor if sp is not None else 1,
        ccl_manager=ccl_manager,
    )
    if (resolved.tp or resolved.sp) and ccl_manager is None:
        msg = "ccl_manager is required when tensor- or sequence-parallel factor is greater than 1"
        raise ValueError(msg)
    if resolved.tp and resolved.sp and resolved.tp_axis == resolved.sp_axis:
        msg = f"tensor and sequence parallelism cannot share mesh axis {resolved.tp_axis}"
        raise ValueError(msg)
    for name, axis, factor in (
        ("tensor", resolved.tp_axis, resolved.tp_factor),
        ("sequence", resolved.sp_axis, resolved.sp_factor),
    ):
        if factor > 1 and tuple(mesh_device.shape)[axis] != factor:
            msg = (
                f"{name}-parallel factor {factor} does not match mesh axis {axis} "
                f"of shape {tuple(mesh_device.shape)}"
            )
            raise ValueError(msg)
    return resolved


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


def vision_cu_seqlens(grid_thw: torch.Tensor) -> tuple[int, ...]:
    """The attention block boundaries of a batch of images and videos.

    `cu_seqlens = repeat_interleave(h * w, t).cumsum()` padded with a leading 0, matching the
    reference. One image is one block; a video is one block *per frame*, since `t` repeats its spatial
    extent. Attention never crosses a boundary, so `fl2va` (a single image) collapses to plain full
    attention while `ref2va` needs the split.
    """
    bounds = [0]
    for t, h, w in grid_thw.tolist():
        for _ in range(int(t)):
            bounds.append(bounds[-1] + int(h) * int(w))
    return tuple(bounds)


def pad_patches_for_sp(
    patches: torch.Tensor,
    pos_embeds: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: Sequence[int],
    *,
    sp_factor: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, ...], int]:
    """Pad a patch batch so its SP shards are tile-aligned, isolating the pad in a phantom window.

    SP requires `total % (sp_factor * 32) == 0` (see `Qwen3VlVisionAttention`), which production grids
    do not always satisfy -- e.g. two_refs' 38,144 patches divide sp=4/8's alignment but not sp=32's
    1024. Unlike the decoder, the tower cannot pad blindly: attention is non-causal block-diagonal
    (tail pad would join the last image's window) and the merger folds consecutive 4-row groups. So
    the pad rows are appended as their OWN attention window -- one extra `cu_seqlens` boundary -- and
    flow through as isolated garbage that `Qwen3VlVisionModel.forward(logical_patches=...)` trims
    after the SP gather, before any consumer sees the tokens.

    Zero patches and pos_embeds, identity rope rows (cos=1, sin=0). The pad count is a multiple of
    `sp_factor * 32`, hence of the merge group, so the garbage folds into whole tail tokens. Returns
    `(patches, pos_embeds, rope, cu_seqlens, logical_patches)`; a no-op (inputs returned as-is) when
    the count is already aligned.
    """
    total = patches.shape[0]
    mult = sp_factor * _TILE
    padded = -(-total // mult) * mult
    if padded == total:
        return patches, pos_embeds, rope, tuple(cu_seqlens), total
    if cu_seqlens[-1] != total:
        msg = f"cu_seqlens must span [0, {total}], got {cu_seqlens[0]}..{cu_seqlens[-1]}"
        raise ValueError(msg)
    npad = padded - total
    cos, sin = rope
    return (
        torch.nn.functional.pad(patches, (0, 0, 0, npad)),
        torch.nn.functional.pad(pos_embeds, (0, 0, 0, npad)),
        (
            torch.cat([cos, torch.ones(npad, cos.shape[-1], dtype=cos.dtype)], dim=0),
            torch.cat([sin, torch.zeros(npad, sin.shape[-1], dtype=sin.dtype)], dim=0),
        ),
        (*tuple(cu_seqlens), padded),
        total,
    )


def vision_rope_tensors(
    grid_thw: torch.Tensor,
    *,
    head_dim: int,
    spatial_merge_size: int,
    rope_theta: float = 10000.0,
    padded_head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` of shape `(total_patches, padded_head_dim)` for the tower's rotary embedding.

    The two position axes each contribute `head_dim // 4` frequencies, giving `head_dim // 2` distinct
    values. These are laid out **interleaved** -- channels `(2j, 2j+1)` share frequency `j` -- to feed
    `ttnn.experimental.rotary_embedding_llama`, whose tile transformation matrix rotates adjacent pairs.
    This matches the SPLIT->INTERLEAVED permute applied to the q/k projection at load
    (`_rope_permute_qk`); the halves layout the reference uses would need the unpermuted weights.

    Padded from `head_dim` to `padded_head_dim` with an **identity tail** (cos=1, sin=0), so the zero
    channels `_pad_head_dim` appended to each head pass through the rotation untouched. Built on the
    host and uploaded, as elsewhere in this port.
    """
    padded_head_dim = padded_head_dim or math.ceil(head_dim / _TILE) * _TILE
    position_ids = vision_rope_position_ids(grid_thw, spatial_merge_size=spatial_merge_size)
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = (position_ids.unsqueeze(-1) * inv_freq).flatten(1)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    cos, sin = emb.cos(), emb.sin()
    pad = padded_head_dim - head_dim
    if pad:
        cos = torch.nn.functional.pad(cos, (0, pad), value=1.0)
        sin = torch.nn.functional.pad(sin, (0, pad), value=0.0)
    return cos, sin


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


def _rope_permute_qk(t: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder each head's output channels SPLIT-rotation -> INTERLEAVED for `rotary_embedding_llama`.

    Moves the SPLIT pair `(i, i + head_dim // 2)` to adjacent interleaved slots `(2i, 2i + 1)`, so the
    op's tile transformation matrix (which rotates neighbours) reproduces the reference's `rotate_half`.
    Applied to the q and k projections (weight rows on `axis=0`, or bias) BEFORE `_pad_head_dim`, so the
    appended zeros stay the identity tail; v is left untouched. Mirrors the gemma connector's
    `_permute_qk` -- omitting it makes interleaved RoPE score against the wrong channels (PCC ~0.09).
    """
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    rest = t.shape[1:]
    return t.reshape(num_heads, head_dim, *rest).index_select(1, perm).reshape(num_heads * head_dim, *rest)


def _with_batch_axis(x: ttnn.Tensor) -> tuple[ttnn.Tensor, bool]:
    """`(x, added)` with a leading batch axis, because the CCL paths require rank >= 3.

    This tower carries 2-D `(rows, width)` activations end to end, unlike the decoder's
    `(batch, seq, hidden)`. `RowParallelLinear` reduce-scatters on `dim=3` after a SINGLE unsqueeze, so
    a 2-D input leaves it indexing past the end (`IndexError` in `get_rs_ping_pong_buffer`). Adding the
    axis here is local to this file rather than a change to shared `layers/linear.py`.
    """
    if len(x.shape) >= 3:
        return x, False
    return ttnn.reshape(x, (1, x.shape[0], x.shape[1])), True


def _drop_batch_axis(x: ttnn.Tensor, added: bool) -> ttnn.Tensor:
    return ttnn.reshape(x, (x.shape[-2], x.shape[-1])) if added else x


def _trim_tokens(x: ttnn.Tensor, real_tokens: int | None) -> ttnn.Tensor:
    """Drop the SP-alignment pad's merged garbage tokens from a gathered `(tokens, hidden)` tensor.

    Only valid AFTER the SP gather: the pad occupies the trailing rows of the trailing shard, so on
    the full sequence it is exactly the tail. A no-op when nothing was padded."""
    if real_tokens is None or x.shape[-2] <= real_tokens:
        return x
    return x[:real_tokens, :]


def _gather_hidden(x: ttnn.Tensor, p: VisionParallel) -> ttnn.Tensor:
    """All-gather a column-fractured activation back to full width, when TP is on."""
    if not p.tp:
        return x
    x, added = _with_batch_axis(x)
    x = p.ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=p.tp_axis, use_hyperparams=True)
    return _drop_batch_axis(x, added)


def _row_parallel_forward(linear, x: ttnn.Tensor, p: VisionParallel) -> ttnn.Tensor:
    """Run a row-parallel linear on a 2-D activation and gather its fractured result to full width.

    The reduce-scatter (inside `linear.forward`) splits on the HIDDEN dim, which pad-dances when
    `hidden/tp` is not a whole number of tiles. The block's `proj`/`fc2` use `_row_parallel_seq_forward`
    to avoid that; this hidden-dim path remains for the merger, whose output width is tile-aligned.
    """
    if not p.tp:
        return linear.forward(x)
    x, added = _with_batch_axis(x)
    out = p.ccl_manager.all_gather_persistent_buffer(
        linear.forward(x), dim=-1, mesh_axis=p.tp_axis, use_hyperparams=True
    )
    return _drop_batch_axis(out, added)


def _row_parallel_seq_forward(linear, x: ttnn.Tensor, p: VisionParallel) -> ttnn.Tensor:
    """Row-parallel linear whose all-reduce is split on the SEQUENCE dim, not the hidden dim.

    `reduce_scatter + all_gather` is an all-reduce on any axis, so this returns the identical
    full-width, full-sequence result as `_row_parallel_forward` -- but both collectives run on the
    tile-aligned sequence dim, dodging the `hidden/tp = 144` pad dance.

    The reduce-scatter needs `rows / tp_factor` to be a whole number of tiles. When the row count does
    not divide that way, the rows are zero-padded up to `tp_factor * TILE` here, purely for the
    collective, and the pad is sliced back off after the gather. An all-reduce is row-independent and
    the pad rows are zero, so real rows are untouched -- and the padding lives entirely inside this op,
    so attention never sees it (no sequence padding, no mask changes, no tower plumbing). Grids whose
    shard is already aligned add no pad rows and pay nothing.
    """
    if not p.tp:
        return linear.forward(x)
    x, added = _with_batch_axis(x)  # (rows, width) -> (1, rows, width); sequence is dim 1
    rows = x.shape[1]
    npad = (-rows) % (p.tp_factor * _TILE)
    if npad:
        x = ttnn.pad(x, [(0, 0), (0, npad), (0, 0)], value=0.0)
    # `reduce_scatter_dim=-2` targets the sequence axis of the `(1, rows, width)` activation (the
    # linear is rank-preserving; negative dims are rank-agnostic), leaving a `(1, rows/tp, width)`
    # shard. Gather it back on the same sequence axis (dim 1 here) to reconstruct the full reduced
    # sequence, then drop the pad rows.
    out = linear.forward(x, reduce_scatter_dim=-2)
    out = p.ccl_manager.all_gather_persistent_buffer(out, dim=1, mesh_axis=p.tp_axis, use_hyperparams=True)
    if npad:
        out = out[:, :rows, :]
    return _drop_batch_axis(out, added)


def _interleave_for_col_parallel(t: torch.Tensor, *, parts: int, tp_factor: int) -> torch.Tensor:
    """Reorder a `parts`-way concatenated weight/bias so column fracturing keeps each part per device.

    `qkv` is stored as `[q | k | v]` over the row axis. `ColParallelLinear` fractures `out_features`
    into `tp_factor` CONTIGUOUS chunks, which would hand device 0 the first eighth of `q` alone and no
    `k` or `v` at all. Grouping by device first -- `[q0 k0 v0][q1 k1 v1]...` -- makes the contiguous
    chunk the correct `(q, k, v)` triple for that device's heads.

    Operates on the torch layout (rows = out_features); `ColParallelLinear._prepare_torch_state`
    transposes afterwards.
    """
    if tp_factor == 1:
        return t
    rows = t.shape[0]
    if rows % (parts * tp_factor) != 0:
        msg = f"cannot interleave {rows} rows into {parts} parts across {tp_factor} devices"
        raise ValueError(msg)
    trailing = t.shape[1:]
    return t.reshape(parts, tp_factor, rows // (parts * tp_factor), *trailing).transpose(0, 1).reshape(rows, *trailing)


class Qwen3VlVisionMLP(Module):
    """Two biased linears with a GELU between. `hidden_act` is `gelu_pytorch_tanh` for this config."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mesh_device,
        parallel: VisionParallel | None = None,
        linear_compute_kernel_config=None,
    ) -> None:
        super().__init__()
        self._p = parallel or VisionParallel()
        self._act = hidden_act

        if self._p.tp:
            if intermediate_size % self._p.tp_factor != 0:
                msg = f"intermediate_size {intermediate_size} is not divisible by TP factor {self._p.tp_factor}"
                raise ValueError(msg)
            kw = dict(
                mesh_device=mesh_device,
                mesh_axis=self._p.tp_axis,
                ccl_manager=self._p.ccl_manager,
                compute_kernel_config=linear_compute_kernel_config,
            )
            self.linear_fc1 = ColParallelLinear(hidden_size, intermediate_size, bias=True, **kw)
            self.linear_fc2 = RowParallelLinear(intermediate_size, hidden_size, bias=True, **kw)
        else:
            kw = dict(mesh_device=mesh_device, compute_kernel_config=linear_compute_kernel_config)
            self.linear_fc1 = Linear(hidden_size, intermediate_size, bias=True, **kw)
            self.linear_fc2 = Linear(intermediate_size, hidden_size, bias=True, **kw)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.linear_fc1.forward(x)
        x = ttnn.gelu(x) if self._act.startswith("gelu") else ttnn.silu(x)
        # `RowParallelLinear` reduce-scatters, so its result is fractured on columns; the residual add
        # and the next LayerNorm both need the full width back (cf. `Qwen3VlMlp` in model_qwen3vl.py).
        # The sequence-dim all-reduce (aligned) replaces the hidden-dim one (the 144 pad dance).
        return _row_parallel_seq_forward(self.linear_fc2, x, self._p)


class Qwen3VlVisionAttention(Module):
    """Plain MHA over one attention block, with biases and no QK-norm.

    `head_dim` is `hidden_size // num_heads == 72` for this config, which is not tile-aligned. `ttnn`
    SDPA rejects it outright (`TT_FATAL logical_shape[3] == legacy_shape[3]`), so q/k/v/o are padded to
    96 at load time and `scale` is passed explicitly as `72 ** -0.5`. Leaving `scale` to SDPA would
    make it `96 ** -0.5` and silently change the softmax temperature -- wrong output, not a crash.

    Attention is confined to one image, or to one frame of a video: `cu_seqlens` names the boundaries
    (see [`vision_cu_seqlens`]). A single image is a single block, so `fl2va` reduces to plain full
    attention. Multiple blocks are handled by attending within each in turn rather than by a
    block-diagonal mask -- an `s x s` mask is 17 GiB for a full `ref2va` request of nine images and
    three videos, where a dozen or so smaller attentions cost nothing extra.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mesh_device,
        parallel: VisionParallel | None = None,
        linear_compute_kernel_config=None,
    ) -> None:
        super().__init__()
        self._p = parallel or VisionParallel()
        self.mesh_device = mesh_device
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.padded_head_dim = math.ceil(self.head_dim / _TILE) * _TILE
        self.scale = self.head_dim**-0.5

        self.inner = num_heads * self.padded_head_dim
        # TP shards HEADS (not head_dim): each device owns `num_heads // tp_factor` whole heads, so
        # SDPA needs no cross-device traffic for TP and the rotary tensors stay unsharded.
        if num_heads % self._p.tp_factor != 0:
            msg = f"num_heads {num_heads} is not divisible by TP factor {self._p.tp_factor}"
            raise ValueError(msg)
        self.num_local_heads = num_heads // self._p.tp_factor
        self.local_inner = self.num_local_heads * self.padded_head_dim

        # Ring SDPA runs its CCL workers on cores the compute kernel must not also own, so the last
        # column is reserved for them and `ccl_core_grid_offset` points at it (as in attention_wan.py).
        full_grid = mesh_device.compute_with_storage_grid_size()
        self._sdpa_worker_grid = (full_grid.x - 1, full_grid.y)

        if self._p.tp:
            kw = dict(
                mesh_device=mesh_device,
                mesh_axis=self._p.tp_axis,
                ccl_manager=self._p.ccl_manager,
                compute_kernel_config=linear_compute_kernel_config,
            )
            self.qkv = ColParallelLinear(hidden_size, 3 * self.inner, bias=True, **kw)
            self.proj = RowParallelLinear(self.inner, hidden_size, bias=True, **kw)
        else:
            kw = dict(mesh_device=mesh_device, compute_kernel_config=linear_compute_kernel_config)
            self.qkv = Linear(hidden_size, 3 * self.inner, bias=True, **kw)
            self.proj = Linear(self.inner, hidden_size, bias=True, **kw)

        # Tile transformation matrix for the fused interleaved RoPE (`rotary_embedding_llama`). A shared
        # constant, replicated across the mesh; the q/k weights were permuted SPLIT->INTERLEAVED at load
        # (`_rope_permute_qk`) so this neighbour-rotation matches the reference's `rotate_half`.
        self._rope_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

        # Match the decoder attention's SDPA precision (`model_qwen3vl.py::Qwen3VlAttention`):
        # HiFi4 with fp32 accumulation. The default is lower precision, and while a single block still
        # scores ~99.99% either way, 27 of them compound -- the tower is deep enough that the
        # difference is visible in the merged output where one block hides it.
        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _ring_program_config(self, local_seq_len: int) -> ttnn.SDPAProgramConfig:
        """Flash tiling for the ring, sized from the LOCAL shard.

        Built per call rather than in `__init__` like the replicated path's config, because the shard
        length is a property of the request (patch count / sp_factor) and a chunk larger than the shard
        over-reads it. Mirrors the decoder's `_sdpa_program_config` clamp.
        """
        chunk = min(-(-local_seq_len // _TILE) * _TILE, 128)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self._sdpa_worker_grid,
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,  # False is the more accurate softmax
        )

    def _windowed_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Flash tiling for windowed (block-diagonal) attention.

        Windows shorter than the chunk are fine, and so are windows that are not tile-aligned: the
        mask generator searches `cu_window_seqlens` per Q chunk, so one chunk may straddle several
        windows (`tests/ttnn/unit_tests/operations/sdpa/test_windowed_sdpa.py` covers 33-row windows
        at chunk 64). Uses the full compute grid -- unlike the ring, nothing here reserves cores for
        CCL workers.

        K chunk 512 (vs the q chunk's 128) is a FIDELITY choice, not just a perf one: the flash
        streaming softmax pays one running-max/output rescale per K chunk, and the per-block one-step
        error scales with the rescale count -- measured 3.10 / 2.34 / 1.15 / 0.67 % at k_chunk
        64/128/256/512 on the tower's worst block (real weights, 21K-key windows), the dominant term
        of the tower's end-to-end error. Purely k-sided: q256/k256 measured identical to q128/k256,
        and q256/k512 exceeds L1, so q stays at 128.

        (64, 960) is the accuracy-optimal point that fits the PIPELINE's L1 tenancy: the scores CB
        scales with q*k (~65K elements is the bare frontier) and the K/V chunk CBs with k alone
        (k1536 fails even at q64), so large k requires q64. (64, 1024) fits standalone but clashes
        with the pipeline's resident L1 buffers by ~13 KB (CCL persistent state etc. -- caught by
        test_pipeline_encode_vision_smoke), so k steps down one tile-multiple. Chosen deliberately
        over (128, 512)'s 6.5 % / 270 ms: ~0.6 RMSE points for ~10 % tower time (q64 doubles Q-loop
        iterations). q chunk size has no accuracy effect (measured); only the K-chunk count does. The fitted sweep puts the
        non-chunking error floor at ~0.27 %/step -- the bf16 CPU reference's own level -- so a
        kernel carrying the rescale path in higher precision would reach reference parity at any
        chunk size and reclaim both the L1 and the q64 latency.
        """
        tiles = -(-seq_len // _TILE) * _TILE
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=min(tiles, 64),
            k_chunk_size=min(tiles, 960),
            exp_approx_mode=False,  # False is the more accurate softmax
        )

    def _rope_interleave_qk(self, parts: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Permute only the q and k thirds of a `[q|k|v]` pack SPLIT->INTERLEAVED; leave v as is."""
        q, k, v = parts
        pk = dict(num_heads=self.num_heads, head_dim=self.head_dim)
        return _rope_permute_qk(q, **pk), _rope_permute_qk(k, **pk), v

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        kw = dict(num_heads=self.num_heads, head_dim=self.head_dim, padded=self.padded_head_dim)
        tp = self._p.tp_factor
        if (w := state.get("qkv.weight")) is not None:
            # the reference packs [q|k|v] on the output axis; permute q/k to interleaved for the fused
            # RoPE, then pad each third's heads independently. v carries no position and is untouched.
            parts = self._rope_interleave_qk(w.chunk(3, dim=0))
            padded = torch.cat([_pad_head_dim(part, axis=0, **kw) for part in parts])
            state["qkv.weight"] = _interleave_for_col_parallel(padded, parts=3, tp_factor=tp)
        if (b := state.get("qkv.bias")) is not None:
            parts = self._rope_interleave_qk(b.chunk(3, dim=0))
            padded = torch.cat([_pad_head_dim(part, axis=0, **kw) for part in parts])
            state["qkv.bias"] = _interleave_for_col_parallel(padded, parts=3, tp_factor=tp)
        if (w := state.get("proj.weight")) is not None:
            # `proj` is row-parallel: its INPUT axis is the one that fractures, and it fractures
            # contiguously into the same per-device head groups `qkv` produces. So padding the head
            # dimension is all that is needed -- no interleave, and no transpose (RowParallelLinear
            # does its own).
            state["proj.weight"] = _pad_head_dim(w, axis=1, **kw)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        cu_seqlens: Sequence[int] | None = None,
    ) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        qkv = self.qkv.forward(hidden_states)

        # `nlp_create_qkv_heads` does the head split in a single op, replacing the
        # 3x(slice + reshape + permute) this used to take: it emits q/k/v as
        # `(1, num_local_heads, seq_len, padded_head_dim)` directly. Crucially it builds its output from
        # the tensor's *logical* shape, so a patch count that is not a multiple of 32 (784 for a 28x28
        # grid) survives -- unlike `ttnn.split`, which reported the tile-padded count and forced the old
        # slice-based path. The local qkv is `[q_d | k_d | v_d]` of width `3 * local_inner` (see
        # `_interleave_for_col_parallel`), which is the packed layout the op expects; MHA means the q and
        # kv head counts are equal, and head_dim is inferred from the width as the padded 96.
        qkv = ttnn.reshape(qkv, (1, 1, seq_len, 3 * self.local_inner))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_local_heads,
            num_kv_heads=self.num_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Fused interleaved RoPE: one `rotary_embedding_llama` kernel per tensor replaces the
        # rotate_half slice/neg/concat/mul chain. cos/sin arrive as `(seq, padded_head_dim)`; the op
        # wants `[1, 1, seq, padded_head_dim]` (head-broadcast), and adding unit leading dims is a
        # metadata view, not a copy. The padded tail of cos/sin is identity, so channels 72..95 (the
        # `_pad_head_dim` zeros) pass through untouched.
        cos, sin = pos_embeds
        cos = ttnn.reshape(cos, (1, 1, cos.shape[-2], cos.shape[-1]))
        sin = ttnn.reshape(sin, (1, 1, sin.shape[-2], sin.shape[-1]))
        q = ttnn.experimental.rotary_embedding_llama(
            q,
            cos,
            sin,
            self._rope_trans_mat,
            is_decode_mode=False,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k,
            cos,
            sin,
            self._rope_trans_mat,
            is_decode_mode=False,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        single_block = cu_seqlens is None or len(cu_seqlens) <= 2
        if self._p.sp:
            # `seq_len` here is the LOCAL shard; the logical sequence spans the whole SP axis.
            # Ring SDPA rejects a non-tile-aligned shard deep in the device op ("Per-device Q seq
            # length must be divisible by TILE_HEIGHT"), and the windowed path offsets whole tiles;
            # check it here so the constraint is legible. This is stricter than, and therefore
            # subsumes, the merger's merge-group alignment.
            if seq_len % _TILE != 0:
                msg = (
                    f"sequence-parallel shard has {seq_len} rows, which is not a multiple of {_TILE}; "
                    f"the patch count must be divisible by sp_factor * {_TILE}"
                )
                raise ValueError(msg)
            if single_block:
                attn = self._ring_attention(q, k, v, seq_len)
            else:
                attn = self._windowed_sp_attention(q, k, v, seq_len, cu_seqlens)
        elif single_block:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=self.scale,
                # Same flash tiling as the windowed path (nothing about it is windowed-specific).
                # Without it the op defaults to chunk 32, and the streaming softmax's per-chunk
                # rescale error scales with the chunk count -- at fl2va sizes (16K-65K keys, one
                # block) that is the worst configuration of all; see _windowed_program_config.
                program_config=self._windowed_program_config(seq_len),
                compute_kernel_config=self._sdpa_compute_kernel_config,
            )
        else:
            if cu_seqlens[0] != 0 or cu_seqlens[-1] != seq_len:
                msg = f"cu_seqlens must span [0, {seq_len}], got {cu_seqlens[0]}..{cu_seqlens[-1]}"
                raise ValueError(msg)
            # One windowed call rather than one SDPA per block: the device synthesizes the
            # block-diagonal mask from `cu_window_seqlens`, so the boundaries never become host-side
            # slices. `uint32` / ROW_MAJOR / 1-D and 2..1024 entries are what the op validates; 18
            # blocks (`max_load`) is the most this model can present, well inside that.
            cu_window = ttnn.from_torch(
                torch.tensor(cu_seqlens, dtype=torch.int32),
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=self.scale,
                program_config=self._windowed_program_config(seq_len),
                compute_kernel_config=self._sdpa_compute_kernel_config,
                cu_window_seqlens=cu_window,
            )
        # `nlp_concat_heads` fuses the head-axis transpose + merge that this used to do as a separate
        # `permute((0,2,1,3)) + reshape`: it takes `(1, num_local_heads, seq_len, padded_head_dim)` and
        # emits `(1, 1, seq_len, num_local_heads * padded_head_dim)` = `(1, 1, seq_len, local_inner)`.
        # The trailing reshape just drops the leading unit dims to the `(seq_len, local_inner)` that
        # row-parallel `proj` expects; the padded head_dim (96) is concatenated as-is, matching the
        # packed width `_interleave_for_col_parallel` produced on the way in.
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, (seq_len, self.local_inner))
        # Row-parallel `proj` reduce-scatters then gathers back to full width. Split that all-reduce on
        # the tile-aligned sequence dim (dodging the hidden-dim `144` pad dance); the helper pads the
        # shard internally when needed, so every grid takes this path.
        return _row_parallel_seq_forward(self.proj, attn, self._p)

    def _ring_attention(self, q, k, v, local_seq_len: int) -> ttnn.Tensor:
        """Full attention over a sequence sharded on the SP axis.

        `ring_joint_scaled_dot_product_attention` is the same primitive the Wan denoiser drives
        (`wan2_2/attention_wan.py`); it gathers k/v around the ring while streaming the softmax, so no
        device ever materializes the whole `s x s` score matrix. The joint inputs are the API's
        cross-attention slots and are unused here -- zero-width tensors keep it pure self-attention.
        """
        sp_axis, ccl = self._p.sp_axis, self._p.ccl_manager
        empty_joint = bf16_tensor(
            torch.zeros((1, self.num_local_heads, 0, self.padded_head_dim)), device=self.mesh_device
        )
        attn, _joint, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            empty_joint,
            empty_joint,
            empty_joint,
            persistent_output_buffer_k=ccl.get_ag_ping_pong_buffer(k.shape, 2, sp_axis, dtype=k.dtype),
            persistent_output_buffer_v=ccl.get_ag_ping_pong_buffer(v.shape, 2, sp_axis, dtype=v.dtype),
            joint_strategy="rear",
            logical_n=local_seq_len * self._p.sp_factor,
            program_config=self._ring_program_config(local_seq_len),
            compute_kernel_config=self._sdpa_compute_kernel_config,
            dim=2,
            scale=self.scale,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(sp_axis),
            num_links=ccl.num_links,
            cluster_axis=sp_axis,
            mesh_device=self.mesh_device,
            topology=ccl.topology,
            subdevice_id=ccl.ccl_sub_device_id,
            ccl_core_grid_offset=(self._sdpa_worker_grid[0], 0),
            use_column_major_ccl=True,
        )
        return attn

    def _windowed_sp_attention(self, q, k, v, local_seq_len: int, cu_seqlens: Sequence[int]) -> ttnn.Tensor:
        """Windowed (block-diagonal) attention over a sequence sharded on the SP axis.

        Q stays the local shard -- the linear ops keep their full SP benefit and the output needs no
        combine -- while K/V are all-gathered to the full sequence once. `cu_window_seqlens` is global,
        so the op needs each shard's global origin to resolve its rows' windows: a 1-element-per-device
        tensor sharded on the SP axis carries it. A scalar offset could not -- it is baked into the
        program, and every device runs the SAME program; the tensor keeps the divergence in data.

        A shard straddling a window boundary costs nothing here: each row's window is decided by the
        on-device mask against the full gathered K, never by where the shard was cut.
        """
        sp_axis, ccl = self._p.sp_axis, self._p.ccl_manager
        global_seq_len = local_seq_len * self._p.sp_factor
        if cu_seqlens[0] != 0 or cu_seqlens[-1] != global_seq_len:
            msg = f"cu_seqlens must span [0, {global_seq_len}], got {cu_seqlens[0]}..{cu_seqlens[-1]}"
            raise ValueError(msg)
        # Consecutive same-shape gathers land in the two halves of the ping-pong buffer pair, so the
        # v gather does not clobber k.
        k = ccl.all_gather_persistent_buffer(k, dim=-2, mesh_axis=sp_axis, use_hyperparams=True)
        v = ccl.all_gather_persistent_buffer(v, dim=-2, mesh_axis=sp_axis, use_hyperparams=True)
        cu_window = ttnn.from_torch(
            torch.tensor(cu_seqlens, dtype=torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        q_offsets = typed_tensor(
            torch.arange(self._p.sp_factor, dtype=torch.int32) * local_seq_len,
            ttnn.uint32,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=0,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=self._windowed_program_config(local_seq_len),
            compute_kernel_config=self._sdpa_compute_kernel_config,
            cu_window_seqlens=cu_window,
            windowed_q_token_offset_tensor=q_offsets,
        )


class Qwen3VlVisionBlock(Module):
    """Pre-norm attention and MLP with `LayerNorm` (eps 1e-6), not the decoder's RMSNorm."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        hidden_act: str,
        norm_eps: float,
        mesh_device,
        parallel: VisionParallel | None = None,
        linear_compute_kernel_config=None,
    ) -> None:
        super().__init__()
        parallel = parallel or VisionParallel()
        # Both norms stay replicated over the full hidden width: `attn` and `mlp` each all-gather
        # their column-fractured result, so activations are full-width at every block boundary and
        # LayerNorm needs no cross-device statistics. Under SP they are row-wise and so unaffected.
        self.norm1 = LayerNorm(hidden_size, norm_eps=norm_eps, mesh_device=mesh_device)
        self.attn = Qwen3VlVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mesh_device=mesh_device,
            parallel=parallel,
            linear_compute_kernel_config=linear_compute_kernel_config,
        )
        self.norm2 = LayerNorm(hidden_size, norm_eps=norm_eps, mesh_device=mesh_device)
        self.mlp = Qwen3VlVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            mesh_device=mesh_device,
            parallel=parallel,
            linear_compute_kernel_config=linear_compute_kernel_config,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        cu_seqlens: Sequence[int] | None = None,
    ) -> ttnn.Tensor:
        hidden_states = hidden_states + self.attn.forward(
            self.norm1.forward(hidden_states), pos_embeds=pos_embeds, cu_seqlens=cu_seqlens
        )
        return hidden_states + self.mlp.forward(self.norm2.forward(hidden_states))


class Qwen3VlVisionPatchMerger(Module):
    """Merges a `spatial_merge_size ** 2` block of patches into one `out_hidden_size` token.

    `use_postshuffle_norm` is the only structural difference between the tower's output merger and its
    deepstack mergers, and it is not merely where the reshape sits: pre-shuffle normalizes each patch
    independently over `hidden_size`, while post-shuffle normalizes the concatenated group of four over
    `hidden_size * merge ** 2`. Different statistics, so the two are not interchangeable.

    The reference uses `nn.GELU()` here -- the exact erf form -- where the block MLP uses
    `gelu_pytorch_tanh`. Both map to `ttnn.gelu`'s default (accurate) mode: the two torch forms differ
    by at most 4.7e-4, which is ~23x below the 1.1e-2 bf16 quantization noise floor, so the distinction
    is not observable at this precision. Do not switch to `fast_and_approximate_mode=True`, which is
    2.4e-2 and would be.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        norm_eps: float,
        use_postshuffle_norm: bool,
        mesh_device,
        parallel: VisionParallel | None = None,
        linear_compute_kernel_config=None,
    ) -> None:
        super().__init__()
        self._p = parallel or VisionParallel()
        self.merged_size = hidden_size * spatial_merge_size**2
        self.spatial_merge_size = spatial_merge_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = LayerNorm(
            self.merged_size if use_postshuffle_norm else hidden_size, norm_eps=norm_eps, mesh_device=mesh_device
        )
        if self._p.tp:
            if self.merged_size % self._p.tp_factor != 0:
                msg = f"merged_size {self.merged_size} is not divisible by TP factor {self._p.tp_factor}"
                raise ValueError(msg)
            kw = dict(
                mesh_device=mesh_device,
                mesh_axis=self._p.tp_axis,
                ccl_manager=self._p.ccl_manager,
                compute_kernel_config=linear_compute_kernel_config,
            )
            self.linear_fc1 = ColParallelLinear(self.merged_size, self.merged_size, bias=True, **kw)
            self.linear_fc2 = RowParallelLinear(self.merged_size, out_hidden_size, bias=True, **kw)
        else:
            kw = dict(mesh_device=mesh_device, compute_kernel_config=linear_compute_kernel_config)
            self.linear_fc1 = Linear(self.merged_size, self.merged_size, bias=True, **kw)
            self.linear_fc2 = Linear(self.merged_size, out_hidden_size, bias=True, **kw)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """`(total_patches, hidden_size)` -> `(total_patches // merge ** 2, out_hidden_size)`."""
        # The reshape folds `merge ** 2` CONSECUTIVE rows into one token, so under SP a device's row
        # count must be a whole number of merge groups. Otherwise a group straddles two devices and the
        # reshape silently mixes patches from different tokens -- wrong output, no error.
        group = self.spatial_merge_size**2
        if self._p.sp and x.shape[-2] % group != 0:
            msg = (
                f"sequence-parallel shard has {x.shape[-2]} rows, not a multiple of the {group}-row "
                f"merge group; the patch count must be divisible by sp_factor * {group}"
            )
            raise ValueError(msg)

        if self.use_postshuffle_norm:
            x = self.norm.forward(ttnn.reshape(x, (-1, self.merged_size)))
        else:
            x = ttnn.reshape(self.norm.forward(x), (-1, self.merged_size))
        return _row_parallel_forward(self.linear_fc2, ttnn.gelu(self.linear_fc1.forward(x)), self._p)


class Qwen3VlVisionModel(Module):
    """The Qwen3-VL vision tower.

    `forward` returns `(merged_tokens, deepstack_features)`, corresponding to the reference's
    `pooler_output` and `deepstack_features`. The decoder scatters the former into the text sequence at
    `<|image_pad|>` positions and adds the latter into its first few layers.

    The position table is kept on the host rather than the device: interpolating it is a pure function
    of `grid_thw` and the table, so `prepare_pos_embeds` settles it with host arithmetic and the result
    is uploaded, as with the rotary tensors.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 1152,
        num_heads: int = 16,
        depth: int = 27,
        intermediate_size: int = 4304,
        in_channels: int = 3,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 5120,
        hidden_act: str = "gelu_pytorch_tanh",
        norm_eps: float = 1e-6,
        deepstack_visual_indexes: Sequence[int] = (8, 16, 24),
        mesh_device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
        high_fidelity_linears: bool = False,
    ) -> None:
        super().__init__()
        self._p = resolve_vision_parallel(mesh_device, parallel_config, ccl_manager)
        # HiFi4 for every tower linear, mirroring the decoder's `high_fidelity_linears`: the tt_dit-wide
        # default is HiFi2, and the tower's ~10% RMS output error at real weights (vs the fp32
        # reference) is what the fused conditioner amplifies. Off by default; MiniMax-H3 opts in.
        linear_compute_kernel_config = None
        if high_fidelity_linears:
            linear_compute_kernel_config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.spatial_merge_size = spatial_merge_size
        self.num_grid_per_side = int(num_position_embeddings**0.5)
        self.deepstack_visual_indexes = tuple(deepstack_visual_indexes)
        self._pos_embed_weight: torch.Tensor | None = None

        self.patch_embed = Qwen3VlVisionPatchEmbed(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            mesh_device=mesh_device,
        )
        self.blocks = ModuleList(
            Qwen3VlVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                norm_eps=norm_eps,
                mesh_device=mesh_device,
                parallel=self._p,
                linear_compute_kernel_config=linear_compute_kernel_config,
            )
            for _ in range(depth)
        )
        merger_kwargs = dict(
            hidden_size=hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            norm_eps=norm_eps,
            mesh_device=mesh_device,
            parallel=self._p,
            linear_compute_kernel_config=linear_compute_kernel_config,
        )
        self.merger = Qwen3VlVisionPatchMerger(use_postshuffle_norm=False, **merger_kwargs)
        self.deepstack_merger_list = ModuleList(
            Qwen3VlVisionPatchMerger(use_postshuffle_norm=True, **merger_kwargs) for _ in self.deepstack_visual_indexes
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # The position table never reaches the device: it is only read by `prepare_pos_embeds`, which
        # runs on the host. Popping it keeps a strict load from reporting it as unexpected.
        weight = state.pop("pos_embed.weight", None)
        if weight is not None:
            self._pos_embed_weight = weight.detach().float()

    def prepare_pos_embeds(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """The `(total_patches, hidden_size)` interpolated position embedding, on the host."""
        if self._pos_embed_weight is None:
            msg = "pos_embed weight is unavailable; call load_torch_state_dict first"
            raise ValueError(msg)
        return vision_pos_embeds(
            self._pos_embed_weight,
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.spatial_merge_size,
        )

    def prepare_rope(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The `(cos, sin)` the tower's blocks rotate with, on the host."""
        return vision_rope_tensors(grid_thw, head_dim=self.head_dim, spatial_merge_size=self.spatial_merge_size)

    def forward(
        self,
        patches: ttnn.Tensor,
        *,
        pos_embeds: ttnn.Tensor,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        cu_seqlens: Sequence[int] | None = None,
        logical_patches: int | None = None,
    ) -> tuple[ttnn.Tensor, list[ttnn.Tensor]]:
        """`cu_seqlens` confines attention to one image or video frame; see [`vision_cu_seqlens`].

        Omitting it treats the whole input as one block, which is correct for a single image and wrong
        for several -- pass it whenever `grid_thw` has more than one row or a `t` above 1.

        `logical_patches` is the REAL patch count when the input was padded for SP alignment by
        [`pad_patches_for_sp`]: the pad's merged garbage tokens are trimmed off the merged output and
        every deepstack feature after the SP gather (the pad lives on the trailing shard, so only the
        gathered, full-sequence tokens can be tail-sliced). None means nothing was padded.
        """
        hidden_states = ttnn.add(self.patch_embed.forward(patches), pos_embeds)

        # Instrumentation: report the parallel placement and the attention path the blocks will take,
        # so a caller can confirm the sharded windowed/ring path is actually engaged (vs replicated).
        _single_block = cu_seqlens is None or len(cu_seqlens) <= 2
        _path = (
            ("ring" if _single_block else "windowed_sp") if self._p.sp else ("full" if _single_block else "windowed")
        )
        logger.info(
            f"vision tower: path={_path} tp={self._p.tp_factor} sp={self._p.sp_factor} "
            f"local_rows={hidden_states.shape[-2]} blocks={(len(cu_seqlens) - 1) if cu_seqlens else 1}"
        )

        real_tokens = None if logical_patches is None else logical_patches // self.spatial_merge_size**2

        deepstack_features: list[ttnn.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block.forward(hidden_states, pos_embeds=rope, cu_seqlens=cu_seqlens)
            if layer_idx in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_idx)]
                deepstack_features.append(_trim_tokens(self._gather_tokens(merger.forward(hidden_states)), real_tokens))

        return _trim_tokens(self._gather_tokens(self.merger.forward(hidden_states)), real_tokens), deepstack_features

    def _gather_tokens(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Reassemble merged tokens across the SP axis.

        The decoder consumes these through `_scatter_rows`, which walks `vision_runs` over the whole
        token sequence, so the tower must hand back every token on every device -- SP ends here. Safe as
        a plain concatenation because SP shards rows contiguously (device `d` holds rows
        `[d * S/sp, (d+1) * S/sp)`), so device order equals token order for any number of blocks.

        The `ttnn.clone` is load-bearing. Every CCL gather here writes into a persistent buffer that
        `CCLManager` caches by `(shape, dim, mesh_axis)`, and all four mergers emit the SAME
        `(tokens, out_hidden_size)` shape -- so they share one buffer. Deepstack features are retained
        across the remaining blocks while later mergers gather into that buffer, so without the clone
        they are silently overwritten (a feature reads PCC 0.009% while the output tokens read
        99.99%). Cloning moves the result out of the buffer, as `Qwen3VlTextEncoder.forward` does for
        its embedding gather.
        """
        if not (self._p.sp or self._p.tp):
            return x
        if self._p.sp:
            x, added = _with_batch_axis(x)
            x = self._p.ccl_manager.all_gather_persistent_buffer(
                x, dim=-2, mesh_axis=self._p.sp_axis, use_hyperparams=True
            )
            x = _drop_batch_axis(x, added)
        return ttnn.clone(x)
