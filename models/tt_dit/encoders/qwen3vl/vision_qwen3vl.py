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

import ttnn

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import LayerNorm
from ...utils.tensor import bf16_tensor

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


def _gather_hidden(x: ttnn.Tensor, p: VisionParallel) -> ttnn.Tensor:
    """All-gather a column-fractured activation back to full width, when TP is on."""
    if not p.tp:
        return x
    x, added = _with_batch_axis(x)
    x = p.ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=p.tp_axis, use_hyperparams=True)
    return _drop_batch_axis(x, added)


def _row_parallel_forward(linear, x: ttnn.Tensor, p: VisionParallel) -> ttnn.Tensor:
    """Run a row-parallel linear on a 2-D activation and gather its fractured result to full width."""
    if not p.tp:
        return linear.forward(x)
    x, added = _with_batch_axis(x)
    out = p.ccl_manager.all_gather_persistent_buffer(
        linear.forward(x), dim=-1, mesh_axis=p.tp_axis, use_hyperparams=True
    )
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
    ) -> None:
        super().__init__()
        self._p = parallel or VisionParallel()
        self._act = hidden_act

        if self._p.tp:
            if intermediate_size % self._p.tp_factor != 0:
                msg = f"intermediate_size {intermediate_size} is not divisible by TP factor {self._p.tp_factor}"
                raise ValueError(msg)
            kw = dict(mesh_device=mesh_device, mesh_axis=self._p.tp_axis, ccl_manager=self._p.ccl_manager)
            self.linear_fc1 = ColParallelLinear(hidden_size, intermediate_size, bias=True, **kw)
            self.linear_fc2 = RowParallelLinear(intermediate_size, hidden_size, bias=True, **kw)
        else:
            self.linear_fc1 = Linear(hidden_size, intermediate_size, bias=True, mesh_device=mesh_device)
            self.linear_fc2 = Linear(intermediate_size, hidden_size, bias=True, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.linear_fc1.forward(x)
        x = ttnn.gelu(x) if self._act.startswith("gelu") else ttnn.silu(x)
        # `RowParallelLinear` reduce-scatters, so its result is fractured on columns; the residual add
        # and the next LayerNorm both need the full width back (cf. `Qwen3VlMlp` in model_qwen3vl.py).
        return _row_parallel_forward(self.linear_fc2, x, self._p)


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
        self, *, hidden_size: int, num_heads: int, mesh_device, parallel: VisionParallel | None = None
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
            kw = dict(mesh_device=mesh_device, mesh_axis=self._p.tp_axis, ccl_manager=self._p.ccl_manager)
            self.qkv = ColParallelLinear(hidden_size, 3 * self.inner, bias=True, **kw)
            self.proj = RowParallelLinear(self.inner, hidden_size, bias=True, **kw)
        else:
            self.qkv = Linear(hidden_size, 3 * self.inner, bias=True, mesh_device=mesh_device)
            self.proj = Linear(self.inner, hidden_size, bias=True, mesh_device=mesh_device)

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

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        kw = dict(num_heads=self.num_heads, head_dim=self.head_dim, padded=self.padded_head_dim)
        tp = self._p.tp_factor
        if (w := state.get("qkv.weight")) is not None:
            # the reference packs [q|k|v] on the output axis; pad each third's heads independently
            padded = torch.cat([_pad_head_dim(part, axis=0, **kw) for part in w.chunk(3, dim=0)])
            state["qkv.weight"] = _interleave_for_col_parallel(padded, parts=3, tp_factor=tp)
        if (b := state.get("qkv.bias")) is not None:
            padded = torch.cat([_pad_head_dim(part, axis=0, **kw) for part in b.chunk(3, dim=0)])
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

        # Sliced rather than `ttnn.split`: split reports a *tile-padded* row count on its outputs, so
        # a patch count that is not a multiple of 32 (784 for a 28x28 grid) would make the reshape
        # below disagree with `seq_len`. Slicing the last dimension keeps the logical row count.
        # Under TP the local slice is `[q_d | k_d | v_d]` of width `3 * local_inner` (see
        # `_interleave_for_col_parallel`), so the stride and the head count are both the local ones.
        q, k, v = (
            ttnn.permute(
                ttnn.reshape(
                    qkv[..., i * self.local_inner : (i + 1) * self.local_inner],
                    (1, seq_len, self.num_local_heads, self.padded_head_dim),
                ),
                (0, 2, 1, 3),
            )
            for i in range(3)
        )

        cos, sin = pos_embeds
        q = _apply_vision_rope(q, cos, sin, self.head_dim)
        k = _apply_vision_rope(k, cos, sin, self.head_dim)

        single_block = cu_seqlens is None or len(cu_seqlens) <= 2
        if self._p.sp:
            # `seq_len` here is the LOCAL shard; the logical block spans the whole SP axis.
            if not single_block:
                msg = (
                    f"sequence parallelism supports a single attention block, got {len(cu_seqlens) - 1}. "
                    "Ring SDPA takes no cu_seqlens and an even row split does not align with block "
                    "boundaries, so multiple images/frames need a block-interleaved SP layout (device d "
                    "holds part d of every block) plus the inverse permutation after the output gather. "
                    "Until that lands, use sp_factor=1 for ref2va and video; TP is unaffected."
                )
                raise NotImplementedError(msg)
            # Ring SDPA rejects a non-tile-aligned shard deep in the device op ("Per-device Q seq
            # length must be divisible by TILE_HEIGHT"); check it here so the constraint is legible.
            # This is stricter than, and therefore subsumes, the merger's merge-group alignment.
            if seq_len % _TILE != 0:
                msg = (
                    f"sequence-parallel shard has {seq_len} rows, which is not a multiple of {_TILE}; "
                    f"the patch count must be divisible by sp_factor * {_TILE}"
                )
                raise ValueError(msg)
            attn = self._ring_attention(q, k, v, seq_len)
        elif single_block:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=self.scale,
                compute_kernel_config=self._sdpa_compute_kernel_config,
            )
        else:
            if cu_seqlens[0] != 0 or cu_seqlens[-1] != seq_len:
                msg = f"cu_seqlens must span [0, {seq_len}], got {cu_seqlens[0]}..{cu_seqlens[-1]}"
                raise ValueError(msg)
            attn = ttnn.concat(
                [
                    ttnn.transformer.scaled_dot_product_attention(
                        q[:, :, start:end, :],
                        k[:, :, start:end, :],
                        v[:, :, start:end, :],
                        is_causal=False,
                        scale=self.scale,
                        compute_kernel_config=self._sdpa_compute_kernel_config,
                    )
                    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])
                ],
                dim=-2,
            )
        attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), (seq_len, self.local_inner))
        # Row-parallel `proj` consumes exactly this fractured width and reduce-scatters, so gather back.
        return _row_parallel_forward(self.proj, attn, self._p)

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
        self,
        *,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        hidden_act: str,
        norm_eps: float,
        mesh_device,
        parallel: VisionParallel | None = None,
    ) -> None:
        super().__init__()
        parallel = parallel or VisionParallel()
        # Both norms stay replicated over the full hidden width: `attn` and `mlp` each all-gather
        # their column-fractured result, so activations are full-width at every block boundary and
        # LayerNorm needs no cross-device statistics. Under SP they are row-wise and so unaffected.
        self.norm1 = LayerNorm(hidden_size, norm_eps=norm_eps, mesh_device=mesh_device)
        self.attn = Qwen3VlVisionAttention(
            hidden_size=hidden_size, num_heads=num_heads, mesh_device=mesh_device, parallel=parallel
        )
        self.norm2 = LayerNorm(hidden_size, norm_eps=norm_eps, mesh_device=mesh_device)
        self.mlp = Qwen3VlVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            mesh_device=mesh_device,
            parallel=parallel,
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
            kw = dict(mesh_device=mesh_device, mesh_axis=self._p.tp_axis, ccl_manager=self._p.ccl_manager)
            self.linear_fc1 = ColParallelLinear(self.merged_size, self.merged_size, bias=True, **kw)
            self.linear_fc2 = RowParallelLinear(self.merged_size, out_hidden_size, bias=True, **kw)
        else:
            self.linear_fc1 = Linear(self.merged_size, self.merged_size, bias=True, mesh_device=mesh_device)
            self.linear_fc2 = Linear(self.merged_size, out_hidden_size, bias=True, mesh_device=mesh_device)

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
    ) -> None:
        super().__init__()
        self._p = resolve_vision_parallel(mesh_device, parallel_config, ccl_manager)
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
    ) -> tuple[ttnn.Tensor, list[ttnn.Tensor]]:
        """`cu_seqlens` confines attention to one image or video frame; see [`vision_cu_seqlens`].

        Omitting it treats the whole input as one block, which is correct for a single image and wrong
        for several -- pass it whenever `grid_thw` has more than one row or a `t` above 1.
        """
        hidden_states = ttnn.add(self.patch_embed.forward(patches), pos_embeds)

        deepstack_features: list[ttnn.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block.forward(hidden_states, pos_embeds=rope, cu_seqlens=cu_seqlens)
            if layer_idx in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_idx)]
                deepstack_features.append(self._gather_tokens(merger.forward(hidden_states)))

        return self._gather_tokens(self.merger.forward(hidden_states)), deepstack_features

    def _gather_tokens(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Reassemble merged tokens across the SP axis.

        The decoder consumes these through `_scatter_rows`, which walks `vision_runs` over the whole
        token sequence, so the tower must hand back every token on every device -- SP ends here. Safe as
        a plain concatenation only because a single attention block means device order equals token
        order; the multi-block layout that `Qwen3VlVisionAttention` rejects would need a permutation.

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
