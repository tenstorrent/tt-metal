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

from collections.abc import Sequence
from pathlib import Path

import torch

import ttnn
from models.tt_dit.encoders.vision import (
    VisionAttention,
    VisionParallel,
    VisionPatchEmbed,
    drop_batch_axis,
    resolve_vision_parallel,
    row_parallel_forward,
    row_parallel_seq_forward,
    vision_rope_tensors,
    with_batch_axis,
)
from models.tt_dit.layers.linear import ColParallelLinear, Linear, RowParallelLinear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager

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


def pad_patches_for_sp(
    patches: torch.Tensor,
    pos_embeds: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: Sequence[int],
    *,
    sp_factor: int,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, ...], int]:
    """Pad a patch batch so its SP shards are tile-aligned, isolating the pad in phantom windows.

    Windowed attention costs each window's length squared, so the pad is carved into windows of
    `sp_factor * 32` rows, the last taking the remainder. With `pad_to`, pad to at least that many
    patches and always append a phantom window (empty when nothing is padded), so every batch at one
    padded size takes the same windowed-attention programs.

    The pad tokens stay in the tower's output; the text encoder's `merge_vision` gathers past them.
    """
    total = patches.shape[0]
    mult = sp_factor * _TILE
    padded = -(-total // mult) * mult
    if pad_to is not None:
        if pad_to % mult != 0:
            raise ValueError(f"pad_to must be a multiple of {mult}, got {pad_to}")
        padded = max(padded, pad_to)
    elif padded == total:
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
        (*tuple(cu_seqlens), *range(total + mult, padded, mult), padded),
        total,
    )


def _gather_hidden(x: ttnn.Tensor, p: VisionParallel) -> ttnn.Tensor:
    """All-gather a column-fractured activation back to full width, when TP is on."""
    if not p.tp:
        return x
    x, added = with_batch_axis(x)
    x = p.ccl_manager.all_gather(x, dim=-1, mesh_axis=p.tp_axis, use_hyperparams=True)
    return drop_batch_axis(x, added=added)


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
        return row_parallel_seq_forward(self.linear_fc2, x, self._p)


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
        kv_gather_capacity: int | None = None,
    ) -> None:
        super().__init__()
        parallel = parallel or VisionParallel()
        # Both norms stay replicated over the full hidden width: `attn` and `mlp` each all-gather
        # their column-fractured result, so activations are full-width at every block boundary and
        # LayerNorm needs no cross-device statistics. Under SP they are row-wise and so unaffected.
        self.norm1 = LayerNorm(hidden_size, norm_eps=norm_eps, mesh_device=mesh_device)
        self.attn = VisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mesh_device=mesh_device,
            parallel=parallel,
            linear_compute_kernel_config=linear_compute_kernel_config,
            kv_gather_capacity=kv_gather_capacity,
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
        return row_parallel_forward(self.linear_fc2, ttnn.gelu(self.linear_fc1.forward(x)), self._p)


class Qwen3VlVisionModel(Module):
    """The Qwen3-VL vision tower.

    `forward` returns `(merged_tokens, deepstack_features)`, corresponding to the reference's
    `pooler_output` and `deepstack_features`. The decoder scatters the former into the text sequence at
    `<|image_pad|>` positions and adds the latter into its first few layers.

    The position table is kept on the host rather than the device: interpolating it is a pure function
    of `grid_thw` and the table, so `prepare_pos_embeds` settles it with host arithmetic and the result
    is uploaded, as with the rotary tensors. Being no `Parameter`, it is saved to and loaded from the
    weight cache next to them by `save` and `load`.
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
        kv_gather_capacity: int | None = None,
        use_persistent_ccl_buffers: bool = True,
    ) -> None:
        super().__init__()
        self._p = resolve_vision_parallel(mesh_device, parallel_config, ccl_manager)._replace(
            use_persistent_ccl_buffers=use_persistent_ccl_buffers
        )
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

        self.patch_embed = VisionPatchEmbed(
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
                kv_gather_capacity=kv_gather_capacity,
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

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        if self._pos_embed_weight is None:
            msg = "pos_embed weight is unavailable; call load_torch_state_dict first"
            raise ValueError(msg)
        super().save(directory, prefix=prefix)
        torch.save(self._pos_embed_weight, Path(directory) / f"{prefix}pos_embed.weight.pt")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().load(directory, prefix=prefix)
        self._pos_embed_weight = torch.load(Path(directory) / f"{prefix}pos_embed.weight.pt")

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

        Input padded by [`pad_patches_for_sp`] comes back with its pad tokens after the real ones; the
        text encoder's `merge_vision` never reads them.
        """
        hidden_states = ttnn.add(self.patch_embed.forward(patches), pos_embeds)

        _single_block = cu_seqlens is None or len(cu_seqlens) <= 2
        _path = (
            ("ring" if _single_block else "windowed_sp") if self._p.sp else ("full" if _single_block else "windowed")
        )
        # logger.debug(
        #     f"vision tower: path={_path} tp={self._p.tp_factor} sp={self._p.sp_factor} "
        #     f"local_rows={hidden_states.shape[-2]} blocks={(len(cu_seqlens) - 1) if cu_seqlens else 1}"
        # )

        deepstack_features: list[ttnn.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block.forward(hidden_states, pos_embeds=rope, cu_seqlens=cu_seqlens)
            if layer_idx in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_idx)]
                deepstack_features.append(self._gather_tokens(merger.forward(hidden_states)))

        return self._gather_tokens(self.merger.forward(hidden_states)), deepstack_features

    def _gather_tokens(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Reassemble merged tokens across the SP axis.

        The decoder consumes these through `merge_vision`, which walks `vision_runs` over the whole
        token sequence, so the tower must hand back every token on every device -- SP ends here. Safe as
        a plain concatenation because SP shards rows contiguously, so device order equals token order.

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
            x, added = with_batch_axis(x)
            x = self._p.ccl_manager.all_gather(x, dim=-2, mesh_axis=self._p.sp_axis, use_hyperparams=True)
            x = drop_batch_axis(x, added=added)
        return ttnn.clone(x)
