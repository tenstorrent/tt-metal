# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-VL vision tower.

The tower turns raw patches into `out_hidden_size` tokens for the text decoder, matching
`Qwen2_5_VisionTransformerPretrainedModel`'s `pooler_output`. Qwen-Image-Edit runs it on the input
image as part of prompt encoding.

It shares its attention, patch embedding and rotary embedding with the Qwen3-VL tower
(`encoders/vision.py`). The differences are:

- Norms are RMSNorm, and the block MLP is SwiGLU with biases.
- There are no learned position embeddings and no deepstack features.
- Most layers attend within windows of `window_size` pixels (8 x 8 patches) rather than within the
  whole image; the layers in `fullatt_block_indexes` attend within the image. The reference
  reorders the patches window by window up front so that every window is contiguous, and undoes the
  reordering after the merger. [`prepare_vision_inputs`] does the former on the host, and
  [`Qwen25VlVisionModel.forward`] the latter on the device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

import ttnn
from models.tt_dit.encoders.vision import (
    VisionAttention,
    VisionParallel,
    VisionPatchEmbed,
    resolve_vision_parallel,
    row_parallel_forward,
    row_parallel_seq_forward,
    vision_cu_seqlens,
    vision_rope_tensors,
)
from models.tt_dit.layers.linear import ColParallelLinear, Linear, RowParallelLinear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import RMSNorm
from models.tt_dit.parallel.config import EncoderParallelConfig
from models.tt_dit.parallel.manager import CCLManager

_TILE = 32

# SDPA's windowed mode loads `cu_window_seqlens` into a single tile.
_MAX_WINDOWS = _TILE * _TILE - 1


def vision_window_index(
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """`(window_index, cu_window_seqlens)` for the windowed attention layers.

    `window_index` lists the merge groups (of `spatial_merge_size ** 2` patches) window by window, and
    `cu_window_seqlens` the window boundaries in patches, in that order. Windows at the right and
    bottom edges are smaller when the merged grid is not a multiple of the window. Matches
    `transformers.vision_utils.get_vision_window_index`.
    """
    merger_window = window_size // spatial_merge_size // patch_size
    merge_unit = spatial_merge_size**2
    window_index: list[torch.Tensor] = []
    bounds = [0]
    offset = 0

    for t, h, w in grid_thw.int().tolist():
        grid_h, grid_w = h // spatial_merge_size, w // spatial_merge_size
        index = torch.arange(t * grid_h * grid_w).reshape(t, grid_h, grid_w)
        pad_h = merger_window - grid_h % merger_window
        pad_w = merger_window - grid_w % merger_window
        windows_h = (grid_h + pad_h) // merger_window
        windows_w = (grid_w + pad_w) // merger_window
        index = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), value=-100)
        index = index.reshape(t, windows_h, merger_window, windows_w, merger_window)
        index = index.permute(0, 1, 3, 2, 4).reshape(t, windows_h * windows_w, merger_window, merger_window)
        seqlens = (index != -100).sum([2, 3]).reshape(-1)
        index = index.reshape(-1)
        window_index.append(index[index != -100] + offset)
        for length in seqlens.tolist():
            # Padding can leave whole windows empty; the reference drops the repeated boundaries.
            if length:
                bounds.append(bounds[-1] + length * merge_unit)
        offset += t * grid_h * grid_w

    return torch.cat(window_index), tuple(bounds)


@dataclass
class VisionInputs:
    """Host tensors for one [`Qwen25VlVisionModel.forward`] call, made by [`prepare_vision_inputs`].

    `patches`, `cos` and `sin` are in window order and padded to whole tiles. The padding rows form
    a block of their own in both `cu_seqlens` and `cu_window_seqlens`, so they attend only among
    themselves. `reverse_index` lists, in the reference's order, the row of each merged token in the
    tower's output, which leaves out the padding tokens.
    """

    patches: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    cu_seqlens: tuple[int, ...]
    cu_window_seqlens: tuple[int, ...]
    reverse_index: torch.Tensor


def prepare_vision_inputs(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    head_dim: int,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> VisionInputs:
    """Reorders the processor's patches into windows and builds the rotary and block tensors.

    The patch count is padded to a multiple of `TILE * spatial_merge_size ** 2`, so that both the
    patches and the merged tokens fill whole tiles.
    """
    merge_unit = spatial_merge_size**2
    num_patches = pixel_values.shape[0]
    multiple = _TILE * merge_unit
    padded_patches = -(-num_patches // multiple) * multiple

    window_index, cu_window_seqlens = vision_window_index(
        grid_thw, spatial_merge_size=spatial_merge_size, window_size=window_size, patch_size=patch_size
    )
    cu_seqlens = vision_cu_seqlens(grid_thw)
    cos, sin = vision_rope_tensors(grid_thw, head_dim=head_dim, spatial_merge_size=spatial_merge_size)

    def to_windows(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(num_patches // merge_unit, merge_unit, -1)[window_index]
        return x.reshape(num_patches, -1)

    patches, cos, sin = to_windows(pixel_values), to_windows(cos), to_windows(sin)

    npad = padded_patches - num_patches
    if npad:
        patches = torch.nn.functional.pad(patches, (0, 0, 0, npad))
        cos = torch.nn.functional.pad(cos, (0, 0, 0, npad), value=1.0)
        sin = torch.nn.functional.pad(sin, (0, 0, 0, npad), value=0.0)
        cu_seqlens = (*cu_seqlens, padded_patches)
        cu_window_seqlens = (*cu_window_seqlens, padded_patches)

    if len(cu_window_seqlens) - 1 > _MAX_WINDOWS:
        msg = f"{len(cu_window_seqlens) - 1} attention windows exceed the {_MAX_WINDOWS} SDPA supports"
        raise ValueError(msg)

    return VisionInputs(
        patches=patches,
        cos=cos,
        sin=sin,
        cu_seqlens=cu_seqlens,
        cu_window_seqlens=cu_window_seqlens,
        reverse_index=torch.argsort(window_index),
    )


class Qwen25VlVisionMLP(Module):
    """SwiGLU with biases.

    The intermediate size (3420) is not tile-aligned, so it is zero-padded at load time to a multiple
    of `TILE * tp_factor`. The padded gate and up rows produce `silu(0) * 0 = 0`, and the padded
    columns of `down_proj` multiply those zeros.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        mesh_device: ttnn.MeshDevice,
        parallel: VisionParallel | None = None,
        linear_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> None:
        super().__init__()
        self._p = parallel or VisionParallel()
        multiple = _TILE * self._p.tp_factor
        self.intermediate_size = intermediate_size
        self.padded_intermediate_size = -(-intermediate_size // multiple) * multiple

        if self._p.tp:
            kw = dict(
                mesh_device=mesh_device,
                mesh_axis=self._p.tp_axis,
                ccl_manager=self._p.ccl_manager,
                compute_kernel_config=linear_compute_kernel_config,
            )
            self.gate_proj = ColParallelLinear(hidden_size, self.padded_intermediate_size, bias=True, **kw)
            self.up_proj = ColParallelLinear(hidden_size, self.padded_intermediate_size, bias=True, **kw)
            self.down_proj = RowParallelLinear(self.padded_intermediate_size, hidden_size, bias=True, **kw)
        else:
            kw = dict(mesh_device=mesh_device, compute_kernel_config=linear_compute_kernel_config)
            self.gate_proj = Linear(hidden_size, self.padded_intermediate_size, bias=True, **kw)
            self.up_proj = Linear(hidden_size, self.padded_intermediate_size, bias=True, **kw)
            self.down_proj = Linear(self.padded_intermediate_size, hidden_size, bias=True, **kw)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        npad = self.padded_intermediate_size - self.intermediate_size
        if not npad:
            return
        for name in ("gate_proj.weight", "up_proj.weight"):
            if (w := state.get(name)) is not None:
                state[name] = torch.nn.functional.pad(w, (0, 0, 0, npad))
        for name in ("gate_proj.bias", "up_proj.bias"):
            if (b := state.get(name)) is not None:
                state[name] = torch.nn.functional.pad(b, (0, npad))
        if (w := state.get("down_proj.weight")) is not None:
            state["down_proj.weight"] = torch.nn.functional.pad(w, (0, npad))

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.silu(self.gate_proj.forward(x)) * self.up_proj.forward(x)
        return row_parallel_seq_forward(self.down_proj, x, self._p)


class Qwen25VlVisionBlock(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        mesh_device: ttnn.MeshDevice,
        parallel: VisionParallel | None = None,
        linear_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> None:
        super().__init__()
        parallel = parallel or VisionParallel()
        # Both norms stay replicated over the full hidden width: `attn` and `mlp` each gather their
        # result, so activations are full-width at every block boundary.
        self.norm1 = RMSNorm(hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)
        self.attn = VisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mesh_device=mesh_device,
            parallel=parallel,
            linear_compute_kernel_config=linear_compute_kernel_config,
        )
        self.norm2 = RMSNorm(hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)
        self.mlp = Qwen25VlVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
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
        # The residual stream is float32 (see `Qwen25VlVisionModel`); the linears take bfloat16.
        x = ttnn.typecast(self.norm1.forward(hidden_states), ttnn.bfloat16)
        x = self.attn.forward(x, pos_embeds=pos_embeds, cu_seqlens=cu_seqlens)
        hidden_states = hidden_states + ttnn.typecast(x, ttnn.float32)
        x = ttnn.typecast(self.norm2.forward(hidden_states), ttnn.bfloat16)
        return hidden_states + ttnn.typecast(self.mlp.forward(x), ttnn.float32)


class Qwen25VlVisionPatchMerger(Module):
    """Merges a `spatial_merge_size ** 2` block of patches into one `out_hidden_size` token.

    Normalizes each patch over `hidden_size` before concatenating the group, then applies a two-layer
    GELU MLP. The reference's `ln_q`, `mlp.0` and `mlp.2` load into `norm`, `linear_fc1` and
    `linear_fc2`.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        norm_eps: float,
        mesh_device: ttnn.MeshDevice,
        parallel: VisionParallel | None = None,
        linear_compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
    ) -> None:
        super().__init__()
        self._p = parallel or VisionParallel()
        self.merged_size = hidden_size * spatial_merge_size**2
        self.norm = RMSNorm(hidden_size, norm_eps=norm_eps, bias=False, mesh_device=mesh_device)
        if self._p.tp:
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

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        renames = {"ln_q.": "norm.", "mlp.0.": "linear_fc1.", "mlp.2.": "linear_fc2."}
        for key in list(state):
            for old, new in renames.items():
                if key.startswith(old):
                    state[new + key.removeprefix(old)] = state.pop(key)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """`(patches, hidden_size)` -> `(patches // merge ** 2, out_hidden_size)`."""
        x = ttnn.reshape(ttnn.typecast(self.norm.forward(x), ttnn.bfloat16), (-1, self.merged_size))
        return row_parallel_forward(self.linear_fc2, ttnn.gelu(self.linear_fc1.forward(x)), self._p)


class Qwen25VlVisionModel(Module):
    """The Qwen2.5-VL vision tower.

    The defaults are the configuration of Qwen2.5-VL-7B, the text encoder of Qwen-Image and
    Qwen-Image-Edit. Sequence parallelism is not supported.

    From block 17 on, a few tokens of the residual stream carry activations around 10^4, where
    bfloat16 resolves steps of 64, so the residual stream is float32; the norms read it and the
    linears take bfloat16. The linears run at HiFi4. Against the float32 reference, on a photo at
    384² and 1024², the two raise the PCC of the output from 99.50 % / 98.05 % to 99.82 % / 99.06 %.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 1280,
        num_heads: int = 16,
        depth: int = 32,
        intermediate_size: int = 3420,
        in_channels: int = 3,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        window_size: int = 112,
        fullatt_block_indexes: Sequence[int] = (7, 15, 23, 31),
        out_hidden_size: int = 3584,
        norm_eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        self._p = resolve_vision_parallel(mesh_device, parallel_config, ccl_manager)
        if self._p.sp:
            msg = "the Qwen2.5-VL vision tower does not support sequence parallelism"
            raise ValueError(msg)
        linear_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.head_dim = hidden_size // num_heads
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.window_size = window_size
        self.fullatt_block_indexes = frozenset(fullatt_block_indexes)

        self.patch_embed = VisionPatchEmbed(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            bias=False,
            mesh_device=mesh_device,
        )
        self.blocks = ModuleList(
            Qwen25VlVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
                mesh_device=mesh_device,
                parallel=self._p,
                linear_compute_kernel_config=linear_compute_kernel_config,
            )
            for _ in range(depth)
        )
        self.merger = Qwen25VlVisionPatchMerger(
            hidden_size=hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            norm_eps=norm_eps,
            mesh_device=mesh_device,
            parallel=self._p,
            linear_compute_kernel_config=linear_compute_kernel_config,
        )

    @classmethod
    def from_hf_config(
        cls,
        config: Qwen2_5_VLVisionConfig,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> Qwen25VlVisionModel:
        return cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            depth=config.depth,
            intermediate_size=config.intermediate_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            window_size=config.window_size,
            fullatt_block_indexes=config.fullatt_block_indexes,
            out_hidden_size=config.out_hidden_size,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def prepare_inputs(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> VisionInputs:
        return prepare_vision_inputs(
            pixel_values,
            grid_thw,
            head_dim=self.head_dim,
            spatial_merge_size=self.spatial_merge_size,
            window_size=self.window_size,
            patch_size=self.patch_size,
        )

    def forward(
        self,
        patches: ttnn.Tensor,
        *,
        rope: tuple[ttnn.Tensor, ttnn.Tensor],
        reverse_index: ttnn.Tensor,
        cu_seqlens: Sequence[int],
        cu_window_seqlens: Sequence[int],
    ) -> ttnn.Tensor:
        """Runs the tower on the inputs [`prepare_vision_inputs`] made, uploaded to the device.

        `reverse_index` is a `(1, tokens)` uint32 row-major tensor. Returns the `(tokens,
        out_hidden_size)` merged tokens in the reference's order.
        """
        hidden_states = ttnn.typecast(self.patch_embed.forward(patches), ttnn.float32)
        for i, block in enumerate(self.blocks):
            blocks = cu_seqlens if i in self.fullatt_block_indexes else cu_window_seqlens
            hidden_states = block.forward(hidden_states, pos_embeds=rope, cu_seqlens=blocks)

        merged = self.merger.forward(hidden_states)
        merged = ttnn.embedding(reverse_index, merged, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(merged, (merged.shape[-2], merged.shape[-1]))
