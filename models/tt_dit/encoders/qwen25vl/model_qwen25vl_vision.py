# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

import ttnn

from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from .model_qwen25vl import _apply_rope

if TYPE_CHECKING:
    pass


@dataclass
class Qwen25VlVisionContext:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    fsdp_mesh_axis: int | None = None


class Qwen25VlVisionPatchEmbed(Module):
    def __init__(
        self,
        *,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
        ctx: "Qwen25VlVisionContext",
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self._in_features = in_channels * temporal_patch_size * patch_size * patch_size

        self.proj = ColParallelLinear(
            self._in_features,
            embed_dim,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.pop("proj.weight", None)
        if weight is not None:
            flat = weight.reshape(self.embed_dim, -1).contiguous()
            state["proj.weight"] = flat

    def forward(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        x = self.proj.forward(pixel_values)
        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
        return x


def build_vision_rope_tensors(
    grid_thw: Sequence[tuple[int, int, int]],
    *,
    head_dim: int,
    spatial_merge_size: int,
    theta: float = 10000.0,
    pad_to: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_ids_list: list[torch.Tensor] = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

        pos_ids_list.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids_list, dim=0)

    max_grid = max(max(h, w) for _, h, w in grid_thw)
    rope_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    seq = torch.arange(max_grid, dtype=torch.float32)
    freqs_table = torch.outer(seq, inv_freq)

    rotary = freqs_table[pos_ids].flatten(1)
    emb = torch.cat((rotary, rotary), dim=-1)
    cos = emb.cos().to(torch.float32)
    sin = emb.sin().to(torch.float32)

    if pad_to is not None and pad_to > head_dim:
        pad_cols = pad_to - head_dim
        cos = torch.cat([cos, torch.ones(cos.shape[0], pad_cols, dtype=cos.dtype)], dim=-1)
        sin = torch.cat([sin, torch.zeros(sin.shape[0], pad_cols, dtype=sin.dtype)], dim=-1)

    return cos, sin


class Qwen25VlVisionMLP(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        ctx: Qwen25VlVisionContext,
    ) -> None:
        super().__init__()

        self.gate_proj = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.up_proj = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.silu(self.gate_proj.forward(x)) * self.up_proj.forward(x)
        x = self.down_proj.forward(x)
        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
        return x


class Qwen25VlVisionAttention(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        ctx: Qwen25VlVisionContext,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            msg = f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            raise ValueError(msg)

        head_dim = hidden_size // num_heads
        tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1

        if num_heads % tp_factor != 0:
            msg = f"num_heads {num_heads} must be divisible by tp_factor {tp_factor}"
            raise ValueError(msg)

        padded_head_dim = ((head_dim + 31) // 32) * 32
        proj_in_features = num_heads * padded_head_dim

        self.qkv_proj = ColParallelLinear(
            hidden_size,
            3 * hidden_size,
            bias=True,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.proj = ColParallelLinear(
            proj_in_features,
            hidden_size,
            bias=True,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )

        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        self._head_dim = head_dim
        self._padded_head_dim = padded_head_dim
        self._num_heads = num_heads
        self._num_local_heads = num_heads // tp_factor
        self._tp_axis = ctx.tp_axis
        self._tp_factor = tp_factor
        self._ccl_manager = ctx.ccl_manager
        self._device = ctx.device

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        qkv_weight = state.pop("qkv.weight", None)
        qkv_bias = state.pop("qkv.bias", None)

        if qkv_weight is not None:
            w = qkv_weight.view(3, self._num_heads, self._head_dim, -1)
            w = w.view(3, self._tp_factor, self._num_local_heads, self._head_dim, -1)
            w = w.permute(1, 0, 2, 3, 4).contiguous()
            w = w.view(self._tp_factor * 3 * self._num_local_heads * self._head_dim, -1)
            state["qkv_proj.weight"] = w

        if qkv_bias is not None:
            b = qkv_bias.view(3, self._num_heads, self._head_dim)
            b = b.view(3, self._tp_factor, self._num_local_heads, self._head_dim)
            b = b.permute(1, 0, 2, 3).contiguous()
            state["qkv_proj.bias"] = b.view(-1)

        proj_weight = state.get("proj.weight", None)
        if proj_weight is not None and self._padded_head_dim != self._head_dim:
            out_features = proj_weight.shape[0]
            w = proj_weight.view(out_features, self._num_heads, self._head_dim)
            pad = torch.zeros(
                out_features,
                self._num_heads,
                self._padded_head_dim - self._head_dim,
                dtype=w.dtype,
            )
            w = torch.cat([w, pad], dim=-1).contiguous()
            state["proj.weight"] = w.view(out_features, self._num_heads * self._padded_head_dim)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        qkv = self.qkv_proj.forward(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(qkv, 1),
            num_heads=self._num_local_heads,
            num_kv_heads=self._num_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos, sin = pos_embeds
        if len(cos.shape) == 3:
            cos = ttnn.reshape(cos, (cos.shape[0], 1, cos.shape[1], cos.shape[2]))
            sin = ttnn.reshape(sin, (sin.shape[0], 1, sin.shape[1], sin.shape[2]))
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        x = ttnn.transformer.concatenate_heads(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = self.proj.forward(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x


class Qwen25VlVisionBlock(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        rms_norm_eps: float,
        ctx: Qwen25VlVisionContext,
    ) -> None:
        super().__init__()

        self.norm1 = RMSNorm(hidden_size, norm_eps=rms_norm_eps, bias=False, mesh_device=ctx.device)
        self.norm2 = RMSNorm(hidden_size, norm_eps=rms_norm_eps, bias=False, mesh_device=ctx.device)
        self.attn = Qwen25VlVisionAttention(hidden_size=hidden_size, num_heads=num_heads, ctx=ctx)
        self.mlp = Qwen25VlVisionMLP(hidden_size=hidden_size, intermediate_size=intermediate_size, ctx=ctx)

    def forward(self, x: ttnn.Tensor, *, pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        residual = x
        x = self.norm1.forward(x)
        x = self.attn.forward(x, pos_embeds=pos_embeds)
        x = x + residual

        residual = x
        x = self.norm2.forward(x)
        x = self.mlp.forward(x)
        x = x + residual

        return x


class Qwen25VlPatchMerger(Module):
    def __init__(
        self,
        *,
        context_dim: int,
        out_dim: int,
        spatial_merge_size: int,
        rms_norm_eps: float,
        ctx: Qwen25VlVisionContext,
    ) -> None:
        super().__init__()

        self.spatial_merge_size = spatial_merge_size
        self.merge_unit = spatial_merge_size * spatial_merge_size
        self.merged_dim = context_dim * self.merge_unit

        self.ln_q = RMSNorm(context_dim, norm_eps=rms_norm_eps, bias=False, mesh_device=ctx.device)
        self.fc1 = ColParallelLinear(
            self.merged_dim,
            self.merged_dim,
            bias=True,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.fc2 = ColParallelLinear(
            self.merged_dim,
            out_dim,
            bias=True,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "mlp.0.weight" in state:
            state["fc1.weight"] = state.pop("mlp.0.weight")
        if "mlp.0.bias" in state:
            state["fc1.bias"] = state.pop("mlp.0.bias")
        if "mlp.2.weight" in state:
            state["fc2.weight"] = state.pop("mlp.2.weight")
        if "mlp.2.bias" in state:
            state["fc2.bias"] = state.pop("mlp.2.bias")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.ln_q.forward(x)

        seq_len = x.shape[-2]
        merged_seq = seq_len // self.merge_unit
        context_dim = x.shape[-1]
        x = ttnn.reshape(x, (1, merged_seq, context_dim * self.merge_unit))

        x = ttnn.gelu(self.fc1.forward(x))
        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
        x = self.fc2.forward(x)
        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L306
class Qwen25VlVisionEncoder(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        depth: int,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        fsdp_mesh_axis = None
        if is_fsdp and parallel_config is not None:
            tp_axis = parallel_config.tensor_parallel.mesh_axis
            other_axis = 1 - tp_axis
            if device.shape[other_axis] > 1:
                fsdp_mesh_axis = other_axis

        ctx = Qwen25VlVisionContext(
            device=device,
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        self.patch_embed = Qwen25VlVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            ctx=ctx,
        )

        self.blocks = ModuleList(
            Qwen25VlVisionBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                rms_norm_eps=rms_norm_eps,
                ctx=ctx,
            )
            for _ in range(depth)
        )

        self.merger = Qwen25VlPatchMerger(
            context_dim=hidden_size,
            out_dim=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            rms_norm_eps=rms_norm_eps,
            ctx=ctx,
        )

        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._head_dim = hidden_size // num_heads
        self._spatial_merge_size = spatial_merge_size
        self._rope_theta = rope_theta
        self._device = device
        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def build_pos_embeds(self, grid_thw: Sequence[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        padded_head_dim = ((self._head_dim + 31) // 32) * 32
        return build_vision_rope_tensors(
            grid_thw,
            head_dim=self._head_dim,
            spatial_merge_size=self._spatial_merge_size,
            theta=self._rope_theta,
            pad_to=padded_head_dim,
        )

    def forward(
        self,
        pixel_values: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        x = self.patch_embed.forward(pixel_values)

        for block in self.blocks:
            x = block.forward(x, pos_embeds=pos_embeds)

        x = self.merger.forward(x)
        return x
