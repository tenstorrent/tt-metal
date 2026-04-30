# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import ttnn

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_CHUNK_SIZE = 128


@dataclass
class Qwen25VlContext:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    fsdp_mesh_axis: int | None = None


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L769
class Qwen25VlTextEncoder(Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        rope_theta: float,
        mrope_section: Sequence[int],
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        # FSDP: For encoders, we can only use FSDP if there's a separate axis from TP.
        # Since the encoder runs on a submesh (e.g., 1x4), we need to check if the other axis
        # has size > 1. If the mesh is 1xN, FSDP can't be enabled because there's no second axis.
        fsdp_mesh_axis = None
        if is_fsdp and parallel_config is not None:
            tp_axis = parallel_config.tensor_parallel.mesh_axis
            # Check if there's a different axis that can be used for FSDP
            other_axis = 1 - tp_axis  # If TP is on axis 1, check axis 0; if TP is on axis 0, check axis 1
            if device.shape[other_axis] > 1:
                fsdp_mesh_axis = other_axis

        ctx = Qwen25VlContext(
            device=device,
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        self.embed_tokens = Embedding(vocab_size, hidden_size, device=ctx.device, mesh_axis=ctx.tp_axis)
        self.layers = ModuleList(
            Qwen25VlDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                rms_norm_eps=rms_norm_eps,
                ctx=ctx,
            )
            for _ in range(num_hidden_layers)
        )
        self.norm = Qwen25VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)

        self._device = ctx.device
        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager
        self._mrope_section = mrope_section
        self._head_dim = hidden_size // num_attention_heads
        self._rope_theta = rope_theta

    def forward(
        self,
        input_ids: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> list[ttnn.Tensor]:
        batch_size, seq_len = input_ids.shape
        input_ids, pos_embeds, attention_bias, padded_seq_len = self._pad_inputs(
            input_ids, pos_embeds, attention_mask, seq_len
        )

        input_embeds = self.embed_tokens.forward(input_ids)
        return self._forward_decoder(
            input_embeds,
            pos_embeds=pos_embeds,
            attention_bias=attention_bias,
            original_seq_len=seq_len,
            padded_seq_len=padded_seq_len,
        )

    def forward_embeds(
        self,
        inputs_embeds: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> list[ttnn.Tensor]:
        batch_size, seq_len, _ = inputs_embeds.shape
        if attention_mask is not None:
            if seq_len < MAX_CHUNK_SIZE:
                padded_seq_len = -(-seq_len // 32) * 32
            else:
                padded_seq_len = -(-seq_len // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE

            if padded_seq_len != seq_len:
                inputs_embeds = ttnn.pad(inputs_embeds, [(0, 0), (0, padded_seq_len - seq_len), (0, 0)], value=0.0)
            pos_embeds = tuple(ttnn.pad(x, [(0, padded_seq_len - seq_len), (0, 0)], value=0) for x in pos_embeds)

            assert attention_mask.shape == (batch_size, seq_len)
            attention_mask = ttnn.pad(attention_mask, [(0, padded_seq_len - seq_len)], value=0)
            attention_bias = prepare_attention_bias(attention_mask)
        else:
            padded_seq_len = seq_len
            attention_bias = None

        del attention_mask

        return self._forward_decoder(
            inputs_embeds,
            pos_embeds=pos_embeds,
            attention_bias=attention_bias,
            original_seq_len=seq_len,
            padded_seq_len=padded_seq_len,
        )

    def _pad_inputs(
        self,
        input_ids: ttnn.Tensor,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        attention_mask: ttnn.Tensor | None,
        seq_len: int,
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor], ttnn.Tensor | None, int]:
        if attention_mask is not None:
            if seq_len < MAX_CHUNK_SIZE:
                padded_seq_len = -(-seq_len // 32) * 32
            else:
                padded_seq_len = -(-seq_len // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE

            input_ids = ttnn.pad(input_ids, [(0, padded_seq_len - seq_len)], value=0)
            pos_embeds = tuple(ttnn.pad(x, [(0, padded_seq_len - seq_len), (0, 0)], value=0) for x in pos_embeds)

            batch_size = input_ids.shape[0]
            assert attention_mask.shape == (batch_size, seq_len)
            attention_mask = ttnn.pad(attention_mask, [(0, padded_seq_len - seq_len)], value=0)
            attention_bias = prepare_attention_bias(attention_mask)
        else:
            padded_seq_len = seq_len
            attention_bias = None

        return input_ids, pos_embeds, attention_bias, padded_seq_len

    def _forward_decoder(
        self,
        hidden_states: ttnn.Tensor,
        *,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        attention_bias: ttnn.Tensor | None,
        original_seq_len: int,
        padded_seq_len: int,
    ) -> list[ttnn.Tensor]:
        if self._tp_axis is not None:
            hidden_states = self._ccl_manager.all_gather_persistent_buffer(
                hidden_states, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True
            )
            hidden_states = ttnn.clone(hidden_states)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer.forward(
                hidden_states,
                attention_bias=attention_bias,
                pos_embeds=pos_embeds,
            )

        hidden_states = self.norm.forward(hidden_states)

        if padded_seq_len != original_seq_len:
            hidden_states_list = [x[:, :original_seq_len, :] for x in [hidden_states]]
        else:
            hidden_states_list = [hidden_states]

        return hidden_states_list

    def create_rope_tensors(
        self, batch_size: int, sequence_length: int, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return create_rope_tensors(
            batch_size,
            sequence_length,
            attention_mask,
            self._head_dim,
            self._rope_theta,
            self._mrope_section,
        )


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L684
class Qwen25VlDecoderLayer(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        hidden_act: str,
        rms_norm_eps: float,
        ctx: Qwen25VlContext,
    ) -> None:
        super().__init__()

        self.self_attn = Qwen25VlAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            ctx=ctx,
        )
        self.mlp = Qwen25VlMlp(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act, ctx=ctx
        )
        self.input_layernorm = Qwen25VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)
        self.post_attention_layernorm = Qwen25VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attention_bias: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x, attention_bias=attention_bias, pos_embeds=pos_embeds)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        return x + residual


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L590
class Qwen25VlAttention(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        ctx: Qwen25VlContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        if hidden_size % num_heads != 0:
            msg = f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            raise ValueError(msg)

        head_dim = hidden_size // num_heads
        tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        group_count = num_key_value_heads
        group_size = num_heads // num_key_value_heads

        opt_group_count, opt_group_size, split_factor = optimal_groups(group_count, group_size, tp_factor)
        padded_heads = opt_group_count * opt_group_size

        self.qkv_proj = ColParallelLinear(
            hidden_size,
            (padded_heads + 2 * opt_group_count) * head_dim,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.o_proj = ColParallelLinear(
            padded_heads * head_dim,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )

        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            # packer_l1_acc=True,
        )

        self._head_dim = head_dim
        self._group_count = group_count
        self._group_size = group_size
        self._num_local_heads = padded_heads // tp_factor
        self._num_local_kv_heads = opt_group_count // tp_factor
        self._group_size_padding = opt_group_size * split_factor - group_size
        self._group_count_padding = opt_group_count - group_count * split_factor
        self._split_factor = split_factor
        self._tp_axis = ctx.tp_axis
        self._tp_factor = tp_factor
        self._device = ctx.device
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def _prepare_qkv(q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
            q = q.unflatten(0, [self._group_count, self._group_size, self._head_dim])
            k = k.unflatten(0, [self._group_count, 1, self._head_dim])
            v = v.unflatten(0, [self._group_count, 1, self._head_dim])

            # pad group size
            q = _pad(q, self._group_size_padding, dim=1)

            # split groups
            s = self._split_factor
            q = q.flatten(0, 1).unflatten(0, [self._group_count * s, -1])
            k = k.repeat_interleave(s, dim=0)
            v = v.repeat_interleave(s, dim=0)

            # pad group count
            q = _pad(q, self._group_count_padding, dim=0)
            k = _pad(k, self._group_count_padding, dim=0)
            v = _pad(v, self._group_count_padding, dim=0)

            # fuse
            q = q.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_heads])
            k = k.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_kv_heads])
            v = v.flatten(0, 1).unflatten(0, [self._tp_factor, self._num_local_kv_heads])

            return torch.cat([q, k, v], dim=1).flatten(0, 2)

        if "q_proj.weight" in state and "k_proj.weight" in state and "v_proj.weight" in state:
            state["qkv_proj.weight"] = _prepare_qkv(
                state.pop("q_proj.weight"), state.pop("k_proj.weight"), state.pop("v_proj.weight")
            )

        if "q_proj.bias" in state and "k_proj.bias" in state and "v_proj.bias" in state:
            state["qkv_proj.bias"] = _prepare_qkv(
                state.pop("q_proj.bias"), state.pop("k_proj.bias"), state.pop("v_proj.bias")
            )

        if "o_proj.weight" in state:
            o = state["o_proj.weight"]

            o = o.unflatten(1, [self._group_count, self._group_size, self._head_dim])

            # pad group size
            o = _pad(o, self._group_size_padding, dim=2)

            # split groups
            o = o.flatten(1, 2).unflatten(1, [self._group_count * self._split_factor, -1])

            # pad group count
            o = _pad(o, self._group_count_padding, dim=1)

            state["o_proj.weight"] = o.flatten(1, 3)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        attention_bias: ttnn.Tensor | None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        x = self.qkv_proj.forward(x)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(x, 1),
            num_heads=self._num_local_heads,
            num_kv_heads=self._num_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        cos, sin = pos_embeds
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            is_causal=attention_bias is None,
            program_config=self._sdpa_program_config(q.shape[2]),
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        x = ttnn.transformer.concatenate_heads(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = self.o_proj.forward(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x

    def _sdpa_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        grid_size = self._device.compute_with_storage_grid_size()

        seq_len = -(-seq_len // 32) * 32
        chunk_size = min(seq_len, MAX_CHUNK_SIZE)

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=chunk_size,
            k_chunk_size=chunk_size,
            exp_approx_mode=False,
        )


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L529
class Qwen25VlMlp(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ctx: Qwen25VlContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        # intermediate_size is much greater than hidden_size
        self.gate_proj = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.up_proj = ColParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
            fsdp_mesh_axis=ctx.fsdp_mesh_axis,
            ccl_manager=ctx.ccl_manager,
        )

        if hidden_act != "silu":
            msg = f"unsupported activation function: {hidden_act}"
            raise ValueError(msg)
        self.act_fn = ttnn.silu

        self._ccl_manager = ctx.ccl_manager
        self._tp_axis = ctx.tp_axis

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.act_fn(self.gate_proj.forward(x)) * self.up_proj.forward(x)
        x = self.down_proj(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x


class Qwen25VlRmsNorm(RMSNorm):
    def __init__(self, size: int, *, eps: float, ctx: Qwen25VlContext) -> None:
        super().__init__(size, norm_eps=eps, bias=False, mesh_device=ctx.device)

        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return super().forward(x, compute_kernel_config=self._compute_kernel_config)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    # interleaved format leads to lower PCC
    # return x * cos + ttnn.alt_complex_rotate90(x) * sin
    return x * cos + _rotate_half(x) * sin


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


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
        ctx: Qwen25VlVisionContext,
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


def optimal_groups(group_count: int, group_size: int, device_count: int) -> tuple[int, int, int]:
    # In order to distribute heads evenly on devices, three operations are possibly performed:
    # 1. Pad to increase group size.
    # 2. Pad to increase group count (= number of key/value heads).
    # 3. Split groups into smaller groups defined by a split factor.
    # For a particular split factor, padding sizes follow from the requirements that the padded
    # group size must be divisible by this factor and the new group count must be divisible by the
    # device count. We choose this factor such that memory requirements are minimized.

    best_split_factor = 1
    best_size = math.inf
    best_group_count = group_count
    best_group_size = group_size

    for s in range(1, group_size + 1):
        new_group_size = -(-group_size // s)  # = ceil(group_size / s)
        new_group_count = -(-group_count * s // device_count) * device_count

        # query heads + 2 * key/value heads
        size = new_group_size * new_group_count + 2 * new_group_count

        if size < best_size:
            best_size = size
            best_split_factor = s
            best_group_count = new_group_count
            best_group_size = new_group_size

    return best_group_count, best_group_size, best_split_factor


def _pad(t: torch.Tensor, amount: int, *, dim: int) -> torch.Tensor:
    """Pad tensor with `amount` zeros on the end of dimension `dim`."""
    padding = [0] * (2 * t.ndim)
    padding[-(dim * 2 + 1)] = amount
    return torch.nn.functional.pad(t, padding)


def prepare_attention_bias(attention_mask: ttnn.Tensor) -> ttnn.Tensor:
    batch_size, seq_len = attention_mask.shape

    # convert to causal attention mask
    attention_mask = attention_mask.reshape([batch_size, 1, 1, seq_len])
    attention_mask = ttnn.expand(attention_mask, [-1, -1, seq_len, -1])
    attention_mask = ttnn.tril(attention_mask)

    attention_mask = (attention_mask - 1.0) * math.inf

    return ttnn.clone(attention_mask, dtype=ttnn.bfloat4_b)


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L491
# and https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L545
def create_rope_tensors(
    batch_size: int,
    sequence_length: int,
    attention_mask: torch.Tensor | None,
    head_dim: int,
    rope_theta: float,
    mrope_section: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if attention_mask is not None:
        assert attention_mask.shape == (batch_size, sequence_length)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    else:
        position_ids = torch.arange(sequence_length).view(1, 1, -1).expand(3, batch_size, -1)

    inv_freq = rope_theta ** (-torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim)
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, batch_size, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    s = list(mrope_section) * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(s, dim=-1))], dim=-1).unsqueeze(1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(s, dim=-1))], dim=-1).unsqueeze(1)

    return cos, sin
