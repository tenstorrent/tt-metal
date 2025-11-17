# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import ttnn
from models.tt_transformers.tt.common import get_rot_transformation_mat

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils import tensor

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class Qwen25VlContext:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None


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
    ) -> None:
        super().__init__()

        ctx = Qwen25VlContext(
            device=device,
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            ccl_manager=ccl_manager,
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
                mrope_section=mrope_section,
                ctx=ctx,
            )
            for _ in range(num_hidden_layers)
        )
        self.norm = Qwen25VlRmsNorm(hidden_size, eps=rms_norm_eps, device=ctx.device)
        # self.rotary_emb = Qwen25VlRotaryEmbedding(
        #     ctx=ctx, head_dim=hidden_size // num_attention_heads, rope_theta=rope_theta
        # )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def forward(
        self,
        input_ids: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        # position_ids: ttnn.Tensor,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> list[ttnn.Tensor]:
        batch_size, seq_len = input_ids.shape

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)

            # convert to causal attention mask
            attention_mask = ttnn.unsqueeze(attention_mask, 1)
            attention_mask = ttnn.expand(attention_mask, [-1, seq_len, -1])
            attention_mask = ttnn.tril(attention_mask)

            attention_mask = (attention_mask - 1.0) * math.inf

            attention_mask = ttnn.clone(attention_mask, dtype=ttnn.bfloat4_b)

        input_embeds = self.embed_tokens.forward(input_ids)
        # pos_embeds = self.rotary_emb.forward(position_ids)

        if self._tp_axis is not None:
            input_embeds = self._ccl_manager.all_gather_persistent_buffer(
                input_embeds, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True
            )

        hidden_states = input_embeds
        hidden_states_list = []

        for decoder_layer in self.layers:
            hidden_states_list.append(hidden_states)

            hidden_states = decoder_layer.forward(
                hidden_states,
                causal_attn_mask=attention_mask,
                pos_embeds=pos_embeds,
            )

        hidden_states = self.norm.forward(hidden_states)
        hidden_states_list.append(hidden_states)

        return hidden_states_list


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
        mrope_section: Sequence[int],
        ctx: Qwen25VlContext,
    ) -> None:
        super().__init__()

        self.self_attn = Qwen25VlAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            mrope_section=mrope_section,
            ctx=ctx,
        )
        self.mlp = Qwen25VlMlp(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act, ctx=ctx
        )
        self.input_layernorm = Qwen25VlRmsNorm(hidden_size, eps=rms_norm_eps, device=ctx.device)
        self.post_attention_layernorm = Qwen25VlRmsNorm(hidden_size, eps=rms_norm_eps, device=ctx.device)

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        causal_attn_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> ttnn.Tensor:
        residual = x
        x = self.input_layernorm.forward(x)
        x = self.self_attn.forward(x, causal_mask=causal_attn_mask, pos_embeds=pos_embeds)
        x += residual

        residual = x
        x = self.post_attention_layernorm.forward(x)
        x = self.mlp.forward(x)
        x += residual

        return x


# adapted from https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L590
class Qwen25VlAttention(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        mrope_section: Sequence[int],
        ctx: Qwen25VlContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        if hidden_size % num_heads != 0:
            msg = f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            raise ValueError(msg)

        self._qkv_parallel = True  # TODO: remove

        head_dim = hidden_size // num_heads
        tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None and self._qkv_parallel else 1

        repeat_kv_heads = 1
        # f = num_heads // num_key_value_heads
        # num_additional_heads = math.inf
        # repeat_kv_heads = 1
        # for a in range(1, f + 1):
        #     if f % a != 0:
        #         continue
        #     pad, pad_kv = _num_head_padding(num_heads, num_key_value_heads * a, tp_factor)
        #     value = pad + pad_kv + (a - 1) * num_key_value_heads
        #     if value < num_additional_heads:
        #         num_additional_heads = value
        #         repeat_kv_heads = a

        # pad, pad_kv = _num_head_padding(num_heads, num_key_value_heads * repeat_kv_heads, tp_factor)

        self.qkv_proj = ColParallelLinear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads * repeat_kv_heads) * head_dim,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis if self._qkv_parallel else None,
        )
        self.o_proj = ColParallelLinear(
            num_heads * head_dim, hidden_size, bias=False, mesh_device=ctx.device, mesh_axis=ctx.tp_axis
        )

        grid_size = ctx.device.compute_with_storage_grid_size()

        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            # packer_l1_acc=True,
        )

        self._rope_mat = tensor.from_torch(
            get_rot_transformation_mat(head_dim),
            device=ctx.device,
        )  # TODO: bloat4_b?

        self._head_dim = head_dim
        self._mrope_section = mrope_section
        self._num_local_heads = num_heads // tp_factor
        self._num_local_kv_heads = repeat_kv_heads * num_key_value_heads // tp_factor
        self._repeat_kv_heads = repeat_kv_heads
        self._tp_axis = ctx.tp_axis
        self._tp_factor = tp_factor
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def _merge_tensors(q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
            # if self.padding_config is not None:
            #     q = pad_weight_tensor(q, self.padding_config, pad_output_dim=True)
            #     k = pad_weight_tensor(k, self.padding_config, pad_output_dim=True)
            #     v = pad_weight_tensor(v, self.padding_config, pad_output_dim=True)

            # q = q.unflatten(0, [self._num_heads, 2, self._head_dim // 2]).transpose(1, 2).flatten(0, 2)
            # k = k.unflatten(0, [self._num_kv_heads, 2, self._head_dim // 2]).transpose(1, 2).flatten(0, 2)

            # repeat KV heads
            n = self._repeat_kv_heads
            k = k.unflatten(0, [-1, self._head_dim]).repeat_interleave(n, dim=0).flatten(0, 1)
            v = v.unflatten(0, [-1, self._head_dim]).repeat_interleave(n, dim=0).flatten(0, 1)

            # fuse
            q = q.unflatten(0, [self._tp_factor, self._num_local_heads, self._head_dim])
            k = k.unflatten(0, [self._tp_factor, self._num_local_kv_heads, self._head_dim])
            v = v.unflatten(0, [self._tp_factor, self._num_local_kv_heads, self._head_dim])
            return torch.cat([q, k, v], dim=1).flatten(0, 2)

        if "q_proj.weight" in state and "k_proj.weight" in state and "v_proj.weight" in state:
            state["qkv_proj.weight"] = _merge_tensors(
                state.pop("q_proj.weight"), state.pop("k_proj.weight"), state.pop("v_proj.weight")
            )

        if "q_proj.bias" in state and "k_proj.bias" in state and "v_proj.bias" in state:
            state["qkv_proj.bias"] = _merge_tensors(
                state.pop("q_proj.bias"), state.pop("k_proj.bias"), state.pop("v_proj.bias")
            )

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        causal_mask: ttnn.Tensor | None,
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

        # q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self._rope_mat, is_decode_mode=False)
        # k = ttnn.experimental.rotary_embedding_llama(k, cos, sin, self._rope_mat, is_decode_mode=False)

        # k = ttnn.repeat_interleave(k, repeats=self._num_local_heads // self._num_local_kv_heads, dim=1)
        # v = ttnn.repeat_interleave(v, repeats=self._num_local_heads // self._num_local_kv_heads, dim=1)

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=ttnn.unsqueeze(causal_mask, 1) if causal_mask is not None else None,
            is_causal=causal_mask is None,
            program_config=self._sdpa_program_config,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        x = ttnn.transformer.concatenate_heads(x)

        if self._tp_axis is not None and self._qkv_parallel:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = self.o_proj.forward(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        return x


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
            hidden_size, intermediate_size, bias=False, mesh_device=ctx.device, mesh_axis=ctx.tp_axis
        )
        self.up_proj = ColParallelLinear(
            hidden_size, intermediate_size, bias=False, mesh_device=ctx.device, mesh_axis=ctx.tp_axis
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            mesh_device=ctx.device,
            mesh_axis=ctx.tp_axis,
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


class Qwen25VlRmsNorm(Module):
    def __init__(self, size: int, *, eps: float, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.weight = Parameter(total_shape=[size], device=device)
        self._eps = eps

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].reshape([-1])

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        norm = ttnn.mean(ttnn.pow(x, 2), dim=-1, keepdim=True)
        norm = ttnn.rsqrt(norm + self._eps)
        return x * (norm * self.weight.data)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    return x * cos + _rotate_half(x) * sin


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)
