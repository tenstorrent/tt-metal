# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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

        if attention_mask is not None:
            if seq_len < MAX_CHUNK_SIZE:
                # make sequence length a multiple of tile size
                padded_seq_len = -(-seq_len // 32) * 32
            else:
                # make sequence length a multiple of MAX_CHUNK_SIZE
                padded_seq_len = -(-seq_len // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE

            input_ids = ttnn.pad(input_ids, [(0, padded_seq_len - seq_len)], value=0)
            pos_embeds = tuple(ttnn.pad(x, [(0, padded_seq_len - seq_len), (0, 0)], value=0) for x in pos_embeds)

            assert attention_mask.shape == (batch_size, seq_len)
            attention_mask = ttnn.pad(attention_mask, [(0, padded_seq_len - seq_len)], value=0)
            attention_bias = prepare_attention_bias(attention_mask)
        else:
            # padding is only required by `ttnn.transformer.scaled_dot_product_attention` when using
            # an attention mask
            padded_seq_len = seq_len

            attention_bias = None

        del attention_mask

        input_embeds = self.embed_tokens.forward(input_ids)

        if self._tp_axis is not None:
            input_embeds = self._ccl_manager.all_gather_persistent_buffer(
                input_embeds, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True
            )
            # clone to move out of persistent buffer
            input_embeds = ttnn.clone(input_embeds)

        hidden_states = input_embeds
        # hidden_states_list = []

        for decoder_layer in self.layers:
            # hidden_states_list.append(hidden_states)

            hidden_states = decoder_layer.forward(
                hidden_states,
                attention_bias=attention_bias,
                pos_embeds=pos_embeds,
            )

        hidden_states = self.norm.forward(hidden_states)
        # hidden_states_list.append(hidden_states)

        if padded_seq_len != seq_len:
            # hidden_states_list = [x[:, :seq_len, :] for x in hidden_states_list]
            hidden_states_list = [x[:, :seq_len, :] for x in [hidden_states]]
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


def optimal_groups(group_count: int, group_size: int, device_count: int) -> tuple[int, int, int]:
    # In order to distribute heads evenly on devices, three operations are possibly performed:
    # 1. Pad to increase group size.
    # 2. Pad to increase group count (= number of key/value heads).
    # 3. Split groups into smaller groups defined by a split factor.
    # For a particular split factor, padding sizes follow from the requirements that the padded
    # group size must be divisible by this factor and the new group count must be divisible by the
    # device count. We choose this factor such that memory requirments are minimized.

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
