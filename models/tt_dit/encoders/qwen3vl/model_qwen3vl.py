# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL TEXT tower ported to tt_dit.

Adapted from the Qwen2.5-VL text encoder anchor
(models/tt_dit/encoders/qwen25vl/model_qwen25vl.py) and the HuggingFace reference
transformers/models/qwen3_vl/modeling_qwen3_vl.py (transformers 5.10.2).

Deltas vs Qwen2.5-VL text:
  1. q_norm / k_norm: Qwen3VLTextRMSNorm(head_dim) applied to the query and key heads
     AFTER the qkv projection / head split and BEFORE RoPE. (reference lines 466-469, 482-483)
  2. Interleaved mRoPE: mrope_interleaved=True, mrope_section=[24,20,20], rope_theta=5e6.
     (reference Qwen3VLTextRotaryEmbedding.forward / apply_interleaved_mrope, lines 354-389)
  3. Intermediate hidden-state collection following HF's output_hidden_states tuple semantics:
     hidden_states[0] = embedding output, hidden_states[k] = output after decoder layer k.
"""

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
from ...utils import tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_CHUNK_SIZE = 128


@dataclass
class Qwen3VlContext:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    fsdp_mesh_axis: int | None = None


# adapted from transformers/models/qwen3_vl/modeling_qwen3_vl.py::Qwen3VLTextModel (line 741)
class Qwen3VlTextEncoder(Module):
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
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        mrope_section: Sequence[int],
        mrope_interleaved: bool = True,
        device: ttnn.MeshDevice,
        parallel_config: EncoderParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        # FSDP: For encoders, we can only use FSDP if there's a separate axis from TP.
        fsdp_mesh_axis = None
        if is_fsdp and parallel_config is not None:
            tp_axis = parallel_config.tensor_parallel.mesh_axis
            other_axis = 1 - tp_axis
            if device.shape[other_axis] > 1:
                fsdp_mesh_axis = other_axis

        ctx = Qwen3VlContext(
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
            Qwen3VlDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                ctx=ctx,
            )
            for _ in range(num_hidden_layers)
        )
        self.norm = Qwen3VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)

        self._device = ctx.device
        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager
        self._mrope_section = mrope_section
        self._mrope_interleaved = mrope_interleaved
        self._head_dim = head_dim
        self._rope_theta = rope_theta

    def forward(
        self,
        input_ids: ttnn.Tensor,
        *,
        attention_mask: ttnn.Tensor | None = None,
        pos_embeds: tuple[ttnn.Tensor, ttnn.Tensor],
        select_layers: Sequence[int] | None = None,
    ) -> list[ttnn.Tensor]:
        """Run the text tower.

        Returns a list of hidden states following HF's `output_hidden_states` tuple semantics:
          index 0   = embedding output (input_embeds)
          index k   = output AFTER decoder layer k (before the final norm),
                      except the LAST entry, which is the final-norm output
                      (matching HF where the last hidden state is `self.norm(...)`).

        If `select_layers` is provided, only those indices (into the full-length
        hidden-state tuple, where index 0 is the embedding) are returned, in order.

        NOTE on the final norm: HF's `output_hidden_states` tuple stores the *pre-norm*
        layer outputs for indices 0..num_layers-1 and applies `self.norm` only to
        `last_hidden_state`. When the caller taps intermediate layers (e.g. KREA-2's
        select_layers 2,5,...,35) it reads the PRE-NORM outputs, exactly as returned
        here. The full-final-norm output is available as the extra trailing entry.
        """
        batch_size, seq_len = input_ids.shape

        if attention_mask is not None:
            if seq_len < MAX_CHUNK_SIZE:
                padded_seq_len = -(-seq_len // 32) * 32
            else:
                padded_seq_len = -(-seq_len // MAX_CHUNK_SIZE) * MAX_CHUNK_SIZE

            input_ids = ttnn.pad(input_ids, [(0, padded_seq_len - seq_len)], value=0)
            pos_embeds = tuple(ttnn.pad(x, [(0, padded_seq_len - seq_len), (0, 0)], value=0) for x in pos_embeds)

            assert attention_mask.shape == (batch_size, seq_len)
            attention_mask = ttnn.pad(attention_mask, [(0, padded_seq_len - seq_len)], value=0)
            attention_bias = prepare_attention_bias(attention_mask)
        else:
            padded_seq_len = seq_len
            attention_bias = None

        del attention_mask

        input_embeds = self.embed_tokens.forward(input_ids)

        if self._tp_axis is not None:
            input_embeds = self._ccl_manager.all_gather_persistent_buffer(
                input_embeds, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True
            )
            input_embeds = ttnn.clone(input_embeds)

        hidden_states = input_embeds

        # HF hidden_states tuple: index 0 = embedding output, index k = output after layer k.
        hidden_states_list = [hidden_states]

        for decoder_layer in self.layers:
            hidden_states = decoder_layer.forward(
                hidden_states,
                attention_bias=attention_bias,
                pos_embeds=pos_embeds,
            )
            hidden_states_list.append(hidden_states)

        # HF applies the final norm only to `last_hidden_state`. Append it as an extra
        # trailing entry so callers that want the fully-normed output can read it.
        hidden_states_list.append(self.norm.forward(hidden_states))

        if padded_seq_len != seq_len:
            hidden_states_list = [x[:, :seq_len, :] for x in hidden_states_list]

        if select_layers is not None:
            hidden_states_list = [hidden_states_list[i] for i in select_layers]

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
            mrope_interleaved=self._mrope_interleaved,
        )


# adapted from Qwen3VLTextDecoderLayer (line 528)
class Qwen3VlDecoderLayer(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        hidden_act: str,
        rms_norm_eps: float,
        ctx: Qwen3VlContext,
    ) -> None:
        super().__init__()

        self.self_attn = Qwen3VlAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            ctx=ctx,
        )
        self.mlp = Qwen3VlMlp(
            hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act, ctx=ctx
        )
        self.input_layernorm = Qwen3VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)
        self.post_attention_layernorm = Qwen3VlRmsNorm(hidden_size, eps=rms_norm_eps, ctx=ctx)

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


# adapted from Qwen3VLTextAttention (line 440)
class Qwen3VlAttention(Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        ctx: Qwen3VlContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        group_count = num_key_value_heads
        group_size = num_heads // num_key_value_heads

        opt_group_count, opt_group_size, split_factor = optimal_groups(group_count, group_size, tp_factor)
        padded_heads = opt_group_count * opt_group_size

        self.qkv_proj = ColParallelLinear(
            hidden_size,
            (padded_heads + 2 * opt_group_count) * head_dim,
            bias=False,
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

        # Qwen3 delta: per-head RMSNorm over head_dim, applied to q and k before RoPE.
        self.q_norm = Qwen3VlRmsNorm(head_dim, eps=rms_norm_eps, ctx=ctx)
        self.k_norm = Qwen3VlRmsNorm(head_dim, eps=rms_norm_eps, ctx=ctx)

        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
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
        def _prepare_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            q = q.unflatten(0, [self._group_count, self._group_size, self._head_dim])
            k = k.unflatten(0, [self._group_count, 1, self._head_dim])
            v = v.unflatten(0, [self._group_count, 1, self._head_dim])

            q = _pad(q, self._group_size_padding, dim=1)

            s = self._split_factor
            q = q.flatten(0, 1).unflatten(0, [self._group_count * s, -1])
            k = k.repeat_interleave(s, dim=0)
            v = v.repeat_interleave(s, dim=0)

            q = _pad(q, self._group_count_padding, dim=0)
            k = _pad(k, self._group_count_padding, dim=0)
            v = _pad(v, self._group_count_padding, dim=0)

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
            o = _pad(o, self._group_size_padding, dim=2)
            o = o.flatten(1, 2).unflatten(1, [self._group_count * self._split_factor, -1])
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

        # Qwen3 delta: per-head RMSNorm over head_dim on q and k, before RoPE.
        # q/k are [batch, heads, seq, head_dim]; RMSNorm reduces over the last dim.
        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

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


# adapted from Qwen3VLTextMLP (line 512)
class Qwen3VlMlp(Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ctx: Qwen3VlContext,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

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


class Qwen3VlRmsNorm(RMSNorm):
    def __init__(self, size: int, *, eps: float, ctx: Qwen3VlContext) -> None:
        super().__init__(size, norm_eps=eps, bias=False, mesh_device=ctx.device)

        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return super().forward(x, compute_kernel_config=self._compute_kernel_config)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    # HF-style rotate_half (matches Qwen3VL apply_rotary_pos_emb, reference line 414).
    return x * cos + _rotate_half(x) * sin


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ttnn.concat([ttnn.neg(x2), x1], dim=-1)


def optimal_groups(group_count: int, group_size: int, device_count: int) -> tuple[int, int, int]:
    best_split_factor = 1
    best_size = math.inf
    best_group_count = group_count
    best_group_size = group_size

    for s in range(1, group_size + 1):
        new_group_size = -(-group_size // s)
        new_group_count = -(-group_count * s // device_count) * device_count

        size = new_group_size * new_group_count + 2 * new_group_count

        if size < best_size:
            best_size = size
            best_split_factor = s
            best_group_count = new_group_count
            best_group_size = new_group_size

    return best_group_count, best_group_size, best_split_factor


def _pad(t: torch.Tensor, amount: int, *, dim: int) -> torch.Tensor:
    padding = [0] * (2 * t.ndim)
    padding[-(dim * 2 + 1)] = amount
    return torch.nn.functional.pad(t, padding)


def prepare_attention_bias(attention_mask: ttnn.Tensor) -> ttnn.Tensor:
    batch_size, seq_len = attention_mask.shape

    attention_mask = attention_mask.reshape([batch_size, 1, 1, seq_len])
    attention_mask = ttnn.expand(attention_mask, [-1, -1, seq_len, -1])
    attention_mask = tensor.tril(attention_mask)

    attention_mask = (attention_mask - 1.0) * math.inf

    return ttnn.clone(attention_mask, dtype=ttnn.bfloat4_b)


# adapted from Qwen3VLTextRotaryEmbedding.forward / apply_interleaved_mrope (reference lines 354-389)
def create_rope_tensors(
    batch_size: int,
    sequence_length: int,
    attention_mask: torch.Tensor | None,
    head_dim: int,
    rope_theta: float,
    mrope_section: Sequence[int],
    *,
    mrope_interleaved: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin for Qwen3-VL interleaved multimodal RoPE.

    For TEXT-only input all three rope sections (temporal/height/width) use identical
    position indices, so `freqs[0] == freqs[1] == freqs[2]` and the interleaving step is a
    no-op; the resulting cos/sin are the standard single-position-per-token RoPE tensors.
    We nevertheless replicate the reference construction exactly (including the interleaved
    mrope reorganization) so the code is correct if non-text position_ids are ever supplied.

    Returns cos, sin each of shape [batch, 1, sequence_length, head_dim].
    """
    if attention_mask is not None:
        assert attention_mask.shape == (batch_size, sequence_length)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    else:
        position_ids = torch.arange(sequence_length).view(1, 1, -1).expand(3, batch_size, -1)

    # inv_freq per reference compute_default_rope_parameters (line 347)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, batch_size, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    # freqs: (3, batch, seq, head_dim // 2)
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

    if mrope_interleaved:
        # apply_interleaved_mrope (reference lines 374-389): reorganize the chunked
        # [T..H..W..] frequency layout into the interleaved [THWTHW...] layout.
        freqs_t = freqs[0].clone()  # start from the temporal section
        for dim, offset in enumerate((1, 2), start=1):  # height, width
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        freqs = freqs_t  # (batch, seq, head_dim // 2)
    else:
        # Non-interleaved (Qwen2.5-style) chunked section selection.
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        s = list(mrope_section) * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(s, dim=-1))], dim=-1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(s, dim=-1))], dim=-1)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1)  # (batch, 1, seq, head_dim)
    sin = emb.sin().unsqueeze(1)

    return cos, sin
