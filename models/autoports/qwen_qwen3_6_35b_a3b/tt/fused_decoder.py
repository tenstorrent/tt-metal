# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused TTNN decoder layer for ``Qwen/Qwen3.6-35B-A3B``.

This stage preserves the functional decoder public contract while replacing
same-input projection groups and adjacent activation/multiply pairs with fused
TTNN runtime paths.  Setup helpers may prepare packed weights on the host; the
measured prefill/decode forwards stay on device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import ttnn
from models.common.lightweightmodule import LightweightModule

from .functional_decoder import (
    DEFAULT_MOE_CHUNK_SIZE,
    HIDDEN_SIZE,
    LINEAR_ATTENTION_CHUNK_SIZE,
    MODEL_ID,
    TILE_SIZE,
    FunctionalDecoderResult,
    QwenFullAttentionCache,
    QwenLinearAttentionState,
    _as_device_tensor,
    _concat_dim2_bounded,
    _l2_norm_last_dim,
    _layer_state,
    _QwenTextConfig,
    _require,
    _rms_norm,
    _rms_weight,
    _row_major_core_rangeset,
    _shape,
    _slice,
    _slice_last,
    _sparse_matmul_program_config,
    _text_config,
)

_SILU = [ttnn.UnaryOpType.SILU]
_SIGMOID = [ttnn.UnaryOpType.SIGMOID]
_SOFTPLUS = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)]


@dataclass(frozen=True)
class FusedDecoderGraphSummary:
    """Small runtime-independent summary used by tests and docs."""

    packed_attention_qkgv: bool
    packed_linear_attention_inputs: bool
    packed_shared_expert_gate_up: bool
    packed_routed_expert_gate_up: bool
    binary_activation_folding: bool
    dedicated_rotary_embedding_batch1: bool


GRAPH_SUMMARY = FusedDecoderGraphSummary(
    packed_attention_qkgv=True,
    packed_linear_attention_inputs=True,
    packed_shared_expert_gate_up=True,
    packed_routed_expert_gate_up=True,
    binary_activation_folding=True,
    dedicated_rotary_embedding_batch1=True,
)


def _packed_linear_weight(state: dict[str, Any], keys: tuple[str, ...], *, device, dtype) -> ttnn.Tensor:
    import torch

    packed = torch.cat([_require(state, key) for key in keys], dim=0)
    return _as_device_tensor(
        packed.transpose(-1, -2).contiguous().unsqueeze(0).unsqueeze(0), device=device, dtype=dtype
    )


def _packed_gate_up_expert_weight(state: dict[str, Any], key: str, *, device, dtype) -> ttnn.Tensor:
    return _as_device_tensor(
        _require(state, key).transpose(-1, -2).unsqueeze(0).contiguous(), device=device, dtype=dtype
    )


def _silu_mul_fused(gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.mul(
        gate,
        up,
        input_tensor_a_activations=_SILU,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _partial_rope_fallback(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, rotary_dim: int) -> ttnn.Tensor:
    head_dim = _shape(x)[-1]
    x_rot = _slice_last(x, 0, rotary_dim)
    x_pass = _slice_last(x, rotary_dim, head_dim)
    half = rotary_dim // 2
    x1 = _slice_last(x_rot, 0, half)
    x2 = _slice_last(x_rot, half, rotary_dim)
    rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_rot = ttnn.add(
        ttnn.mul(x_rot, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.mul(rotated, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.concat([out_rot, x_pass], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _apply_partial_rope_fused(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    rotary_dim: int,
    *,
    decode_layout: bool,
) -> ttnn.Tensor:
    head_dim = _shape(x)[-1]
    x_rot = _slice_last(x, 0, rotary_dim)
    x_pass = _slice_last(x, rotary_dim, head_dim)
    cos_shape = _shape(cos)
    sin_shape = _shape(sin)
    can_use_dedicated = (
        cos_shape == sin_shape
        and len(cos_shape) == 4
        and cos_shape[0] == 1
        and cos_shape[1] == 1
        and cos_shape[-1] == rotary_dim
    )
    if not can_use_dedicated:
        return _partial_rope_fallback(x, cos, sin, rotary_dim)

    token_index = 0 if decode_layout else None
    out_rot = ttnn.experimental.rotary_embedding(
        x_rot,
        cos,
        sin,
        token_index,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if _shape(out_rot)[:-1] != _shape(x_rot)[:-1]:
        out_rot = _slice(out_rot, (0, 0, 0, 0), _shape(x_rot))
    return ttnn.concat([out_rot, x_pass], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class _FusedFullAttention(LightweightModule):
    def __init__(self, state: dict[str, Any], cfg: _QwenTextConfig, *, device, dtype):
        super().__init__()
        if cfg.attention_bias:
            raise NotImplementedError("Qwen3.6-35B-A3B text attention is expected to be bias-free")
        self.cfg = cfg
        self.device = device
        self.qkgv_proj = _packed_linear_weight(
            state,
            ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
            device=device,
            dtype=dtype,
        )
        self.o_proj = _packed_linear_weight(state, ("self_attn.o_proj.weight",), device=device, dtype=dtype)
        self.q_norm_weight = _rms_weight(state, "self_attn.q_norm.weight", device=device, add_unit_offset=True)
        self.k_norm_weight = _rms_weight(state, "self_attn.k_norm.weight", device=device, add_unit_offset=True)
        self.kv_width = cfg.num_key_value_heads * cfg.head_dim

    @property
    def q_width(self) -> int:
        return self.cfg.num_attention_heads * self.cfg.head_dim

    @property
    def q_gate_width(self) -> int:
        return 2 * self.q_width

    def _project_qkgv(self, x: ttnn.Tensor):
        packed = ttnn.linear(x, self.qkgv_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_and_gate = _slice_last(packed, 0, self.q_gate_width)
        k = _slice_last(packed, self.q_gate_width, self.q_gate_width + self.kv_width)
        v = _slice_last(packed, self.q_gate_width + self.kv_width, self.q_gate_width + 2 * self.kv_width)
        return q_and_gate, k, v

    def _reshape_prefill_heads(self, q_and_gate: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, batch: int, seq_len: int):
        q_and_gate = ttnn.reshape(q_and_gate, (batch, seq_len, self.cfg.num_attention_heads, 2 * self.cfg.head_dim))
        q = _slice_last(q_and_gate, 0, self.cfg.head_dim)
        gate = _slice_last(q_and_gate, self.cfg.head_dim, 2 * self.cfg.head_dim)
        gate = ttnn.reshape(gate, (1, batch, seq_len, self.q_width))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (batch, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (batch, seq_len, self.cfg.num_key_value_heads, self.cfg.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))
        return q, gate, k, v

    def _reshape_decode_heads(self, q_and_gate: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor, batch: int):
        q_and_gate = ttnn.reshape(q_and_gate, (batch, 1, self.cfg.num_attention_heads, 2 * self.cfg.head_dim))
        q = _slice_last(q_and_gate, 0, self.cfg.head_dim)
        gate = _slice_last(q_and_gate, self.cfg.head_dim, 2 * self.cfg.head_dim)
        q = ttnn.permute(q, (1, 0, 2, 3))
        gate = ttnn.reshape(gate, (1, batch, 1, self.q_width))
        q = ttnn.reshape(q, (1, batch, self.cfg.num_attention_heads, self.cfg.head_dim))
        k = ttnn.reshape(k, (1, batch, self.cfg.num_key_value_heads, self.cfg.head_dim))
        v = ttnn.reshape(v, (1, batch, self.cfg.num_key_value_heads, self.cfg.head_dim))
        return q, gate, k, v

    def _norm_and_rope(self, q: ttnn.Tensor, k: ttnn.Tensor, position_embeddings, *, decode_layout: bool):
        q = _rms_norm(q, self.q_norm_weight, self.cfg.rms_norm_eps)
        k = _rms_norm(k, self.k_norm_weight, self.cfg.rms_norm_eps)
        cos, sin = position_embeddings
        q = _apply_partial_rope_fused(q, cos, sin, self.cfg.rotary_dim, decode_layout=decode_layout)
        k = _apply_partial_rope_fused(k, cos, sin, self.cfg.rotary_dim, decode_layout=decode_layout)
        return q, k

    def _decode_update_mem_config(self, batch: int, head_dim: int, *, core_offset: int = 0):
        grid = self.device.compute_with_storage_grid_size()
        shard_grid = _row_major_core_rangeset(grid, batch, start=core_offset)
        shard_spec = ttnn.ShardSpec(
            shard_grid,
            (TILE_SIZE, head_dim),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    def _cache_update_tensor(self, tensor: ttnn.Tensor, *, batch: int, core_offset: int = 0) -> ttnn.Tensor:
        if _shape(tensor)[2] < TILE_SIZE:
            tensor = ttnn.pad(
                tensor,
                (1, batch, TILE_SIZE, self.cfg.head_dim),
                (0, 0, 0, 0),
                0.0,
            )
        return ttnn.to_memory_config(
            tensor, self._decode_update_mem_config(batch, self.cfg.head_dim, core_offset=core_offset)
        )

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_embeddings: tuple[ttnn.Tensor, ttnn.Tensor],
        kv_cache: QwenFullAttentionCache | None,
        page_table: ttnn.Tensor | None,
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        _, batch, seq_len, _ = _shape(hidden_states)
        q_and_gate, k, v = self._project_qkgv(hidden_states)
        q, gate, k, v = self._reshape_prefill_heads(q_and_gate, k, v, batch, seq_len)
        q, k = self._norm_and_rope(q, k, position_embeddings, decode_layout=False)

        keys = values = None
        if kv_cache is not None:
            if page_table is None:
                raise ValueError("full-attention paged prefill requires page_table when kv_cache is supplied")
            keys, values = kv_cache.keys, kv_cache.values
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
            if batch == 1:
                ttnn.experimental.paged_fill_cache(keys, k, fill_page_table, batch_idx=user_id)
                ttnn.experimental.paged_fill_cache(values, v, fill_page_table, batch_idx=user_id)
            else:
                for batch_idx in range(batch):
                    k_b = _slice(
                        k,
                        (batch_idx, 0, 0, 0),
                        (batch_idx + 1, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim),
                    )
                    v_b = _slice(
                        v,
                        (batch_idx, 0, 0, 0),
                        (batch_idx + 1, self.cfg.num_key_value_heads, seq_len, self.cfg.head_dim),
                    )
                    ttnn.experimental.paged_fill_cache(keys, k_b, fill_page_table, batch_idx=user_id + batch_idx)
                    ttnn.experimental.paged_fill_cache(values, v_b, fill_page_table, batch_idx=user_id + batch_idx)

        q_sdpa = ttnn.typecast(q, dtype=ttnn.bfloat16) if q.dtype != ttnn.bfloat16 else q
        k_sdpa = ttnn.typecast(k, dtype=ttnn.bfloat16) if k.dtype != ttnn.bfloat16 else k
        v_sdpa = ttnn.typecast(v, dtype=ttnn.bfloat16) if v.dtype != ttnn.bfloat16 else v

        if chunk_start_idx is not None:
            if kv_cache is None or page_table is None:
                raise ValueError("chunked full-attention prefill requires paged kv_cache and page_table")
            if isinstance(chunk_start_idx, ttnn.Tensor):
                attn_out = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_sdpa,
                    keys,
                    values,
                    page_table,
                    chunk_start_idx_tensor=chunk_start_idx,
                )
            else:
                attn_out = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_sdpa,
                    keys,
                    values,
                    page_table,
                    chunk_start_idx,
                )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                is_causal=True,
                attn_mask=attention_mask,
                scale=self.cfg.head_dim**-0.5,
            )

        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        attn_out = ttnn.reshape(attn_out, (1, batch, seq_len, self.q_width))
        attn_out = ttnn.mul(
            attn_out,
            gate,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.linear(attn_out, self.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_embeddings: tuple[ttnn.Tensor, ttnn.Tensor],
        kv_cache: QwenFullAttentionCache,
        page_table: ttnn.Tensor,
        current_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        _, _, batch, _ = _shape(hidden_states)
        q_and_gate, k, v = self._project_qkgv(hidden_states)
        q, gate, k, v = self._reshape_decode_heads(q_and_gate, k, v, batch)
        q, k = self._norm_and_rope(q, k, position_embeddings, decode_layout=True)

        k_update = self._cache_update_tensor(k, batch=batch)
        v_update = self._cache_update_tensor(v, batch=batch, core_offset=batch)
        ttnn.experimental.paged_fused_update_cache(
            kv_cache.keys,
            k_update,
            kv_cache.values,
            v_update,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )

        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache.keys,
            kv_cache.values,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=self.cfg.head_dim**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = ttnn.reshape(attn_out, (1, batch, 1, self.q_width))
        attn_out = ttnn.mul(
            attn_out,
            gate,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.linear(attn_out, self.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (1, 1, batch, self.cfg.hidden_size))


class _FusedLinearAttention(LightweightModule):
    def __init__(self, state: dict[str, Any], cfg: _QwenTextConfig, *, device, dtype):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
        self.value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.repeat_factor = cfg.linear_num_value_heads // cfg.linear_num_key_heads
        self.packed_qkv_zba_width = self.conv_dim + self.value_dim + 2 * cfg.linear_num_value_heads

        self.in_proj_qkv_zba = _packed_linear_weight(
            state,
            (
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight",
            ),
            device=device,
            dtype=dtype,
        )
        self.out_proj = _packed_linear_weight(state, ("linear_attn.out_proj.weight",), device=device, dtype=dtype)
        self.norm_weight = _rms_weight(state, "linear_attn.norm.weight", device=device, add_unit_offset=False)
        self.dt_bias = _as_device_tensor(
            _require(state, "linear_attn.dt_bias").reshape(1, 1, 1, -1).contiguous(), device=device, dtype=ttnn.bfloat16
        )

        import torch

        a = _require(state, "linear_attn.A_log").float().exp().neg().reshape(1, 1, 1, -1).contiguous()
        self.neg_exp_a_log = _as_device_tensor(a, device=device, dtype=ttnn.bfloat16)

        conv_weight = _require(state, "linear_attn.conv1d.weight")
        self.conv_weights = tuple(
            _as_device_tensor(
                conv_weight[:, 0, idx].reshape(1, 1, 1, -1).contiguous(), device=device, dtype=ttnn.bfloat16
            )
            for idx in range(cfg.linear_conv_kernel_dim)
        )

        chunk = LINEAR_ATTENTION_CHUNK_SIZE
        self.linear_chunk_size = chunk
        mask_shape = (1, 1, chunk, chunk)
        self.chunk_lower_mask = _as_device_tensor(
            torch.tril(torch.ones(mask_shape, dtype=torch.bfloat16)), device=device, dtype=ttnn.bfloat16
        )
        self.chunk_strict_lower_mask = _as_device_tensor(
            torch.tril(torch.ones(mask_shape, dtype=torch.bfloat16), diagonal=-1),
            device=device,
            dtype=ttnn.bfloat16,
        )
        self.chunk_eye = _as_device_tensor(
            torch.eye(chunk, dtype=torch.bfloat16).reshape(mask_shape), device=device, dtype=ttnn.bfloat16
        )
        self.chunk_ones_1x64 = _as_device_tensor(
            torch.ones((1, 1, 1, chunk), dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16
        )
        row_prefix_masks = []
        row_keep_masks = []
        for idx in range(chunk):
            prefix = torch.zeros(mask_shape, dtype=torch.bfloat16)
            prefix[..., idx, :idx] = 1
            keep = torch.ones(mask_shape, dtype=torch.bfloat16)
            keep[..., idx, :] = 0
            row_prefix_masks.append(_as_device_tensor(prefix, device=device, dtype=ttnn.bfloat16))
            row_keep_masks.append(_as_device_tensor(keep, device=device, dtype=ttnn.bfloat16))
        self.row_prefix_masks = tuple(row_prefix_masks)
        self.row_keep_masks = tuple(row_keep_masks)

    def _project_inputs(self, hidden_states: ttnn.Tensor):
        packed = ttnn.linear(
            hidden_states, self.in_proj_qkv_zba, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        mixed_qkv_raw = _slice_last(packed, 0, self.conv_dim)
        z = _slice_last(packed, self.conv_dim, self.conv_dim + self.value_dim)
        beta = _slice_last(
            packed, self.conv_dim + self.value_dim, self.conv_dim + self.value_dim + self.cfg.linear_num_value_heads
        )
        alpha = _slice_last(
            packed, self.conv_dim + self.value_dim + self.cfg.linear_num_value_heads, self.packed_qkv_zba_width
        )
        return mixed_qkv_raw, z, beta, alpha

    def _log_g(self, alpha: ttnn.Tensor) -> ttnn.Tensor:
        base = ttnn.add(alpha, self.dt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.mul(
            base,
            self.neg_exp_a_log,
            input_tensor_a_activations=_SOFTPLUS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _conv_step(self, mixed_qkv: ttnn.Tensor, state: QwenLinearAttentionState):
        next_conv_state = (state.conv_state[1], state.conv_state[2], state.conv_state[3], mixed_qkv)
        acc = ttnn.mul(next_conv_state[0], self.conv_weights[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for idx in range(1, self.cfg.linear_conv_kernel_dim):
            part = ttnn.mul(next_conv_state[idx], self.conv_weights[idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            acc = ttnn.add(acc, part, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG), next_conv_state

    def _step(self, hidden_states: ttnn.Tensor, state: QwenLinearAttentionState):
        _, _, batch, _ = _shape(hidden_states)
        mixed_qkv_raw, z, beta, alpha = self._project_inputs(hidden_states)
        mixed_qkv, conv_state = self._conv_step(mixed_qkv_raw, state)
        g = self._log_g(alpha)

        query = _slice_last(mixed_qkv, 0, self.key_dim)
        key = _slice_last(mixed_qkv, self.key_dim, 2 * self.key_dim)
        value = _slice_last(mixed_qkv, 2 * self.key_dim, self.conv_dim)

        query = ttnn.reshape(query, (1, batch, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim))
        key = ttnn.reshape(key, (1, batch, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim))
        if self.repeat_factor != 1:
            query = ttnn.repeat_interleave(query, self.repeat_factor, dim=2)
            key = ttnn.repeat_interleave(key, self.repeat_factor, dim=2)
        value = ttnn.reshape(value, (1, batch, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim))

        heads = batch * self.cfg.linear_num_value_heads
        query = ttnn.reshape(query, (1, heads, 1, self.cfg.linear_key_head_dim))
        key = ttnn.reshape(key, (1, heads, 1, self.cfg.linear_key_head_dim))
        value = ttnn.reshape(value, (1, heads, 1, self.cfg.linear_value_head_dim))
        query = _l2_norm_last_dim(query, self.cfg.linear_key_head_dim)
        key = _l2_norm_last_dim(key, self.cfg.linear_key_head_dim)
        query = ttnn.mul(query, self.cfg.linear_key_head_dim**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        beta = ttnn.reshape(beta, (1, heads, 1, 1))
        g = ttnn.reshape(g, (1, heads, 1, 1))
        g = ttnn.exp(g, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        decayed_state = ttnn.mul(state.recurrent_state, g, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kv_mem = ttnn.matmul(key, decayed_state, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        delta = ttnn.mul(
            ttnn.subtract(value, kv_mem, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            beta,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        key_t = ttnn.transpose(key, -2, -1)
        outer = ttnn.matmul(key_t, delta, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        recurrent_state = ttnn.add(decayed_state, outer, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        core = ttnn.matmul(query, recurrent_state, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        core = ttnn.reshape(core, (1, 1, heads, self.cfg.linear_value_head_dim))
        z = ttnn.reshape(z, (1, 1, heads, self.cfg.linear_value_head_dim))
        core = _rms_norm(core, self.norm_weight, self.cfg.rms_norm_eps)
        core = ttnn.mul(core, z, input_tensor_b_activations=_SILU, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        core = ttnn.reshape(core, (1, 1, batch, self.value_dim))
        out = ttnn.linear(core, self.out_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out, QwenLinearAttentionState(conv_state=conv_state, recurrent_state=recurrent_state)

    def _conv_prefill(self, mixed_qkv_raw: ttnn.Tensor, state: QwenLinearAttentionState):
        _, batch, length, _ = _shape(mixed_qkv_raw)
        taps = tuple(ttnn.reshape(tap, (1, batch, 1, self.conv_dim)) for tap in state.conv_state)
        conv_input = ttnn.concat(
            [taps[1], taps[2], taps[3], mixed_qkv_raw], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        acc = None
        for idx in range(self.cfg.linear_conv_kernel_dim):
            window = _slice(conv_input, (0, 0, idx, 0), (1, batch, idx + length, self.conv_dim))
            part = ttnn.mul(window, self.conv_weights[idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            acc = part if acc is None else ttnn.add(acc, part, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        history = ttnn.concat(
            [taps[0], taps[1], taps[2], taps[3], mixed_qkv_raw],
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hist_len = _shape(history)[2]
        last4 = _slice(
            history, (0, 0, hist_len - self.cfg.linear_conv_kernel_dim, 0), (1, batch, hist_len, self.conv_dim)
        )
        conv_state = tuple(
            ttnn.reshape(
                _slice(last4, (0, 0, idx, 0), (1, batch, idx + 1, self.conv_dim)),
                (1, 1, batch, self.conv_dim),
            )
            for idx in range(self.cfg.linear_conv_kernel_dim)
        )
        return ttnn.silu(acc, memory_config=ttnn.DRAM_MEMORY_CONFIG), conv_state

    def _reshape_prefill_heads(
        self,
        mixed_qkv: ttnn.Tensor,
        z: ttnn.Tensor,
        beta: ttnn.Tensor,
        log_g: ttnn.Tensor,
        batch: int,
        length: int,
    ):
        query = _slice_last(mixed_qkv, 0, self.key_dim)
        key = _slice_last(mixed_qkv, self.key_dim, 2 * self.key_dim)
        value = _slice_last(mixed_qkv, 2 * self.key_dim, self.conv_dim)

        query = ttnn.reshape(query, (batch, length, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim))
        key = ttnn.reshape(key, (batch, length, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim))
        if self.repeat_factor != 1:
            query = ttnn.repeat_interleave(query, self.repeat_factor, dim=2)
            key = ttnn.repeat_interleave(key, self.repeat_factor, dim=2)

        value = ttnn.reshape(value, (batch, length, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim))
        z = ttnn.reshape(z, (batch, length, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim))
        beta = ttnn.reshape(beta, (batch, length, self.cfg.linear_num_value_heads, 1))
        log_g = ttnn.reshape(log_g, (batch, length, self.cfg.linear_num_value_heads, 1))

        query = self._fold_prefill_heads(query, batch, length, self.cfg.linear_key_head_dim)
        key = self._fold_prefill_heads(key, batch, length, self.cfg.linear_key_head_dim)
        value = self._fold_prefill_heads(value, batch, length, self.cfg.linear_value_head_dim)
        z = self._fold_prefill_heads(z, batch, length, self.cfg.linear_value_head_dim)
        beta = self._fold_prefill_heads(beta, batch, length, 1)
        log_g = self._fold_prefill_heads(log_g, batch, length, 1)

        query = _l2_norm_last_dim(query, self.cfg.linear_key_head_dim)
        key = _l2_norm_last_dim(key, self.cfg.linear_key_head_dim)
        query = ttnn.mul(query, self.cfg.linear_key_head_dim**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return query, key, value, z, beta, log_g

    def _fold_prefill_heads(self, tensor: ttnn.Tensor, batch: int, length: int, head_dim: int) -> ttnn.Tensor:
        tensor = ttnn.permute(tensor, (0, 2, 1, 3))
        return ttnn.reshape(tensor, (1, batch * self.cfg.linear_num_value_heads, length, head_dim))

    def _pad_linear_chunk(self, tensor: ttnn.Tensor, length: int, last_dim: int) -> ttnn.Tensor:
        if length == self.linear_chunk_size:
            return tensor
        shape = list(_shape(tensor))
        shape[2] = self.linear_chunk_size
        shape[3] = last_dim
        return ttnn.pad(tensor, tuple(shape), (0, 0, 0, 0), 0.0)

    def _solve_chunk_attn(self, attn0: ttnn.Tensor) -> ttnn.Tensor:
        solved = attn0
        for idx in range(1, self.linear_chunk_size):
            row = ttnn.mul(solved, self.row_prefix_masks[idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            update = ttnn.matmul(row, solved, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            new_row = ttnn.add(row, update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kept = ttnn.mul(solved, self.row_keep_masks[idx], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            solved = ttnn.add(kept, new_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(solved, self.chunk_eye, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _chunk_gated_delta_rule(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        log_g: ttnn.Tensor,
        beta: ttnn.Tensor,
        recurrent_state: ttnn.Tensor,
    ):
        log_g = ttnn.cumsum(log_g, dim=2, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g_rows = ttnn.matmul(log_g, self.chunk_ones_1x64, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        g_cols = ttnn.transpose(g_rows, -2, -1)
        decay = ttnn.subtract(g_rows, g_cols, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        decay = ttnn.exp(decay, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        decay = ttnn.mul(decay, self.chunk_lower_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        k_beta = ttnn.mul(
            key,
            beta,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_beta = ttnn.mul(
            value,
            beta,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kk = ttnn.matmul(
            k_beta, ttnn.transpose(key, -2, -1), memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        attn0 = ttnn.mul(kk, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn0 = ttnn.mul(
            ttnn.mul(attn0, decay, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            self.chunk_strict_lower_mask,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local_attn = self._solve_chunk_attn(attn0)

        local_value = ttnn.matmul(local_attn, v_beta, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        exp_g = ttnn.exp(log_g, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_cumdecay = ttnn.matmul(
            local_attn,
            ttnn.mul(k_beta, exp_g, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        v_prime = ttnn.matmul(k_cumdecay, recurrent_state, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        v_new = ttnn.subtract(local_value, v_prime, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        content_attn = ttnn.matmul(
            query, ttnn.transpose(key, -2, -1), memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        content_attn = ttnn.mul(content_attn, decay, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_inter = ttnn.matmul(
            ttnn.mul(query, exp_g, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            recurrent_state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        core = ttnn.add(
            attn_inter,
            ttnn.matmul(content_attn, v_new, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        last_idx = self.linear_chunk_size - 1
        g_last = _slice(log_g, (0, 0, last_idx, 0), (1, _shape(log_g)[1], last_idx + 1, 1))
        state_decay = ttnn.mul(
            recurrent_state,
            ttnn.exp(g_last, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state_scale = ttnn.exp(
            ttnn.subtract(g_last, log_g, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            fast_and_approximate_mode=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state_update_key = ttnn.mul(key, state_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        state_update = ttnn.matmul(
            ttnn.transpose(state_update_key, -2, -1), v_new, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        recurrent_state = ttnn.add(state_decay, state_update, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return core, recurrent_state

    def _finish_prefill_chunk(self, core: ttnn.Tensor, z: ttnn.Tensor, batch: int, length: int) -> ttnn.Tensor:
        core = _rms_norm(core, self.norm_weight, self.cfg.rms_norm_eps)
        core = ttnn.mul(core, z, input_tensor_b_activations=_SILU, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        core = ttnn.reshape(core, (batch, self.cfg.linear_num_value_heads, length, self.cfg.linear_value_head_dim))
        core = ttnn.permute(core, (0, 2, 1, 3))
        core = ttnn.reshape(core, (1, batch, length, self.value_dim))
        return ttnn.linear(core, self.out_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(self, hidden_states: ttnn.Tensor, *, linear_state: QwenLinearAttentionState):
        return self._step(hidden_states, linear_state)

    def prefill_forward(self, hidden_states: ttnn.Tensor, *, linear_state: QwenLinearAttentionState):
        _, batch, seq_len, _ = _shape(hidden_states)
        chunks = []
        next_state = linear_state
        for start in range(0, seq_len, self.linear_chunk_size):
            end = min(start + self.linear_chunk_size, seq_len)
            length = end - start
            hidden_chunk = _slice(hidden_states, (0, 0, start, 0), (1, batch, end, self.cfg.hidden_size))
            mixed_qkv_raw, z, beta, alpha = self._project_inputs(hidden_chunk)
            mixed_qkv, conv_state = self._conv_prefill(mixed_qkv_raw, next_state)
            log_g = self._log_g(alpha)

            query, key, value, z_heads, beta, log_g = self._reshape_prefill_heads(
                mixed_qkv, z, beta, log_g, batch, length
            )
            query = self._pad_linear_chunk(query, length, self.cfg.linear_key_head_dim)
            key = self._pad_linear_chunk(key, length, self.cfg.linear_key_head_dim)
            value = self._pad_linear_chunk(value, length, self.cfg.linear_value_head_dim)
            z_heads = self._pad_linear_chunk(z_heads, length, self.cfg.linear_value_head_dim)
            beta = self._pad_linear_chunk(beta, length, 1)
            log_g = self._pad_linear_chunk(log_g, length, 1)

            core, recurrent_state = self._chunk_gated_delta_rule(
                query, key, value, log_g, beta, next_state.recurrent_state
            )
            if length != self.linear_chunk_size:
                core = _slice(
                    core,
                    (0, 0, 0, 0),
                    (1, batch * self.cfg.linear_num_value_heads, length, self.cfg.linear_value_head_dim),
                )
                z_heads = _slice(
                    z_heads,
                    (0, 0, 0, 0),
                    (1, batch * self.cfg.linear_num_value_heads, length, self.cfg.linear_value_head_dim),
                )
            chunks.append(self._finish_prefill_chunk(core, z_heads, batch, length))
            next_state = QwenLinearAttentionState(conv_state=conv_state, recurrent_state=recurrent_state)
        return _concat_dim2_bounded(chunks), next_state


class _FusedQwenMoe(LightweightModule):
    def __init__(self, state: dict[str, Any], cfg: _QwenTextConfig, *, device, dtype, chunk_size: int):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.chunk_size = chunk_size
        self.router = _packed_linear_weight(state, ("mlp.gate.weight",), device=device, dtype=ttnn.bfloat16)

        self.shared_gate_up = _packed_linear_weight(
            state,
            ("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight"),
            device=device,
            dtype=dtype,
        )
        self.shared_down = _packed_linear_weight(
            state, ("mlp.shared_expert.down_proj.weight",), device=device, dtype=dtype
        )
        self.shared_expert_gate = _packed_linear_weight(
            state, ("mlp.shared_expert_gate.weight",), device=device, dtype=ttnn.bfloat16
        )

        self.routed_gate_up = _packed_gate_up_expert_weight(
            state, "mlp.experts.gate_up_proj", device=device, dtype=dtype
        )
        down = _require(state, "mlp.experts.down_proj")
        self.routed_down = _as_device_tensor(
            down.transpose(-1, -2).unsqueeze(0).contiguous(), device=device, dtype=dtype
        )

        import torch

        self.all_expert_sparsity = _as_device_tensor(
            torch.ones((1, 1, 1, cfg.num_experts), dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    @property
    def gate_up_width(self) -> int:
        return 2 * self.cfg.moe_intermediate_size

    @property
    def shared_gate_up_width(self) -> int:
        return 2 * self.cfg.shared_expert_intermediate_size

    def _router_dense(self, flat: ttnn.Tensor) -> ttnn.Tensor:
        logits = ttnn.linear(flat, self.router, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        probs = ttnn.softmax(logits, dim=-1, numeric_stable=True)
        top_values, top_indices = ttnn.topk(probs, k=self.cfg.num_experts_per_tok, dim=-1, sorted=True)
        denom = ttnn.sum(top_values, dim=-1, keepdim=True)
        top_values = ttnn.div(top_values, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.scatter(ttnn.zeros_like(probs), dim=-1, index=top_indices, src=top_values)

    def _shared(self, flat: ttnn.Tensor) -> ttnn.Tensor:
        gate_up = ttnn.linear(flat, self.shared_gate_up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = _slice_last(gate_up, 0, self.cfg.shared_expert_intermediate_size)
        up = _slice_last(gate_up, self.cfg.shared_expert_intermediate_size, self.shared_gate_up_width)
        hidden = _silu_mul_fused(gate, up)
        hidden = ttnn.linear(hidden, self.shared_down, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_scalar = ttnn.linear(
            flat, self.shared_expert_gate, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.mul(
            hidden,
            gate_scalar,
            input_tensor_b_activations=_SIGMOID,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _routed_decode(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        tokens = _shape(flat)[2]
        sparse_routing = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT)
        gate_up_config = _sparse_matmul_program_config(tokens, self.gate_up_width)
        down_config = _sparse_matmul_program_config(tokens, self.cfg.hidden_size)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])

        gate_up = ttnn.sparse_matmul(
            flat,
            self.routed_gate_up,
            sparsity=sparse_routing,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=ttnn.bfloat16,
        )
        gate_up = ttnn.reshape(gate_up, (tokens, self.cfg.num_experts, 1, self.gate_up_width))
        gate_up = ttnn.transpose(gate_up, 1, 2)
        gate = _slice_last(gate_up, 0, self.cfg.moe_intermediate_size)
        up = _slice_last(gate_up, self.cfg.moe_intermediate_size, self.gate_up_width)

        expert_hidden = _silu_mul_fused(gate, up)
        expert_hidden = ttnn.reshape(expert_hidden, (tokens, self.cfg.num_experts, self.cfg.moe_intermediate_size))
        expert_hidden = ttnn.transpose(expert_hidden, 1, 0)
        expert_hidden = ttnn.reshape(
            expert_hidden,
            (1, self.cfg.num_experts, tokens, self.cfg.moe_intermediate_size),
        )
        routed = ttnn.sparse_matmul(
            expert_hidden,
            self.routed_down,
            sparsity=sparse_routing,
            nnz=None,
            is_input_a_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            dtype=ttnn.bfloat16,
        )
        routed = ttnn.permute(routed, (0, 2, 1, 3))
        routed = ttnn.reshape(routed, (tokens, self.cfg.num_experts, self.cfg.hidden_size))
        routing = ttnn.reshape(routing, (tokens, self.cfg.num_experts, 1))
        routed = ttnn.mul(routed, routing, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routed = ttnn.sum(routed, dim=1)
        return ttnn.unsqueeze_to_4D(routed)

    def _routed_prefill_chunk(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        logical_tokens = _shape(flat)[2]
        hidden = _shape(flat)[3]
        physical_tokens = int(math.ceil(logical_tokens / TILE_SIZE) * TILE_SIZE)
        if physical_tokens != logical_tokens:
            flat = ttnn.pad(flat, (1, 1, physical_tokens, hidden), (0, 0, 0, 0), 0.0)
            routing = ttnn.pad(routing, (1, 1, physical_tokens, self.cfg.num_experts), (0, 0, 0, 0), 0.0)

        group_size = physical_tokens // TILE_SIZE
        grouped = ttnn.reshape(flat, (1, group_size, TILE_SIZE, hidden))
        sparsity = ttnn.repeat(self.all_expert_sparsity, (1, 1, group_size, 1))
        nnz = self.cfg.num_experts * group_size
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _sparse_matmul_program_config(TILE_SIZE, self.gate_up_width)
        down_config = _sparse_matmul_program_config(TILE_SIZE, self.cfg.hidden_size)

        gate_up = ttnn.sparse_matmul(
            grouped,
            self.routed_gate_up,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=ttnn.bfloat16,
        )
        gate_up = ttnn.transpose(gate_up, 1, 3)
        gate_up = ttnn.reshape(gate_up, (1, self.cfg.num_experts, physical_tokens, self.gate_up_width))
        gate = _slice_last(gate_up, 0, self.cfg.moe_intermediate_size)
        up = _slice_last(gate_up, self.cfg.moe_intermediate_size, self.gate_up_width)

        expert_hidden = _silu_mul_fused(gate, up)
        expert_hidden = ttnn.reshape(
            expert_hidden, (1, self.cfg.num_experts, physical_tokens, self.cfg.moe_intermediate_size)
        )
        routed = ttnn.sparse_matmul(
            expert_hidden,
            self.routed_down,
            sparsity=self.all_expert_sparsity,
            nnz=self.cfg.num_experts,
            is_input_a_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            dtype=ttnn.bfloat16,
        )
        routed = ttnn.reshape(routed, (1, self.cfg.num_experts, physical_tokens, self.cfg.hidden_size))
        routing = ttnn.permute(routing, (0, 3, 2, 1))
        routed = ttnn.mul(routed, routing, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routed = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(routed, dims=[1]))
        routed = ttnn.reshape(routed, (1, 1, physical_tokens, self.cfg.hidden_size))
        if physical_tokens != logical_tokens:
            routed = _slice(routed, (0, 0, 0, 0), (1, 1, logical_tokens, self.cfg.hidden_size))
        return routed

    def _routed_chunk(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        if _shape(flat)[2] == 1:
            return self._routed_decode(flat, routing)
        return self._routed_prefill_chunk(flat, routing)

    def _forward_chunk(self, flat: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        routed = self._routed_chunk(flat, routing)
        shared = self._shared(flat)
        return ttnn.add(routed, shared, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        _, batch, seq_len, hidden = _shape(hidden_states)
        tokens = batch * seq_len
        flat = ttnn.reshape(hidden_states, (1, 1, tokens, hidden))
        routing = self._router_dense(flat)
        if tokens <= self.chunk_size:
            out = self._forward_chunk(flat, routing)
        else:
            outputs = []
            for start in range(0, tokens, self.chunk_size):
                end = min(start + self.chunk_size, tokens)
                flat_chunk = _slice(flat, (0, 0, start, 0), (1, 1, end, hidden))
                routing_chunk = _slice(routing, (0, 0, start, 0), (1, 1, end, self.cfg.num_experts))
                outputs.append(self._forward_chunk(flat_chunk, routing_chunk))
            out = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (1, batch, seq_len, hidden))


class FusedDecoder(LightweightModule):
    """Qwen3.6-35B-A3B fused decoder layer with the functional public contract."""

    graph_summary = GRAPH_SUMMARY

    def __init__(
        self,
        *,
        cfg: _QwenTextConfig,
        layer_idx: int,
        layer_type: Literal["linear_attention", "full_attention"],
        input_layernorm_weight: ttnn.Tensor,
        post_attention_layernorm_weight: ttnn.Tensor,
        token_mixer: _FusedLinearAttention | _FusedFullAttention,
        mlp: _FusedQwenMoe,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.token_mixer = token_mixer
        self.mlp = mlp

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        cfg = _text_config(hf_config)
        if cfg.hidden_size != HIDDEN_SIZE:
            raise ValueError(f"{MODEL_ID} fused decoder expects hidden_size={HIDDEN_SIZE}, got {cfg.hidden_size}")
        layer_type = cfg.layer_types[layer_idx]
        if layer_type not in ("linear_attention", "full_attention"):
            raise ValueError(f"unsupported Qwen3.6 decoder layer_type: {layer_type}")

        dtype = kwargs.get("weight_dtype", ttnn.bfloat16)
        state = _layer_state(state_dict, layer_idx)
        input_norm = _rms_weight(state, "input_layernorm.weight", device=mesh_device, add_unit_offset=True)
        post_norm = _rms_weight(state, "post_attention_layernorm.weight", device=mesh_device, add_unit_offset=True)
        if layer_type == "linear_attention":
            token_mixer = _FusedLinearAttention(state, cfg, device=mesh_device, dtype=dtype)
        else:
            token_mixer = _FusedFullAttention(state, cfg, device=mesh_device, dtype=dtype)
        moe_chunk_size = int(kwargs.get("moe_chunk_size", DEFAULT_MOE_CHUNK_SIZE))
        if moe_chunk_size <= 0:
            raise ValueError(f"moe_chunk_size must be positive, got {moe_chunk_size}")
        mlp = _FusedQwenMoe(state, cfg, device=mesh_device, dtype=dtype, chunk_size=moe_chunk_size)
        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            layer_type=layer_type,
            input_layernorm_weight=input_norm,
            post_attention_layernorm_weight=post_norm,
            token_mixer=token_mixer,
            mlp=mlp,
        )

    @classmethod
    def allocate_full_attention_cache(
        cls,
        *,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int | None = None,
        block_size: int = 32,
        dtype=ttnn.bfloat16,
    ) -> QwenFullAttentionCache:
        cfg = _text_config(hf_config)
        if max_seq_len is None:
            max_seq_len = cfg.max_position_embeddings
        max_blocks_per_seq = math.ceil(max_seq_len / block_size)
        max_num_blocks = max_batch_size * max_blocks_per_seq

        import torch

        cache_shape = (max_num_blocks, cfg.num_key_value_heads, block_size, cfg.head_dim)
        keys = _as_device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device=mesh_device, dtype=dtype)
        values = _as_device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device=mesh_device, dtype=dtype)
        return QwenFullAttentionCache(
            keys=keys,
            values=values,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def allocate_linear_attention_state(
        cls,
        *,
        hf_config,
        mesh_device,
        batch_size: int,
        dtype=ttnn.bfloat16,
    ) -> QwenLinearAttentionState:
        cfg = _text_config(hf_config)
        conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
        conv_dim += cfg.linear_value_head_dim * cfg.linear_num_value_heads
        heads = batch_size * cfg.linear_num_value_heads

        import torch

        conv_state = tuple(
            _as_device_tensor(
                torch.zeros((1, 1, batch_size, conv_dim), dtype=torch.bfloat16), device=mesh_device, dtype=dtype
            )
            for _ in range(cfg.linear_conv_kernel_dim)
        )
        recurrent_state = _as_device_tensor(
            torch.zeros((1, heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim), dtype=torch.bfloat16),
            device=mesh_device,
            dtype=dtype,
        )
        return QwenLinearAttentionState(conv_state=conv_state, recurrent_state=recurrent_state)

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_embeddings: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        page_table: ttnn.Tensor | None = None,
        kv_cache: QwenFullAttentionCache | None = None,
        linear_state: QwenLinearAttentionState | None = None,
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        attention_mask: ttnn.Tensor | None = None,
    ) -> FunctionalDecoderResult:
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.input_layernorm_weight, self.cfg.rms_norm_eps)
        if self.layer_type == "full_attention":
            if position_embeddings is None:
                raise ValueError("full_attention prefill requires position_embeddings")
            mixer_out = self.token_mixer.prefill_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache,
                page_table=page_table,
                user_id=user_id,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                attention_mask=attention_mask,
            )
            next_linear_state = None
        else:
            if linear_state is None:
                raise ValueError("linear_attention prefill requires linear_state")
            mixer_out, next_linear_state = self.token_mixer.prefill_forward(hidden_states, linear_state=linear_state)

        hidden_states = ttnn.add(residual, mixer_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.post_attention_layernorm_weight, self.cfg.rms_norm_eps)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return FunctionalDecoderResult(hidden_states=hidden_states, kv_cache=kv_cache, linear_state=next_linear_state)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        position_embeddings: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        page_table: ttnn.Tensor | None = None,
        kv_cache: QwenFullAttentionCache | None = None,
        linear_state: QwenLinearAttentionState | None = None,
    ) -> FunctionalDecoderResult:
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.input_layernorm_weight, self.cfg.rms_norm_eps)
        if self.layer_type == "full_attention":
            if position_embeddings is None or kv_cache is None or page_table is None:
                raise ValueError("full_attention decode requires position_embeddings, kv_cache, and page_table")
            mixer_out = self.token_mixer.decode_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache,
                page_table=page_table,
                current_pos=current_pos,
            )
            next_linear_state = None
        else:
            if linear_state is None:
                raise ValueError("linear_attention decode requires linear_state")
            mixer_out, next_linear_state = self.token_mixer.decode_forward(hidden_states, linear_state=linear_state)

        hidden_states = ttnn.add(residual, mixer_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.post_attention_layernorm_weight, self.cfg.rms_norm_eps)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return FunctionalDecoderResult(hidden_states=hidden_states, kv_cache=kv_cache, linear_state=next_linear_state)

    def forward(
        self, hidden_states: ttnn.Tensor, *, mode: Literal["prefill", "decode"], **kwargs
    ) -> FunctionalDecoderResult:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")


__all__ = [
    "FusedDecoder",
    "FusedDecoderGraphSummary",
    "GRAPH_SUMMARY",
    "FunctionalDecoderResult",
    "QwenFullAttentionCache",
    "QwenLinearAttentionState",
]
