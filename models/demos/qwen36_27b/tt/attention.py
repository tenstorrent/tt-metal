# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Gated GQA Attention for Qwen3.6-27B.

Q projection outputs 2× head_dim, split into query + sigmoid gate.
Q/K have per-head RMSNorm. Partial RoPE (25% of head_dim).
Output = o_proj(attn_output * sigmoid(gate))

Two forward paths:
  - decode: S=1, all device ops, KV cache on device
  - prefill: S>1, CPU fallback for correctness
"""

import math
import os

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

# Use the fully on-device decode path (trace-friendly, no host round-trips).
ONDEVICE_ATTN = os.environ.get("QWEN_ONDEVICE_ATTN", "0") == "1"


class TtGatedAttention(LightweightModule):
    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16, weights_dtype=None):
        super().__init__()
        self.device = device
        if weights_dtype is None:
            weights_dtype = getattr(config, "weights_dtype", ttnn.bfloat8_b)
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.rotary_dim = config.rotary_dim
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_len

        prefix = f"model.layers.{layer_idx}.self_attn"

        q_w = state_dict[f"{prefix}.q_proj.weight"].T.contiguous()
        k_w = state_dict[f"{prefix}.k_proj.weight"].T.contiguous()
        v_w = state_dict[f"{prefix}.v_proj.weight"].T.contiguous()
        o_w = state_dict[f"{prefix}.o_proj.weight"].T.contiguous()

        self.q_proj_w = ttnn.from_torch(
            q_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.k_proj_w = ttnn.from_torch(
            k_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.v_proj_w = ttnn.from_torch(
            v_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.o_proj_w = ttnn.from_torch(
            o_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device,
        )

        self.q_norm_w_cpu = state_dict[f"{prefix}.q_norm.weight"][:self.head_dim].float()
        self.k_norm_w_cpu = state_dict[f"{prefix}.k_norm.weight"][:self.head_dim].float()

        TILE = 32
        q_norm_w = (self.q_norm_w_cpu + 1.0).unsqueeze(0).view(1, 1, self.head_dim).reshape(1, 1, self.head_dim // TILE, TILE)
        k_norm_w = (self.k_norm_w_cpu + 1.0).unsqueeze(0).view(1, 1, self.head_dim).reshape(1, 1, self.head_dim // TILE, TILE)
        self.q_norm_w_tt = ttnn.from_torch(q_norm_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self.k_norm_w_tt = ttnn.from_torch(k_norm_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def forward(self, hidden_states, cos, sin, kv_cache=None, mode="decode"):
        if mode == "decode":
            if getattr(self, "trace_decode", False):
                # trace path: fixed preallocated KV cache + device cur_pos; cos/sin are device buffers
                k_cache, v_cache = kv_cache
                out = self._decode_ondevice_cached(
                    hidden_states, cos, sin, k_cache, v_cache, self.cur_pos_tt
                )
                return out, None  # cache updated in place
            if ONDEVICE_ATTN:
                past_k, past_v = kv_cache if kv_cache is not None else (None, None)
                return self._decode_ondevice(hidden_states, cos, sin, past_k, past_v)
            return self._decode(hidden_states, cos, sin, kv_cache)
        return self._prefill(hidden_states, cos, sin, kv_cache)

    def _decode(self, hidden_states, cos, sin, kv_cache):
        """All-device decode path for S=1."""
        q_proj = ttnn.linear(hidden_states, self.q_proj_w)
        k_proj = ttnn.linear(hidden_states, self.k_proj_w)
        v_proj = ttnn.linear(hidden_states, self.v_proj_w)

        # Q: [1,1,1,num_heads*head_dim*2] → split into query and gate
        # Reshape to [1,1,num_heads, head_dim*2] then split
        q_2d = ttnn.reshape(q_proj, [1, 1, self.num_heads, self.head_dim * 2])
        query = ttnn.slice(q_2d, [0, 0, 0, 0], [1, 1, self.num_heads, self.head_dim])
        gate = ttnn.slice(q_2d, [0, 0, 0, self.head_dim], [1, 1, self.num_heads, self.head_dim * 2])
        ttnn.deallocate(q_2d)

        # K: [1,1,1,num_kv_heads*head_dim] → [1,1,num_kv_heads,head_dim]
        key = ttnn.reshape(k_proj, [1, 1, self.num_kv_heads, self.head_dim])

        # V: [1,1,1,num_kv_heads*head_dim] → [1,1,num_kv_heads,head_dim]
        value = ttnn.reshape(v_proj, [1, 1, self.num_kv_heads, self.head_dim])

        # Per-head RMSNorm on Q and K
        # query: [1,1,num_heads,head_dim], need norm across last dim per head
        query = ttnn.rms_norm(query, epsilon=1e-6, weight=self.q_norm_w_tt)
        key = ttnn.rms_norm(key, epsilon=1e-6, weight=self.k_norm_w_tt)

        # Partial RoPE: only first rotary_dim elements get rotated
        # For decode, cos/sin are [1,1,1,rotary_dim] CPU tensors for position p
        cos_cpu = cos if isinstance(cos, torch.Tensor) else ttnn.to_torch(cos)
        sin_cpu = sin if isinstance(sin, torch.Tensor) else ttnn.to_torch(sin)

        # Pull Q/K to CPU for RoPE + attention (RoPE partial is hard to do efficiently on device)
        # This is a temporary approach — still faster than full CPU attention because we avoid
        # q/k/v projection transfers (which are the bulk of the data)
        query_cpu = ttnn.to_torch(query).reshape(1, self.num_heads, 1, self.head_dim)
        key_cpu = ttnn.to_torch(key).reshape(1, self.num_kv_heads, 1, self.head_dim)
        value_cpu = ttnn.to_torch(value).reshape(1, self.num_kv_heads, 1, self.head_dim)
        gate_cpu = ttnn.to_torch(gate).reshape(1, 1, self.num_heads * self.head_dim)

        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)
        ttnn.deallocate(gate)

        query_cpu, key_cpu = self._apply_partial_rotary(query_cpu, key_cpu, cos_cpu, sin_cpu)

        # KV cache update
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            key_cpu = torch.cat([k_cache, key_cpu], dim=2)
            value_cpu = torch.cat([v_cache, value_cpu], dim=2)
        new_kv_cache = (key_cpu.detach(), value_cpu.detach())

        # GQA: expand KV heads
        if self.num_kv_groups > 1:
            key_exp = key_cpu.repeat_interleave(self.num_kv_groups, dim=1)
            value_exp = value_cpu.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            key_exp = key_cpu
            value_exp = value_cpu

        # Attention: Q @ K^T * scale → softmax → @ V
        attn_weights = torch.matmul(query_cpu.float(), key_exp.float().transpose(-1, -2)) * self.scaling
        seq_len = key_exp.shape[2]
        causal_mask = torch.triu(
            torch.full((1, seq_len), float("-inf")),
            diagonal=seq_len,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1).to(query_cpu.dtype)
        attn_output = torch.matmul(attn_weights, value_exp.float()).to(query_cpu.dtype)

        attn_output = attn_output.transpose(1, 2).reshape(1, 1, -1)
        attn_output = attn_output * torch.sigmoid(gate_cpu.float()).to(attn_output.dtype)

        attn_output_tt = ttnn.from_torch(
            attn_output.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        output = ttnn.linear(attn_output_tt, self.o_proj_w)
        return output, new_kv_cache

    # ------------------------------------------------------------------
    # On-device decode (trace-friendly: no host round-trips)
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_half_tt(x):
        d = int(x.shape[-1])
        b, s, h = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        x1 = ttnn.slice(x, [0, 0, 0, 0], [b, s, h, d // 2])
        x2 = ttnn.slice(x, [0, 0, 0, d // 2], [b, s, h, d])
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def _apply_partial_rope_tt(self, x, cos_tt, sin_tt):
        """Rotate only the first rotary_dim of the last (head) dim. cos/sin: [1,1,1,rotary_dim]."""
        d = self.rotary_dim
        b, s, h = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        x_rot = ttnn.slice(x, [0, 0, 0, 0], [b, s, h, d])
        x_pass = ttnn.slice(x, [0, 0, 0, d], [b, s, h, self.head_dim])
        rotated = ttnn.add(ttnn.mul(x_rot, cos_tt), ttnn.mul(self._rotate_half_tt(x_rot), sin_tt))
        return ttnn.concat([rotated, x_pass], dim=-1)

    def _decode_ondevice(self, hidden_states, cos_tt, sin_tt, past_k=None, past_v=None):
        """
        Fully on-device single-token decode. No to_torch/from_torch — trace-capturable.

        hidden_states: [1, 1, 1, hidden]
        cos_tt, sin_tt: [1, 1, 1, rotary_dim] device tensors for the current position
        past_k, past_v: device tensors [1, num_kv_heads, past_len, head_dim] or None
        Returns (output [1,1,1,hidden], (new_k, new_v)) where new_k/v include the current token.
        """
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        # Accept torch cos/sin (from model.get_rope) and torch past KV (from the CPU
        # prefill path on the first decode step); move them to device once.
        def _to_dev(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if isinstance(cos_tt, torch.Tensor):
            cos_tt = _to_dev(cos_tt)
        if isinstance(sin_tt, torch.Tensor):
            sin_tt = _to_dev(sin_tt)
        if past_k is not None and isinstance(past_k, torch.Tensor):
            past_k = _to_dev(past_k)
            past_v = _to_dev(past_v)

        q_proj = ttnn.linear(hidden_states, self.q_proj_w)
        k_proj = ttnn.linear(hidden_states, self.k_proj_w)
        v_proj = ttnn.linear(hidden_states, self.v_proj_w)

        q2 = ttnn.reshape(q_proj, [1, 1, nh, hd * 2])
        query = ttnn.slice(q2, [0, 0, 0, 0], [1, 1, nh, hd])
        gate = ttnn.slice(q2, [0, 0, 0, hd], [1, 1, nh, hd * 2])
        key = ttnn.reshape(k_proj, [1, 1, nkv, hd])
        value = ttnn.reshape(v_proj, [1, 1, nkv, hd])

        query = ttnn.rms_norm(query, epsilon=1e-6, weight=self.q_norm_w_tt)
        key = ttnn.rms_norm(key, epsilon=1e-6, weight=self.k_norm_w_tt)

        query = self._apply_partial_rope_tt(query, cos_tt, sin_tt)
        key = self._apply_partial_rope_tt(key, cos_tt, sin_tt)

        # [1,1,heads,hd] -> [1,heads,1,hd]
        query = ttnn.permute(query, [0, 2, 1, 3])
        key = ttnn.permute(key, [0, 2, 1, 3])
        value = ttnn.permute(value, [0, 2, 1, 3])

        if past_k is not None:
            key = ttnn.concat([past_k, key], dim=2)
            value = ttnn.concat([past_v, value], dim=2)
        new_k, new_v = key, value

        if self.num_kv_groups > 1:
            key = ttnn.repeat_interleave(key, self.num_kv_groups, dim=1)
            value = ttnn.repeat_interleave(value, self.num_kv_groups, dim=1)

        attn = ttnn.transformer.scaled_dot_product_attention(
            query, key, value, is_causal=False, scale=self.scaling
        )  # [1, nh, 1, hd]

        attn = ttnn.permute(attn, [0, 2, 1, 3])  # [1,1,nh,hd]
        attn = ttnn.reshape(attn, [1, 1, 1, nh * hd])
        gate = ttnn.reshape(gate, [1, 1, 1, nh * hd])
        attn = ttnn.mul(attn, ttnn.sigmoid(gate))
        output = ttnn.linear(attn, self.o_proj_w)
        return output, (new_k, new_v)

    def _decode_ondevice_cached(self, hidden_states, cos_tt, sin_tt, k_cache, v_cache, cur_pos_tt):
        """
        Trace-friendly on-device decode: fixed-shape preallocated KV cache updated in
        place via paged_update_cache, attention via scaled_dot_product_attention_decode.
        No growing tensors / concat — capturable in a decode trace.

        hidden_states: [1,1,1,hidden]; cos_tt/sin_tt: [1,1,1,rotary_dim] (device)
        k_cache/v_cache: [1, num_kv_heads, max_seq, head_dim] (device, preallocated)
        cur_pos_tt: device int32 tensor [1] = current position to write/attend up to.
        """
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        q_proj = ttnn.linear(hidden_states, self.q_proj_w)
        k_proj = ttnn.linear(hidden_states, self.k_proj_w)
        v_proj = ttnn.linear(hidden_states, self.v_proj_w)

        q2 = ttnn.reshape(q_proj, [1, 1, nh, hd * 2])
        query = ttnn.slice(q2, [0, 0, 0, 0], [1, 1, nh, hd])
        gate = ttnn.slice(q2, [0, 0, 0, hd], [1, 1, nh, hd * 2])
        key = ttnn.reshape(k_proj, [1, 1, nkv, hd])
        value = ttnn.reshape(v_proj, [1, 1, nkv, hd])

        query = ttnn.rms_norm(query, epsilon=1e-6, weight=self.q_norm_w_tt)
        key = ttnn.rms_norm(key, epsilon=1e-6, weight=self.k_norm_w_tt)
        query = self._apply_partial_rope_tt(query, cos_tt, sin_tt)
        key = self._apply_partial_rope_tt(key, cos_tt, sin_tt)

        # write current K/V into the cache at cur_pos (input layout [1, B, n_kv, hd]).
        # paged_update_cache requires the *input* to be height-sharded (shard width ==
        # head_dim). Shard k/v onto a single core before the in-place update.
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(32, hd),
            core_grid=ttnn.CoreGrid(y=1, x=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        key_s = ttnn.to_memory_config(key, shard_cfg)
        value_s = ttnn.to_memory_config(value, shard_cfg)
        ttnn.experimental.paged_update_cache(k_cache, key_s, update_idxs_tensor=cur_pos_tt)
        ttnn.experimental.paged_update_cache(v_cache, value_s, update_idxs_tensor=cur_pos_tt)

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            query, k_cache, v_cache, cur_pos_tensor=cur_pos_tt, scale=self.scaling
        )  # [1, 1, nh, hd]

        attn = ttnn.reshape(attn, [1, 1, 1, nh * hd])
        gate = ttnn.reshape(gate, [1, 1, 1, nh * hd])
        attn = ttnn.mul(attn, ttnn.sigmoid(gate))
        return ttnn.linear(attn, self.o_proj_w)

    def _prefill(self, hidden_states, cos, sin, kv_cache):
        """CPU fallback for prefill (S > 1)."""
        B = 1
        S = hidden_states.shape[2]

        cos_cpu = cos if isinstance(cos, torch.Tensor) else ttnn.to_torch(cos)
        sin_cpu = sin if isinstance(sin, torch.Tensor) else ttnn.to_torch(sin)

        q_proj_cpu = ttnn.to_torch(ttnn.linear(hidden_states, self.q_proj_w)).reshape(B, S, -1)
        k_proj_cpu = ttnn.to_torch(ttnn.linear(hidden_states, self.k_proj_w)).reshape(B, S, -1)
        v_proj_cpu = ttnn.to_torch(ttnn.linear(hidden_states, self.v_proj_w)).reshape(B, S, -1)

        q_gate = q_proj_cpu.view(B, S, self.num_heads, self.head_dim * 2)
        query, gate = q_gate.chunk(2, dim=-1)
        gate = gate.reshape(B, S, -1)

        query = self._rms_norm(query, self.q_norm_w_cpu).transpose(1, 2)
        key = self._rms_norm(
            k_proj_cpu.view(B, S, self.num_kv_heads, self.head_dim), self.k_norm_w_cpu
        ).transpose(1, 2)
        value = v_proj_cpu.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query, key = self._apply_partial_rotary(query, key, cos_cpu, sin_cpu)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            key = torch.cat([k_cache, key], dim=2)
            value = torch.cat([v_cache, value], dim=2)
        new_kv_cache = (key.detach(), value.detach())

        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        attn_weights = torch.matmul(query.float(), key.float().transpose(-1, -2)) * self.scaling
        causal_mask = torch.triu(
            torch.full((S, key.shape[2]), float("-inf"), device=query.device),
            diagonal=key.shape[2] - S + 1,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value.float()).to(query.dtype)

        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        attn_output = attn_output * torch.sigmoid(gate.float()).to(attn_output.dtype)

        attn_output_tt = ttnn.from_torch(
            attn_output.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        output = ttnn.linear(attn_output_tt, self.o_proj_w)
        return output, new_kv_cache

    @staticmethod
    def _rms_norm(x, weight, eps=1e-6):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight * x).to(x.dtype)

    def _apply_partial_rotary(self, q, k, cos, sin):
        d = self.rotary_dim
        q_rot, q_pass = q[..., :d], q[..., d:]
        k_rot, k_pass = k[..., :d], k[..., d:]

        cos = cos[:, :, :q_rot.shape[2], :]
        sin = sin[:, :, :q_rot.shape[2], :]

        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
