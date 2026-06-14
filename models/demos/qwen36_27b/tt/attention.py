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


def _device_attention_enabled():
    # On-device decode attention is validated correct (matches the CPU path) and
    # ~1.7x faster with an O(1) KV-cache update (vs the CPU torch.cat that slows
    # with sequence length). Default ON; opt out with USE_DEVICE_ATTENTION=0.
    # Read at call time so tests can toggle via env before constructing.
    return os.environ.get("USE_DEVICE_ATTENTION", "1") != "0"


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

        # On-device KV cache (allocated lazily on first device-mode decode).
        self.k_cache = None
        self.v_cache = None
        self._rope_dev = {}  # position -> (cos_dev, sin_dev) memo for device rope

        # SDPA-decode program config: head_dim=256 + long max_seq overflows L1 with
        # the default chunking, so cap k_chunk_size. q_chunk unused in decode.
        try:
            grid = device.compute_with_storage_grid_size()
            gx, gy = grid.x, grid.y
        except Exception:
            gx, gy = 8, 8
        # k_chunk_size must divide the cache seq length (max_seq_len). Pick the
        # largest tile-aligned chunk in {128,64,32} that divides it.
        k_chunk = 32
        for c in (128, 64, 32):
            if self.max_seq_len % c == 0:
                k_chunk = c
                break
        self.sdpa_decode_prog_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            q_chunk_size=32,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    def reset(self):
        """Drop the on-device KV cache (call between independent sequences)."""
        if self.k_cache is not None:
            ttnn.deallocate(self.k_cache)
            ttnn.deallocate(self.v_cache)
        self.k_cache = None
        self.v_cache = None

    def forward(self, hidden_states, cos, sin, kv_cache=None, mode="decode", current_pos=None):
        if mode == "decode":
            if _device_attention_enabled():
                return self._decode_device(hidden_states, cos, sin, current_pos, kv_cache)
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

    # ------------------------------------------------------------------
    # On-device decode path (USE_DEVICE_ATTENTION=1).
    # First cut — validate/iterate with tests/test_attention_decode.py on a
    # healthy board. Layout choices follow tt_transformers/tt/attention.py:
    #   q for SDPA:  [1, b, n_heads, head_dim]
    #   kv cache:    [b, n_kv_heads, max_seq, head_dim]
    # ------------------------------------------------------------------
    def _make_rope_dev(self, cos, sin):
        """Return device cos/sin [1,1,1,rotary_dim] (bf16, tile). Accepts torch or ttnn.
        Built ONCE per decode step (shared across all heads/layers via broadcast)."""
        d = self.rotary_dim
        if isinstance(cos, torch.Tensor):
            cos = ttnn.from_torch(cos.reshape(1, 1, 1, d).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if isinstance(sin, torch.Tensor):
            sin = ttnn.from_torch(sin.reshape(1, 1, 1, d).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return cos, sin

    def _device_partial_rope(self, x, cos_dev, sin_dev, n_heads):
        """Partial RoPE on device. x: [1,1,n_heads,head_dim]; rotates first rotary_dim dims.
        cos_dev/sin_dev: device [1,1,1,rotary_dim], broadcast over the head dim."""
        d = self.rotary_dim
        half = d // 2
        x_rot = ttnn.slice(x, [0, 0, 0, 0], [1, 1, n_heads, d])
        x_pass = ttnn.slice(x, [0, 0, 0, d], [1, 1, n_heads, self.head_dim])
        # rotate_half over the rotary block: concat(-x2, x1)
        x1 = ttnn.slice(x_rot, [0, 0, 0, 0], [1, 1, n_heads, half])
        x2 = ttnn.slice(x_rot, [0, 0, 0, half], [1, 1, n_heads, d])
        rh = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        # ttnn binary ops broadcast the size-1 head dim of cos/sin over n_heads.
        x_rot_new = ttnn.add(ttnn.mul(x_rot, cos_dev), ttnn.mul(rh, sin_dev))
        return ttnn.concat([x_rot_new, x_pass], dim=-1)

    def _decode_device(self, hidden_states, cos, sin, current_pos, kv_cache=None):
        assert current_pos is not None, "device attention decode needs current_pos"
        cos_cpu = cos if isinstance(cos, torch.Tensor) else ttnn.to_torch(cos)
        sin_cpu = sin if isinstance(sin, torch.Tensor) else ttnn.to_torch(sin)

        q_proj = ttnn.linear(hidden_states, self.q_proj_w)
        k_proj = ttnn.linear(hidden_states, self.k_proj_w)
        v_proj = ttnn.linear(hidden_states, self.v_proj_w)

        q_2d = ttnn.reshape(q_proj, [1, 1, self.num_heads, self.head_dim * 2])
        query = ttnn.slice(q_2d, [0, 0, 0, 0], [1, 1, self.num_heads, self.head_dim])
        gate = ttnn.slice(q_2d, [0, 0, 0, self.head_dim], [1, 1, self.num_heads, self.head_dim * 2])
        key = ttnn.reshape(k_proj, [1, 1, self.num_kv_heads, self.head_dim])
        value = ttnn.reshape(v_proj, [1, 1, self.num_kv_heads, self.head_dim])

        # per-head RMSNorm
        query = ttnn.rms_norm(query, epsilon=1e-6, weight=self.q_norm_w_tt)
        key = ttnn.rms_norm(key, epsilon=1e-6, weight=self.k_norm_w_tt)

        # partial RoPE (build device cos/sin once, shared by q and k via broadcast)
        cos_dev, sin_dev = self._make_rope_dev(cos_cpu, sin_cpu)
        query = self._device_partial_rope(query, cos_dev, sin_dev, self.num_heads)
        key = self._device_partial_rope(key, cos_dev, sin_dev, self.num_kv_heads)

        # allocate KV cache lazily: [b=1, n_kv_heads, max_seq, head_dim].
        # Seed from the CPU prefill kv_cache (post-rope keys/values for positions
        # 0..S-1) so device decode continues the prompt context correctly.
        if self.k_cache is None:
            k_host = torch.zeros(1, self.num_kv_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16)
            v_host = torch.zeros(1, self.num_kv_heads, self.max_seq_len, self.head_dim, dtype=torch.bfloat16)
            if kv_cache is not None:
                k_prev, v_prev = kv_cache  # [1, n_kv_heads, S, head_dim]
                S = k_prev.shape[2]
                k_host[:, :, :S, :] = k_prev.to(torch.bfloat16)
                v_host[:, :, :S, :] = v_prev.to(torch.bfloat16)
            self.k_cache = ttnn.from_torch(k_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            self.v_cache = ttnn.from_torch(v_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # update cache at current_pos. update_cache requires input dim[1] == cache dim[1]
        # (= n_kv_heads). tt_transformers layout is [seqlen=1, n_kv_heads, batch, head_dim],
        # so permute [1,1,n_kv,hd] -> [1, n_kv, 1, hd].
        k_upd = ttnn.permute(key, (0, 2, 1, 3))
        v_upd = ttnn.permute(value, (0, 2, 1, 3))
        ttnn.update_cache(self.k_cache, k_upd, current_pos)
        ttnn.update_cache(self.v_cache, v_upd, current_pos)

        # SDPA decode: q [1, b, n_heads, head_dim]
        q_sdpa = ttnn.reshape(query, [1, 1, self.num_heads, self.head_dim])
        pos_tensor = ttnn.from_torch(torch.tensor([current_pos], dtype=torch.int32),
                                     dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_sdpa, self.k_cache, self.v_cache, cur_pos_tensor=pos_tensor, scale=self.scaling,
            program_config=self.sdpa_decode_prog_cfg,
        )
        # attn: [1, b, n_heads, head_dim] -> [1,1,1, n_heads*head_dim]
        attn = ttnn.reshape(attn, [1, 1, 1, self.num_heads * self.head_dim])

        gate_flat = ttnn.reshape(gate, [1, 1, 1, self.num_heads * self.head_dim])
        attn = ttnn.mul(attn, ttnn.sigmoid(gate_flat))
        output = ttnn.linear(attn, self.o_proj_w)
        return output, (self.k_cache, self.v_cache)
