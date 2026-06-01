# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import ttnn

from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode


def _repeat_kv_ttnn(
    kv: ttnn.Tensor,
    repeats: int,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Repeat-interleave KV heads on TT: [B, Kv, S, D] -> [B, Kv*repeats, S, D]."""
    if repeats == 1:
        return kv
    b, kv_heads, s, d = tuple(kv.shape)
    repeated = []
    for h in range(kv_heads):
        head = ttnn.slice(kv, [0, h, 0, 0], [b, h + 1, s, d])
        head_rep = ttnn.repeat(head, (1, repeats, 1, 1))
        ttnn.deallocate(head)
        repeated.append(head_rep)
    out = ttnn.concat(repeated, dim=1, memory_config=memory_config)
    for t in repeated:
        ttnn.deallocate(t)
    return out


class VoxtralTTAttention:
    """GQA attention: fused QKV linear, optional RoPE, SDPA, output proj. Audio tokenizer: ``is_causal``, ``use_qk_norm``, ``attn_mask``."""

    def __init__(
        self,
        device,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        state_dict: dict[str, torch.Tensor],
        weight_prefix: str = "attention",
        weight_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        compute_kernel_config=None,
        sdpa_compute_kernel_config=None,
        activation_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        *,
        is_causal: bool = False,
        use_qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        qk_norm_mode: Mode | str = Mode.PREFILL,
    ) -> None:
        self.device = device
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.output_dtype = output_dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        self.compute_kernel_config = compute_kernel_config
        self.sdpa_compute_kernel_config = sdpa_compute_kernel_config or compute_kernel_config
        self.activation_memory_config = activation_memory_config
        self.is_causal = is_causal
        self.use_qk_norm = use_qk_norm
        self._qk_norm_mode = qk_norm_mode

        self.q_norm: RMSNorm | None = None
        self.k_norm: RMSNorm | None = None
        if use_qk_norm:
            prefix = f"{weight_prefix}." if not weight_prefix.endswith(".") else weight_prefix
            qk_norm_weight_dtype = ttnn.bfloat16
            self.q_norm = RMSNorm(
                device=device,
                dim=num_attention_heads * head_dim,
                eps=qk_norm_eps,
                state_dict=state_dict,
                weight_key="q_norm",
                state_dict_prefix=prefix,
                weight_dtype=qk_norm_weight_dtype,
                is_distributed=None,
                tt_ccl=None,
            )
            self.k_norm = RMSNorm(
                device=device,
                dim=num_key_value_heads * head_dim,
                eps=qk_norm_eps,
                state_dict=state_dict,
                weight_key="k_norm",
                state_dict_prefix=prefix,
                weight_dtype=qk_norm_weight_dtype,
                is_distributed=None,
                tt_ccl=None,
            )

        def get_weight(key: str) -> torch.Tensor:
            if key in state_dict:
                return state_dict[key]
            if f"{key}.weight" in state_dict:
                return state_dict[f"{key}.weight"]
            raise KeyError(f"Missing attention weight for key '{key}'")

        wq = get_weight(f"{weight_prefix}.wq").transpose(-2, -1).contiguous()
        wk = get_weight(f"{weight_prefix}.wk").transpose(-2, -1).contiguous()
        wv = get_weight(f"{weight_prefix}.wv").transpose(-2, -1).contiguous()
        wo = get_weight(f"{weight_prefix}.wo").transpose(-2, -1).contiguous()

        wqkv = torch.cat([wq, wk, wv], dim=-1)

        self.wqkv = ttnn.from_torch(
            wqkv,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.wo = ttnn.from_torch(
            wo,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        attention_mask: torch.Tensor | ttnn.Tensor | None = None,
        *,
        attn_mask: ttnn.Tensor | None = None,
        qk_norm_mode: Mode | str | None = None,
        activation_memory_config=None,
    ) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        _qk_mode = self._qk_norm_mode if qk_norm_mode is None else qk_norm_mode
        act_mem = activation_memory_config or self.activation_memory_config

        _lin_kw = {
            "dtype": self.output_dtype,
            "memory_config": act_mem,
        }
        if self.compute_kernel_config is not None:
            _lin_kw["compute_kernel_config"] = self.compute_kernel_config
        xqkv = ttnn.linear(hidden_states, self.wqkv, **_lin_kw)

        if self.use_qk_norm and self.q_norm is not None and self.k_norm is not None:
            b, one, s, _ = tuple(xqkv.shape)
            q_width = self.num_attention_heads * self.head_dim
            k_width = self.num_key_value_heads * self.head_dim
            v_width = self.num_key_value_heads * self.head_dim
            xq = ttnn.slice(xqkv, [0, 0, 0, 0], [b, one, s, q_width])
            xk = ttnn.slice(xqkv, [0, 0, 0, q_width], [b, one, s, q_width + k_width])
            xv = ttnn.slice(
                xqkv,
                [0, 0, 0, q_width + k_width],
                [b, one, s, q_width + k_width + v_width],
            )
            ttnn.deallocate(xqkv)
            xq = self.q_norm(xq, mode=_qk_mode)
            xk = self.k_norm(xk, mode=_qk_mode)
            xqkv = ttnn.concat([xq, xk, xv], dim=3, memory_config=act_mem)
            ttnn.deallocate(xq)
            ttnn.deallocate(xk)
            ttnn.deallocate(xv)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=act_mem,
        )
        ttnn.deallocate(xqkv)

        identity_rope = cos is None or sin is None
        if not identity_rope:
            cos_slice = cos[:, :seq_len]
            sin_slice = sin[:, :seq_len]
            identity_rope = bool(torch.all(cos_slice == 1) and torch.all(sin_slice == 0))

        mask_tt: ttnn.Tensor | None = attn_mask
        mask_owned = False
        if attention_mask is not None:
            if mask_tt is not None:
                ttnn.deallocate(q)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                raise ValueError("Pass only one of attention_mask and attn_mask.")
            if isinstance(attention_mask, torch.Tensor):
                mask_tt = ttnn.from_torch(
                    attention_mask,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                mask_owned = True
            else:
                mask_tt = attention_mask

        if mask_tt is not None and self.is_causal:
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            raise ValueError("is_causal and attn_mask/attention_mask are mutually exclusive.")

        if not identity_rope:
            assert cos is not None and sin is not None
            q_shape = tuple(q.shape)
            k_shape = tuple(k.shape)

            cos_tt = ttnn.from_torch(
                cos_slice.unsqueeze(1),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=act_mem,
            )
            sin_tt = ttnn.from_torch(
                sin_slice.unsqueeze(1),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=act_mem,
            )

            q_in = q
            if q.dtype != ttnn.bfloat16:
                q_in = ttnn.typecast(q, ttnn.bfloat16)
                ttnn.deallocate(q)
            k_in = k
            if k.dtype != ttnn.bfloat16:
                k_in = ttnn.typecast(k, ttnn.bfloat16)
                ttnn.deallocate(k)

            q_rot = ttnn.experimental.rotary_embedding(q_in, cos_tt, sin_tt)
            k_rot = ttnn.experimental.rotary_embedding(k_in, cos_tt, sin_tt)
            ttnn.deallocate(q_in)
            ttnn.deallocate(k_in)
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)

            if tuple(q_rot.shape) != q_shape:
                q_s = ttnn.slice(q_rot, [0, 0, 0, 0], list(q_shape))
                ttnn.deallocate(q_rot)
                q = q_s
            else:
                q = q_rot

            if tuple(k_rot.shape) != k_shape:
                k_s = ttnn.slice(k_rot, [0, 0, 0, 0], list(k_shape))
                ttnn.deallocate(k_rot)
                k = k_s
            else:
                k = k_rot

        n_rep = self.num_attention_heads // self.num_key_value_heads
        k_rep = _repeat_kv_ttnn(k, n_rep, memory_config=act_mem)
        v_rep = _repeat_kv_ttnn(v, n_rep, memory_config=act_mem)
        if n_rep > 1:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        _sdpa_kw = {
            "attn_mask": mask_tt,
            "is_causal": self.is_causal if mask_tt is None else False,
            "scale": self.scale,
        }
        if self.sdpa_compute_kernel_config is not None:
            _sdpa_kw["compute_kernel_config"] = self.sdpa_compute_kernel_config
        attn_out = ttnn.transformer.scaled_dot_product_attention(q, k_rep, v_rep, **_sdpa_kw)
        if mask_owned and mask_tt is not None:
            ttnn.deallocate(mask_tt)
        ttnn.deallocate(q)
        ttnn.deallocate(k_rep)
        ttnn.deallocate(v_rep)

        attn_out = ttnn.experimental.nlp_concat_heads(
            attn_out,
            memory_config=act_mem,
        )

        out = ttnn.linear(attn_out, self.wo, **_lin_kw)
        ttnn.deallocate(attn_out)
        return out
