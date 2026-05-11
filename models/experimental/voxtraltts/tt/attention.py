# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn


def _repeat_kv_ttnn(kv: ttnn.Tensor, repeats: int) -> ttnn.Tensor:
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
    out = ttnn.concat(repeated, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in repeated:
        ttnn.deallocate(t)
    return out


class VoxtralTTAttention:
    """Voxtral text attention (GQA): TT linear, ``nlp_create_qkv_heads``, TT RoPE, SDPA, output proj.

    RoPE uses ``ttnn.experimental.rotary_embedding`` (HF split-half), matching ``models.tt_transformers`` /
    Mistral-style layouts. Acoustic path passes identity cos/sin so RoPE is a no-op without extra ops.
    ``cos`` / ``sin`` may be host tensors only for upload via ``ttnn.from_torch`` (no PyTorch math in forward).
    """

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
    ) -> None:
        self.device = device
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.output_dtype = output_dtype
        self.scale = 1.0 / math.sqrt(head_dim)
        self.compute_kernel_config = compute_kernel_config

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

        # Fused QKV projection for TT head-creation op.
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
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        # hidden_states: [B, 1, S, H]
        seq_len = hidden_states.shape[-2]

        _lin_kw = {
            "dtype": self.output_dtype,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }
        if self.compute_kernel_config is not None:
            _lin_kw["compute_kernel_config"] = self.compute_kernel_config
        xqkv = ttnn.linear(hidden_states, self.wqkv, **_lin_kw)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        cos_slice = cos[:, :seq_len]
        sin_slice = sin[:, :seq_len]
        identity_rope = bool(torch.all(cos_slice == 1) and torch.all(sin_slice == 0))

        if attention_mask is not None:
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            raise NotImplementedError("VoxtralTTAttention does not support attention_mask on TT yet.")

        # Text RoPE on device (HF-style cos/sin broadcast [1, 1, S, D] over heads).
        if not identity_rope:
            q_shape = tuple(q.shape)
            k_shape = tuple(k.shape)

            cos_tt = ttnn.from_torch(
                cos_slice.unsqueeze(1),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            sin_tt = ttnn.from_torch(
                sin_slice.unsqueeze(1),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

        # Acoustic: identity cos/sin — q, k unchanged.
        n_rep = self.num_attention_heads // self.num_key_value_heads
        k_rep = _repeat_kv_ttnn(k, n_rep)
        v_rep = _repeat_kv_ttnn(v, n_rep)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        _sdpa_kw = {
            "attn_mask": None,
            "is_causal": False,
            "scale": self.scale,
        }
        if self.compute_kernel_config is not None:
            _sdpa_kw["compute_kernel_config"] = self.compute_kernel_config
        attn_out = ttnn.transformer.scaled_dot_product_attention(q, k_rep, v_rep, **_sdpa_kw)
        ttnn.deallocate(q)
        ttnn.deallocate(k_rep)
        ttnn.deallocate(v_rep)

        attn_out = ttnn.experimental.nlp_concat_heads(
            attn_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        out = ttnn.linear(attn_out, self.wo, **_lin_kw)
        ttnn.deallocate(attn_out)
        return out
