# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Full ``Mistral4Attention`` forward (no KV cache): projections on device, RoPE + eager SDPA + ``o_proj`` on CPU.

Supports both ``rope_interleave`` settings: when ``True``, uses HF
``apply_rotary_pos_emb_interleave`` (same as ``Mistral4Attention.forward``).
"""

from __future__ import annotations

import torch

from models.tt_transformers.tt.mistral_small_4.attention_slice import (
    attention_kv_b_and_k_rot_from_compressed_bf16,
    attention_q_after_q_bottleneck_bf16,
)
from models.tt_transformers.tt.mistral_small_4.linear import linear_bf16_no_bias


def attention_forward_hybrid_bf16(
    mesh_device,
    hidden_states_bsh: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.LongTensor,
    attn: torch.nn.Module,
    attention_mask: torch.Tensor | None = None,
    *,
    past_key_values=None,
    layer_idx: int | None = None,
) -> torch.Tensor:
    """
    Match HF ``Mistral4Attention.forward`` output (``attn_output`` only) with:

    - **Device bf16:** ``q_a→norm→q_b``, ``kv_a``, ``kv_norm→kv_b``, ``o_proj``
    - **CPU torch:** reshape splits, RoPE (``apply_rotary_pos_emb``), attention scaling,
      ``eager_attention_forward``, reshape before ``o_proj``

    Args:
        hidden_states_bsh: ``[B, S, H]`` bf16.
        position_embeddings: ``(cos, sin)`` from ``Mistral4RotaryEmbedding`` (same as HF decoder).
        position_ids: ``[B, S]`` long.
        attn: HF ``Mistral4Attention`` module (weights + ``config``; ``q_lora_rank`` must be set).

    Requirements:
        ``attn.config._attn_implementation`` should be ``\"eager\"`` for parity with the eager path.

        attention_mask:
            Same as HF ``Mistral4Attention.forward`` (e.g. from ``create_causal_mask``). ``None`` keeps
            the previous fully-visible attention behavior used by early parity tests.

        past_key_values:
            Optional Hugging Face ``Cache`` (e.g. ``DynamicCache``). When set, current-step ``key_states`` /
            ``value_states`` are appended via ``past_key_values.update(...)`` before SDPA, matching HF
            ``Mistral4Attention.forward``.

        layer_idx:
            Layer index for ``past_key_values.update`` (defaults to ``attn.layer_idx`` when ``past_key_values``
            is not ``None``).
    """
    from transformers.models.mistral4.modeling_mistral4 import (
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_interleave,
        eager_attention_forward,
        get_llama_4_attn_scale,
    )

    cfg = attn.config
    if getattr(cfg, "q_lora_rank", None) is None:
        raise ValueError("attention_forward_hybrid_bf16 requires q_lora_rank (Q-LoRA path)")

    batch_size, seq_length, hidden_size = hidden_states_bsh.shape
    nh = int(attn.num_heads)
    qk_nope = int(attn.qk_nope_head_dim)
    qk_rope = int(attn.qk_rope_head_dim)
    qk_head = int(attn.qk_head_dim)
    v_head = int(attn.v_head_dim)

    # --- Device projections ---
    q_flat = attention_q_after_q_bottleneck_bf16(
        mesh_device,
        hidden_states_bsh,
        attn.q_a_proj.weight,
        attn.q_a_layernorm.weight.data,
        float(attn.q_a_layernorm.variance_epsilon),
        attn.q_b_proj.weight,
    )
    compressed = linear_bf16_no_bias(mesh_device, hidden_states_bsh, attn.kv_a_proj_with_mqa.weight)
    k_after_b, k_rot = attention_kv_b_and_k_rot_from_compressed_bf16(
        mesh_device,
        compressed,
        int(attn.kv_lora_rank),
        attn.kv_a_layernorm.weight.data,
        float(attn.kv_a_layernorm.variance_epsilon),
        attn.kv_b_proj.weight,
    )

    # --- CPU: same layout math as HF forward ---
    query_shape = (batch_size, seq_length, nh, qk_head)
    key_shape = (batch_size, seq_length, nh, qk_nope + v_head)

    q_states = q_flat.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [qk_nope, qk_rope], dim=-1)

    k_pass = k_after_b.view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [qk_nope, v_head], dim=-1)

    k_rot = k_rot.reshape(batch_size, 1, seq_length, qk_rope)

    cos, sin = position_embeddings
    if getattr(cfg, "rope_interleave", False):
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    query_states = query_states * get_llama_4_attn_scale(
        position_ids,
        cfg.rope_parameters.get("llama_4_scaling_beta"),
        cfg.rope_parameters.get("original_max_position_embeddings"),
    ).to(query_states.dtype)

    if past_key_values is not None:
        li = int(attn.layer_idx) if layer_idx is None else int(layer_idx)
        key_states, value_states = past_key_values.update(key_states, value_states, li)

    attn_output, _attn_weights = eager_attention_forward(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=float(attn.scaling),
    )

    # ``eager_attention_forward`` already returns ``[B, S, n_heads, v_head_dim]`` layout.
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()

    # o_proj on device (bias-free linear)
    return linear_bf16_no_bias(mesh_device, attn_output, attn.o_proj.weight)


def attention_forward_reference_torch(
    hidden_states_bsh: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.LongTensor,
    attn: torch.nn.Module,
    attention_mask: torch.Tensor | None = None,
    *,
    past_key_values=None,
    use_cache: bool | None = None,
) -> torch.Tensor:
    """Full HF attention on CPU (for parity)."""
    if use_cache is None:
        use_cache = past_key_values is not None
    with torch.no_grad():
        out, _w = attn(
            hidden_states_bsh,
            position_embeddings,
            attention_mask,
            position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
    return out
