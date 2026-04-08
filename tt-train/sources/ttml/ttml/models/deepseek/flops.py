# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Training FLOPs calculation for DeepSeek-V3 architecture.

Standard convention: each parameter contributes 6 FLOPs per token for training
(2 for forward matmul, 2 for backward input grad, 2 for backward weight grad).
Attention QK^T and softmax@V matmuls are counted separately since they scale
with sequence length rather than parameter count.

For MoE: only the activated experts (top-k + shared) contribute to per-token FLOPs,
not the full set of routed experts.
"""

from __future__ import annotations


def calculate_flops_per_token(config, seq_len: int) -> int:
    """Calculate training FLOPs per token for DeepSeek-V3 architecture.

    Follows the standard LLM FLOPs formula:
      flops = 6 * params_active_per_token
            + 6 * n_layers * seq_len * n_heads * qk_head_dim    (Q @ K^T, fwd+bwd)
            + 6 * n_layers * seq_len * n_heads * v_head_dim     (attn_weights @ V, fwd+bwd)

    Args:
        config: DeepSeekConfig instance
        seq_len: Sequence length used for training

    Returns:
        FLOPs per token (int)
    """
    d = config.dim
    n_layers = config.n_layers
    n_dense = config.n_dense_layers
    n_moe = n_layers - n_dense
    n_heads = config.n_heads

    qk_nope = config.qk_nope_head_dim
    qk_rope = config.qk_rope_head_dim
    qk_head = qk_nope + qk_rope
    v_head = config.v_head_dim
    q_lora = config.q_lora_rank
    kv_lora = config.kv_lora_rank

    # ── MLA parameters per layer ──
    # Q path: wq_a + wq_b
    mla_q = d * q_lora + q_lora * n_heads * qk_head
    # KV path: wkv_a + wkv_b
    mla_kv = d * (kv_lora + qk_rope) + kv_lora * n_heads * (qk_nope + v_head)
    # Output: wo
    mla_wo = n_heads * v_head * d
    mla_params = mla_q + mla_kv + mla_wo

    # ── Dense MLP parameters per layer ──
    dense_mlp_params = 3 * d * config.inter_dim

    # ── MoE active parameters per layer ──
    # Only top-k + shared experts are activated per token
    expert_params = 3 * d * config.moe_inter_dim
    moe_active = (config.n_activated_experts + config.n_shared_experts) * expert_params
    # Gate (router) linear: always active
    gate_params = d * config.n_routed_experts
    moe_active_per_layer = moe_active + gate_params

    # ── Embedding + Head ──
    padded_vocab = ((config.vocab_size + 31) // 32) * 32
    embed_params = padded_vocab * d
    head_params = d * padded_vocab

    # ── Total active parameters per token ──
    active_params = (
        embed_params + head_params + n_layers * mla_params + n_dense * dense_mlp_params + n_moe * moe_active_per_layer
    )

    # ── Attention QK^T and attn@V FLOPs (scale with seq_len) ──
    # Q@K^T: [S, qk_head] @ [qk_head, S] per head per layer
    #   fwd: 2*S*qk_head, bwd dQ: 2*S*qk_head, bwd dK: 2*S*qk_head => 6*S*qk_head
    attn_qk_flops = 6 * n_layers * seq_len * n_heads * qk_head
    # attn_weights @ V: [S, S] @ [S, v_head] per head per layer
    #   fwd: 2*S*v_head, bwd d_attn: 2*S*v_head, bwd dV: 2*S*v_head => 6*S*v_head
    attn_v_flops = 6 * n_layers * seq_len * n_heads * v_head

    # ── Total ──
    # 6 FLOPs per parameter per token (fwd + bwd_input + bwd_weight, each 2 FLOPs)
    flops = 6 * active_params + attn_qk_flops + attn_v_flops

    return flops
