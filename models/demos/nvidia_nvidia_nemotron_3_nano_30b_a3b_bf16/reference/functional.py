# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Pure-PyTorch reference functions for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

No TTNN imports.  Each function mirrors the HuggingFace NemotronH module that
carries the same name in the component inventory so that golden tensors can be
generated and PCC-checked against the upstream model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
# The HuggingFace NemotronHModel uses a plain nn.Embedding:
#   self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
# and in forward:
#   inputs_embeds = self.embeddings(input_ids)   # -> [B, S, hidden_size]
#
# Weight stored in checkpoint as: backbone.embeddings.weight
# Shape: [vocab_size=131072, hidden_size=2688], dtype bfloat16
# ---------------------------------------------------------------------------


def embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Token-embedding lookup for Nemotron-3 Nano.

    Replicates ``NemotronHModel.embeddings(input_ids)``.

    Args:
        input_ids: Long tensor of shape [B, S] — token ids in [0, vocab_size).
        weight:    Float/BF16 tensor of shape [vocab_size, hidden_size] —
                   loaded from ``backbone.embeddings.weight`` in the checkpoint.

    Returns:
        Tensor of shape [B, S, hidden_size] in the same dtype as ``weight``.
    """
    return F.embedding(input_ids, weight)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------
# Despite the component being named "LayerNorm" in the architecture inventory,
# the HuggingFace implementation (NemotronHRMSNorm) is an RMSNorm variant —
# identical to T5LayerNorm / LlamaRMSNorm.  It has a single learnable scale
# vector (weight) and no bias.
#
# Forward path from modeling_nemotron_h.py lines 733-739:
#   input_dtype = hidden_states.dtype
#   hidden_states = hidden_states.to(torch.float32)
#   variance = hidden_states.pow(2).mean(-1, keepdim=True)
#   hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#   return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)
# ---------------------------------------------------------------------------


def layer_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """RMSNorm used as the pre-block LayerNorm in Nemotron-3-Nano-30B.

    Equivalent to NemotronHRMSNorm.forward().

    Args:
        hidden_states: Input tensor of arbitrary leading dims followed by
                       hidden_size (e.g. [batch, seq, hidden_size]).
                       Supports any dtype; computation is done in float32.
        weight:        Scale vector of shape [hidden_size], float32 or bfloat16.
        eps:           Variance epsilon (default matches model config: 1e-5).

    Returns:
        Tensor with the same shape and dtype as ``hidden_states``.
    """
    input_dtype = hidden_states.dtype
    hidden_states_f32 = hidden_states.to(torch.float32)
    variance = hidden_states_f32.pow(2).mean(-1, keepdim=True)
    hidden_states_normed = hidden_states_f32 * torch.rsqrt(variance + eps)
    out = weight.to(torch.float32) * hidden_states_normed
    return out.to(input_dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------
# NemotronH uses standard HF-style Rotary Position Embedding (RoPE) in its
# attention layers.  Config parameters from config.json:
#   rope_theta          = 10000
#   partial_rotary_factor = 1.0   (all head_dim dimensions are rotated)
#   head_dim            = 128
#
# The embedding follows NemotronRotaryEmbedding (copied into NemotronH):
#   inv_freq = 1 / (rope_theta ^ (2i / dim))    for i in [0, dim//2)
#   freqs    = outer(positions, inv_freq)
#   emb      = cat(freqs, freqs, dim=-1)         shape [seq_len, dim]
#   cos      = emb.cos() * attention_scaling
#   sin      = emb.sin() * attention_scaling
#
# apply_rotary_pos_emb then performs:
#   q_embed = (q * cos) + (rotate_half(q) * sin)
#   k_embed = (k * cos) + (rotate_half(k) * sin)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the second half of the last dimension into the first, negated."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_compute_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float = 10000.0,
    partial_rotary_factor: float = 1.0,
    attention_scaling: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple:
    """Compute cosine/sine position embeddings for NemotronH RoPE.

    Replicates NemotronRotaryEmbedding.forward() from modeling_nemotron.py
    (which NemotronH inherits).

    Args:
        position_ids:          Long tensor of shape [batch, seq_len] — token positions.
        head_dim:              Head dimension (128 for NemotronH-30B).
        rope_theta:            RoPE base frequency (10000 for NemotronH-30B).
        partial_rotary_factor: Fraction of head_dim to rotate (1.0 = all dims).
        attention_scaling:     Scale applied to cos/sin (1.0 for default rope_type).
        dtype:                 Output dtype for cos/sin tensors.

    Returns:
        Tuple (cos, sin) each of shape [batch, seq_len, rot_dim] in ``dtype``.
    """
    rot_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))  # [rot_dim // 2]

    # position_ids: [batch, seq_len] -> expand inv_freq to [batch, rot_dim//2, 1]
    batch, seq_len = position_ids.shape
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch, -1, 1)  # [B, rot_dim//2, 1]
    position_ids_expanded = position_ids[:, None, :].float()  # [B, 1, S]

    with torch.autocast(device_type="cpu", enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # freqs: [B, S, rot_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, rot_dim]
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int = 128,
    rope_theta: float = 10000.0,
    partial_rotary_factor: float = 1.0,
    attention_scaling: float = 1.0,
) -> tuple:
    """Apply Rotary Position Embedding to query and key tensors.

    Replicates the combined NemotronRotaryEmbedding.forward() +
    apply_rotary_pos_emb() call from NemotronH attention layers.

    Args:
        query:                 Float tensor [batch, num_heads, seq_len, head_dim].
        key:                   Float tensor [batch, num_kv_heads, seq_len, head_dim].
        position_ids:          Long tensor [batch, seq_len] — token positions.
        head_dim:              Per-head feature dimension (128 for NemotronH-30B).
        rope_theta:            RoPE base (10000 for NemotronH-30B).
        partial_rotary_factor: Fraction of head_dim rotated (1.0 for NemotronH-30B).
        attention_scaling:     Multiplicative scale for cos/sin (1.0 for default RoPE).

    Returns:
        Tuple (query_rotated, key_rotated) with the same shapes and dtype as
        the inputs.
    """
    cos, sin = rope_compute_cos_sin(
        position_ids=position_ids,
        head_dim=head_dim,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        attention_scaling=attention_scaling,
        dtype=query.dtype,
    )
    # cos/sin: [B, S, rot_dim] -> unsqueeze head dim -> [B, 1, S, rot_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    rot_dim = cos.shape[-1]
    q_rot, q_pass = query[..., :rot_dim], query[..., rot_dim:]
    k_rot, k_pass = key[..., :rot_dim], key[..., rot_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q_out = torch.cat((q_embed, q_pass), dim=-1)
    k_out = torch.cat((k_embed, k_pass), dim=-1)
    return q_out, k_out


# ---------------------------------------------------------------------------
# MoEAttention
# ---------------------------------------------------------------------------
# In NemotronH the MoE-layer attention block (``NemotronHAttention``) is
# architecturally identical to the DenseAttention block.  Both use standard
# GQA (Group Query Attention) with KV-cache, RoPE, and a causal attention mask.
#
# Config for NemotronH-30B:
#   num_attention_heads   = 32  (Q heads)
#   num_key_value_heads   = 2   (KV heads)
#   head_dim              = 128
#   hidden_size           = 2688
#   rope_theta            = 10000
#   partial_rotary_factor = 1.0
#
# Weight layout (model.layers.{i}.self_attn.*):
#   q_proj.weight : [num_heads * head_dim, hidden_size]     = [4096, 2688]
#   k_proj.weight : [num_kv_heads * head_dim, hidden_size]  = [256,  2688]
#   v_proj.weight : [num_kv_heads * head_dim, hidden_size]  = [256,  2688]
#   o_proj.weight : [hidden_size, num_heads * head_dim]     = [2688, 4096]
# All biases are absent in this model.
#
# Forward (prefill, no KV cache, single sequence):
#   1. q = x @ q_proj.T                  [B, S, num_heads  * head_dim]
#   2. k = x @ k_proj.T                  [B, S, num_kv_heads * head_dim]
#   3. v = x @ v_proj.T                  [B, S, num_kv_heads * head_dim]
#   4. Reshape to multi-head layout       [B, num_heads, S, head_dim]
#   5. Apply RoPE to q, k
#   6. Expand k, v for GQA               [B, num_heads, S, head_dim]
#   7. Scaled dot-product attention       [B, num_heads, S, head_dim]
#   8. Reshape back to [B, S, hidden_size]
#   9. out = attn @ o_proj.T             [B, S, hidden_size]
# ---------------------------------------------------------------------------


def moe_attention(
    hidden_states: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    position_ids: torch.Tensor,
    num_heads: int = 32,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    rope_theta: float = 10000.0,
    partial_rotary_factor: float = 1.0,
    attention_scaling: float = 1.0,
) -> torch.Tensor:
    """GQA attention block used in MoE transformer layers of NemotronH-30B.

    This function is architecturally identical to the DenseAttention reference;
    the distinction is purely in which layer index it is called on.

    Replicates ``NemotronHAttention.forward()`` (prefill / full-sequence mode,
    no KV-cache state passed in).

    Args:
        hidden_states:  [batch, seq_len, hidden_size] in bfloat16.
        wq:             Q projection weight [num_heads * head_dim, hidden_size].
        wk:             K projection weight [num_kv_heads * head_dim, hidden_size].
        wv:             V projection weight [num_kv_heads * head_dim, hidden_size].
        wo:             Output projection weight [hidden_size, num_heads * head_dim].
        position_ids:   Long tensor [batch, seq_len] — token positions.
        num_heads:      Number of query heads (32 for NemotronH-30B).
        num_kv_heads:   Number of KV heads (2 for NemotronH-30B).
        head_dim:       Head dimension (128 for NemotronH-30B).
        rope_theta:     RoPE base frequency (10000 for NemotronH-30B).
        partial_rotary_factor: Fraction of head_dim to rotate (1.0).
        attention_scaling:    Multiplicative scale for cos/sin (1.0).

    Returns:
        Tensor [batch, seq_len, hidden_size] in the same dtype as
        ``hidden_states``.
    """
    bsz, seq_len, hidden_size = hidden_states.shape
    num_groups = num_heads // num_kv_heads  # GQA expansion factor

    # --- QKV projections ---
    q = F.linear(hidden_states, wq)  # [B, S, num_heads * head_dim]
    k = F.linear(hidden_states, wk)  # [B, S, num_kv_heads * head_dim]
    v = F.linear(hidden_states, wv)  # [B, S, num_kv_heads * head_dim]

    # Reshape to multi-head layout
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, nH,  S, D]
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # [B, nKV, S, D]
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)  # [B, nKV, S, D]

    # --- Rotary Position Embedding ---
    cos, sin = rope_compute_cos_sin(
        position_ids=position_ids,
        head_dim=head_dim,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        attention_scaling=attention_scaling,
        dtype=hidden_states.dtype,
    )
    # cos/sin: [B, S, rot_dim] -> [B, 1, S, rot_dim] for broadcasting over heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat((q_embed, q_pass), dim=-1)  # [B, nH,  S, D]
    k = torch.cat((k_embed, k_pass), dim=-1)  # [B, nKV, S, D]

    # --- GQA: expand K and V to full num_heads ---
    # Each KV head serves num_groups Q heads
    k = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(bsz, num_heads, seq_len, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, num_groups, -1, -1).reshape(bsz, num_heads, seq_len, head_dim)

    # --- Scaled dot-product attention (causal) ---
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        scale=head_dim**-0.5,
    )  # [B, nH, S, D]

    # --- Reshape and output projection ---
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)
    output = F.linear(attn_output, wo)  # [B, S, hidden_size]
    return output


# ---------------------------------------------------------------------------
# DenseAttention
# ---------------------------------------------------------------------------
# NemotronH "dense" attention layers are the 6 blocks where the
# hybrid_override_pattern character is '*'.  Layer indices: 5, 12, 19, 26, 33,
# 42 (0-based).
#
# Each block wraps:
#   1. Pre-block RMSNorm  (backbone.layers.{i}.norm.weight)
#   2. NemotronHAttention (GQA, 32 Q heads / 2 KV heads, head_dim=128)
#      Weights: backbone.layers.{i}.mixer.{q,k,v,o}_proj.weight
#      NOTE: The HuggingFace implementation has RoPE commented out with a TODO,
#            so *no* positional encoding is applied in the reference.
#   3. Residual addition
#
# NemotronHBlock.forward():
#   residual      = hidden_states
#   hidden_states = norm(hidden_states)
#   hidden_states = mixer(hidden_states, cache_position=...)[0]
#   hidden_states = residual + hidden_states
#
# NemotronHAttention.forward() (no RoPE — TODO comment in source):
#   Q = q_proj(x)    [B, S, n_heads*head_dim]
#   K = k_proj(x)    [B, S, n_kv*head_dim]
#   V = v_proj(x)    [B, S, n_kv*head_dim]
#   reshape/transpose to [B, H, S, D]
#   repeat_kv(K, groups), repeat_kv(V, groups)
#   SDPA (causal when seqlen > 1)
#   transpose/reshape -> [B, S, n_heads*head_dim]
#   output = o_proj(attn_output)
# ---------------------------------------------------------------------------


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match Q head count (GQA -> MHA expansion)."""
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


def dense_attention(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    num_heads: int = 32,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Pure-PyTorch DenseAttention block for Nemotron-3 Nano 30B.

    Replicates the NemotronHBlock (block_type='attention') forward pass with
    the NemotronHAttention eager implementation.

    NOTE: RoPE is intentionally absent — the HuggingFace implementation has
    positional embeddings commented out with a TODO for this model variant.

    Args:
        hidden_states:  Input tensor of shape [batch, seq_len, hidden_size].
        norm_weight:    RMSNorm scale vector of shape [hidden_size].
        wq:             Q projection weight  [num_heads*head_dim, hidden_size].
        wk:             K projection weight  [num_kv_heads*head_dim, hidden_size].
        wv:             V projection weight  [num_kv_heads*head_dim, hidden_size].
        wo:             O projection weight  [hidden_size, num_heads*head_dim].
        num_heads:      Number of Q attention heads (32 for NemotronH-30B).
        num_kv_heads:   Number of KV heads (2 for NemotronH-30B GQA).
        head_dim:       Per-head feature dimension (128 for NemotronH-30B).
        norm_eps:       RMSNorm epsilon (1e-5 for NemotronH-30B).

    Returns:
        Tensor of shape [batch, seq_len, hidden_size] — post-residual output.
    """
    residual = hidden_states

    # 1. Pre-norm (RMSNorm)
    normed = layer_norm(hidden_states, norm_weight, eps=norm_eps)

    # 2. Attention (no RoPE — HF source has it TODO-commented)
    bsz, q_len, _ = normed.shape
    num_kv_groups = num_heads // num_kv_heads

    q = F.linear(normed, wq)  # [B, S, num_heads*head_dim]
    k = F.linear(normed, wk)  # [B, S, num_kv_heads*head_dim]
    v = F.linear(normed, wv)  # [B, S, num_kv_heads*head_dim]

    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)  # [B, nH,  S, D]
    k = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # [B, nKV, S, D]
    v = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # [B, nKV, S, D]

    # GQA: expand KV heads to match Q heads
    k = _repeat_kv(k, num_kv_groups)  # [B, nH, S, D]
    v = _repeat_kv(v, num_kv_groups)  # [B, nH, S, D]

    # Scaled dot-product attention (causal for seqlen > 1)
    is_causal = q_len > 1
    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
    )  # [B, nH, S, D]

    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
    attn_out = F.linear(attn_out, wo)  # [B, S, hidden_size]

    # 3. Residual
    return residual + attn_out


# ---------------------------------------------------------------------------
# MoEGate  (NemotronHTopkRouter)
# ---------------------------------------------------------------------------
# NemotronH uses NemotronHTopkRouter as the MoE gate in every MoE layer.
#
# Config for NemotronH-30B (from config.json):
#   n_routed_experts    = 128
#   num_experts_per_tok = 6    (top-6 routing)
#   n_group             = 1
#   topk_group          = 1
#   norm_topk_prob      = True
#   routed_scaling_factor = 2.5
#   hidden_size         = 2688
#
# Weight layout (backbone.layers.{i}.mixer.gate.*):
#   weight               : [n_routed_experts, hidden_size] = [128, 2688], float32
#   e_score_correction_bias : [n_routed_experts] = [128], float32
#
# Forward pass (NemotronHTopkRouter.forward):
#   1. hidden = hidden_states.view(-1, hidden_size)         [tokens, hidden_size]
#   2. router_logits = hidden @ weight.T                    [tokens, 128]
#   3. scores = sigmoid(router_logits)                      [tokens, 128]
#   4. topk_indices = get_topk_indices(scores)              [tokens, 6]
#   5. topk_weights = scores.gather(1, topk_indices)        [tokens, 6]
#   6. if norm_topk_prob:
#          topk_weights /= (topk_weights.sum(-1, keepdim=True) + 1e-20)
#   7. topk_weights *= routed_scaling_factor
#
# get_topk_indices (with n_group=1, topk_group=1 for NemotronH-30B):
#   With n_group=1 the group machinery collapses to a trivial identity:
#   - scores_for_choice = scores + e_score_correction_bias   [tokens, 128]
#   - group_scores      = top2-sum per group => sum of top-2 of all 128      [tokens, 1]
#   - group_idx         = topk(group_scores, 1) = [0]                         [tokens, 1]
#   - group_mask        = ones_like(group_scores)                              [tokens, 1]
#   - score_mask        = ones  (all experts unmasked)                        [tokens, 128]
#   - topk_indices      = topk(scores_for_choice, 6)                          [tokens, 6]
#
# Output: (topk_indices [tokens, top_k], topk_weights [tokens, top_k])
# ---------------------------------------------------------------------------


def moe_gate(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    n_routed_experts: int = 128,
    num_experts_per_tok: int = 6,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple:
    """Pure-PyTorch MoE gate replicating NemotronHTopkRouter.forward().

    All computation is done in float32.  hidden_states may be bfloat16;
    it is cast to float32 internally as the HuggingFace reference does.

    Args:
        hidden_states:           Input tensor of shape [..., hidden_size].
                                 Leading dimensions are flattened to tokens.
        weight:                  Gate projection weight [n_routed_experts, hidden_size],
                                 float32.  Loaded from
                                 ``backbone.layers.{i}.mixer.gate.weight``.
        e_score_correction_bias: Correction bias [n_routed_experts], float32.
                                 Loaded from
                                 ``backbone.layers.{i}.mixer.gate.e_score_correction_bias``.
        n_routed_experts:        Total number of experts (128 for NemotronH-30B).
        num_experts_per_tok:     Number of experts per token (6).
        n_group:                 Number of expert groups (1 for NemotronH-30B).
        topk_group:              Number of groups to keep (1).
        norm_topk_prob:          Normalize routing weights to sum to 1 (True).
        routed_scaling_factor:   Scale applied after normalisation (2.5).

    Returns:
        Tuple (topk_indices, topk_weights) each of shape [tokens, num_experts_per_tok].
        topk_indices: torch.int64
        topk_weights: torch.float32
    """
    # Flatten batch/sequence dimensions -> [tokens, hidden_size]
    tokens = hidden_states.view(-1, hidden_states.shape[-1])
    tokens_f32 = tokens.to(torch.float32)

    # 1. Router logits: [tokens, n_routed_experts]
    router_logits = F.linear(tokens_f32, weight.to(torch.float32))

    # 2. Sigmoid scores
    scores = router_logits.sigmoid()  # [tokens, n_routed_experts]

    # 3. get_topk_indices — implements the grouped top-k selection
    scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)
    # [tokens, n_group, experts_per_group]
    scores_3d = scores_for_choice.view(-1, n_group, n_routed_experts // n_group)
    group_scores = scores_3d.topk(2, dim=-1)[0].sum(dim=-1)  # [tokens, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  # [tokens, topk_group]
    group_mask = torch.zeros_like(group_scores)  # [tokens, n_group]
    group_mask.scatter_(1, group_idx, 1)
    # Expand mask back to per-expert: [tokens, n_group, experts_per_group] -> [tokens, n_routed_experts]
    score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, n_routed_experts // n_group).reshape(-1, n_routed_experts)
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False)[1]

    # 4. Gather routing weights from the *unbiased* scores
    topk_weights = scores.gather(1, topk_indices)  # [tokens, num_experts_per_tok]

    # 5. Normalise
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator

    # 6. Scale
    topk_weights = topk_weights * routed_scaling_factor

    return topk_indices, topk_weights


# ---------------------------------------------------------------------------
# DenseMLP  (NemotronHMLP inside NemotronHBlock with block_type='mlp')
# ---------------------------------------------------------------------------
# In NemotronH the "dense MLP" block is a NemotronHBlock whose mixer is a
# NemotronHMLP module (block_type == 'mlp', i.e. hybrid_override_pattern '-').
# Although the 30B-A3B checkpoint does not contain any '-' layers (its pattern
# only has M / E / *), the component appears in the architecture inventory as a
# separate bring-up target and the identical MLP structure is reused in the
# shared-expert of every MoE layer.
#
# NemotronHMLP forward:
#   out = down_proj( act_fn( up_proj(x) ) )
# where act_fn = ACT2FN["relu2"] = ReLUSquaredActivation = relu(x)^2
#
# NemotronHBlock forward (mlp path):
#   residual      = hidden_states
#   hidden_states = norm(hidden_states)          RMSNorm, same as LayerNorm
#   hidden_states = mixer(hidden_states)         NemotronHMLP
#   hidden_states = residual + hidden_states
#
# Weight layout (backbone.layers.{i}.mixer.*):
#   up_proj.weight   : [intermediate_size, hidden_size] = [1856, 2688]  bfloat16
#   down_proj.weight : [hidden_size, intermediate_size] = [2688, 1856]  bfloat16
#   norm weight      : backbone.layers.{i}.norm.weight  [hidden_size]   bfloat16
# No bias (mlp_bias = False).
#
# Config values for NemotronH-30B:
#   hidden_size       = 2688
#   intermediate_size = 1856   (dense MLP)
# ---------------------------------------------------------------------------


def dense_mlp(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Pure-PyTorch DenseMLP block for Nemotron-3 Nano 30B.

    Replicates the NemotronHBlock (block_type='mlp') forward pass, which
    wraps a NemotronHMLP mixer with pre-block RMSNorm and a residual
    connection.

    Activation is ``relu2`` (squared ReLU): ``relu(x) ** 2``, as defined by
    ``ReLUSquaredActivation`` in ``transformers.activations``.

    NOTE: This model's 30B-A3B checkpoint has no '-' layers in its
    ``hybrid_override_pattern``, so the identical MLP structure appears as
    the *shared expert* in MoE blocks (``backbone.layers.{i}.mixer.shared_experts.*``).
    The reference function is intentionally decoupled from any specific layer
    index so it can be validated with synthetic weights.

    Args:
        hidden_states:  Input tensor of shape [batch, seq_len, hidden_size].
                        Expected dtype: bfloat16.
        norm_weight:    RMSNorm scale vector of shape [hidden_size], bfloat16
                        or float32.  Loaded from
                        ``backbone.layers.{i}.norm.weight``.
        w_up:           Up-projection weight of shape
                        [intermediate_size, hidden_size] (as stored in the
                        checkpoint; ``F.linear`` transposes internally).
                        Loaded from
                        ``backbone.layers.{i}.mixer.up_proj.weight``.
        w_down:         Down-projection weight of shape
                        [hidden_size, intermediate_size].  Loaded from
                        ``backbone.layers.{i}.mixer.down_proj.weight``.
        norm_eps:       RMSNorm epsilon (1e-5 for NemotronH-30B).

    Returns:
        Tensor of shape [batch, seq_len, hidden_size] in the same dtype as
        ``hidden_states`` — the post-residual block output.
    """
    residual = hidden_states

    # 1. Pre-norm (RMSNorm — same as layer_norm() above)
    normed = layer_norm(hidden_states, norm_weight, eps=norm_eps)

    # 2. Up projection
    up = F.linear(normed, w_up)  # [B, S, intermediate_size]

    # 3. Squared-ReLU activation (relu2 = relu(x)^2)
    activated = F.relu(up) ** 2  # [B, S, intermediate_size]

    # 4. Down projection
    down = F.linear(activated, w_down)  # [B, S, hidden_size]

    # 5. Residual
    return residual + down


# ---------------------------------------------------------------------------
# Convenience nn.Module wrapper (not used by TTNN, but useful for golden gen)
# ---------------------------------------------------------------------------


class NemotronHRMSNorm(nn.Module):
    """Drop-in replacement for HuggingFace NemotronHRMSNorm for golden tensor
    generation.  Weights can be loaded directly from a HuggingFace checkpoint.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return layer_norm(hidden_states, self.weight, self.variance_epsilon)


# ---------------------------------------------------------------------------
# SharedExpert  (NemotronHMLP inside NemotronHMOE as shared_experts)
# ---------------------------------------------------------------------------
# In each MoE layer, ALL tokens pass through a shared (non-routed) expert
# implemented as NemotronHMLP with moe_shared_expert_intermediate_size=3712.
#
# NemotronHMOE.forward() (simplified):
#   residuals = hidden_states            # already pre-normed by the block
#   topk_indices, topk_weights = gate(hidden_states)
#   hidden_states = moe(hidden_states, ...)   # routed expert outputs
#   hidden_states = hidden_states + shared_experts(residuals)
#
# So shared_experts receives the pre-normed hidden states (the same tensor
# passed to the MoE gate and routed experts).
#
# NemotronHMLP forward:
#   out = down_proj( relu(up_proj(x))^2 )
# activation = relu2 (squared ReLU), no bias.
#
# Config values for NemotronH-30B:
#   hidden_size                       = 2688
#   moe_shared_expert_intermediate_size = 3712  (distinct from moe_intermediate_size=1856)
#
# Weight layout (backbone.layers.{i}.mixer.shared_experts.*):
#   up_proj.weight   : [moe_shared_expert_intermediate_size, hidden_size] = [3712, 2688]
#   down_proj.weight : [hidden_size, moe_shared_expert_intermediate_size] = [2688, 3712]
# No bias (mlp_bias = False).
# ---------------------------------------------------------------------------


def shared_expert(
    hidden_states: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch SharedExpert (shared MLP) for Nemotron-3 Nano 30B MoE layers.

    Replicates ``NemotronHMLP.forward()`` when used as the shared (non-routed)
    expert inside ``NemotronHMOE``.  All tokens pass through this expert
    regardless of routing decisions.

    The caller is responsible for supplying **pre-normed** hidden states — i.e.
    the same tensor that the block's pre-norm has already processed and that
    ``NemotronHMOE.forward()`` stores as ``residuals`` before routing.

    Activation is ``relu2`` (squared ReLU): ``relu(x) ** 2``, matching all
    NemotronHMLP variants in this model.

    Args:
        hidden_states:  Pre-normed input of shape [batch, seq_len, hidden_size].
                        Expected dtype: bfloat16.
        w_up:           Up-projection weight of shape
                        [moe_shared_expert_intermediate_size, hidden_size]
                        = [3712, 2688].  Loaded from
                        ``backbone.layers.{i}.mixer.shared_experts.up_proj.weight``.
        w_down:         Down-projection weight of shape
                        [hidden_size, moe_shared_expert_intermediate_size]
                        = [2688, 3712].  Loaded from
                        ``backbone.layers.{i}.mixer.shared_experts.down_proj.weight``.

    Returns:
        Tensor of shape [batch, seq_len, hidden_size] in the same dtype as
        ``hidden_states``.  This is the shared-expert contribution to be
        added to the routed-MoE output before the block residual.
    """
    # Up projection: [B, S, hidden_size] -> [B, S, moe_shared_expert_intermediate_size]
    up = F.linear(hidden_states, w_up)

    # relu2 activation: relu(x)^2
    activated = F.relu(up) ** 2

    # Down projection: [B, S, moe_shared_expert_intermediate_size] -> [B, S, hidden_size]
    return F.linear(activated, w_down)


# ---------------------------------------------------------------------------
# LMHead  (final RMSNorm + vocabulary projection)
# ---------------------------------------------------------------------------
# In NemotronHForCausalLM the last step after all transformer layers is:
#   hidden_states = self.model.backbone.norm_f(hidden_states)  # final RMSNorm
#   logits        = self.lm_head(hidden_states)                # nn.Linear, no bias
#
# Weights (both in model-00013-of-00013.safetensors, the last shard):
#   backbone.norm_f.weight : [hidden_size=2688]                    bfloat16
#   lm_head.weight         : [vocab_size=131072, hidden_size=2688] bfloat16
#
# tie_word_embeddings = False — lm_head.weight is *not* shared with the
# token embedding; it is an independent parameter.
#
# The norm_f RMSNorm is identical to the per-layer NemotronHRMSNorm:
#   h = h.float()
#   h = h * rsqrt(mean(h^2, dim=-1, keepdim=True) + eps)
#   h = (weight.float() * h).to(input_dtype)
# ---------------------------------------------------------------------------


def lm_head(
    hidden_states: torch.Tensor,  # [B, S, 2688] final layer output
    norm_f_weight: torch.Tensor,  # [2688] final RMSNorm weight
    lm_head_weight: torch.Tensor,  # [131072, 2688]
    norm_eps: float = 1e-5,
) -> torch.Tensor:  # [B, S, 131072] logits
    """Final RMSNorm + vocabulary projection for NemotronH-30B.

    Replicates the tail of ``NemotronHForCausalLM.forward()``:
        hidden_states = model.backbone.norm_f(hidden_states)
        logits        = lm_head(hidden_states)

    Args:
        hidden_states:  [batch, seq_len, hidden_size] — output of the last
                        transformer layer.  Expected dtype: bfloat16.
        norm_f_weight:  Final RMSNorm scale vector of shape [hidden_size].
                        Loaded from ``backbone.norm_f.weight``.
        lm_head_weight: Vocabulary projection weight of shape
                        [vocab_size, hidden_size].  Loaded from
                        ``lm_head.weight``.  ``F.linear`` transposes it
                        internally, so no manual transpose is needed.
                        ``tie_word_embeddings=False`` — this weight is
                        independent of the token embedding.
        norm_eps:       RMSNorm epsilon (1e-5 for NemotronH-30B).

    Returns:
        Logits tensor of shape [batch, seq_len, vocab_size] in bfloat16.
    """
    # 1. Final RMSNorm (reuses layer_norm defined above)
    normed = layer_norm(hidden_states, norm_f_weight, eps=norm_eps)  # [B, S, 2688]

    # 2. Vocabulary projection — F.linear computes normed @ lm_head_weight.T
    logits = F.linear(normed.float(), lm_head_weight.float()).bfloat16()  # [B, S, 131072]

    return logits


# ---------------------------------------------------------------------------
# MoEExperts  (routed expert computation inside NemotronHMoE)
# ---------------------------------------------------------------------------
# In NemotronH each MoE layer (hybrid_override_pattern 'E') contains a
# NemotronHMoE module whose core is a bank of 128 NemotronHMLP experts.
# After the gate produces (topk_indices, topk_weights), only the top-6
# experts per token are computed; their outputs are weighted and summed.
#
# NemotronHMLP forward (each expert):
#   out = down_proj( relu(up_proj(x))^2 )   -- activation is relu2
# where:
#   up_proj.weight   : [moe_intermediate=1856, hidden=2688]  bfloat16
#   down_proj.weight : [hidden=2688, moe_intermediate=1856]  bfloat16
#
# The full MoEExperts computation:
#   final = zeros_like(hidden_states)
#   for each expert e in 0..127:
#       find all tokens routed to expert e
#       expert_out = down_proj_e( relu(up_proj_e(tokens))^2 )
#       weighted_out = expert_out * topk_weights[token, weight_slot]
#       final.index_add_(0, token_indices, weighted_out)
#
# Config values for NemotronH-30B:
#   n_routed_experts    = 128
#   num_experts_per_tok = 6      (top-6 routing)
#   hidden_size         = 2688
#   moe_intermediate    = 1856   (per-expert intermediate dimension)
# ---------------------------------------------------------------------------


def moe_experts(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_up_weights: list,
    expert_down_weights: list,
) -> torch.Tensor:
    """Pure-PyTorch routed MoE expert computation for NemotronH-30B.

    Replicates the routed-expert portion of ``NemotronHMoE.forward()`` —
    the loop over 128 NemotronHMLP experts, weighted by the gate output.
    This does NOT include the shared expert; it corresponds exactly to the
    ``y`` tensor computed before ``y += shared_expert_output`` in the HF code.

    Activation is ``relu2`` (squared ReLU): ``relu(x) ** 2``, matching
    ``mlp_hidden_act = "relu2"`` in config.json.

    Args:
        hidden_states:       [N_tokens, hidden_size] -- post-norm token features,
                             typically bfloat16.
        topk_indices:        [N_tokens, top_k] int64 -- expert indices from gate.
        topk_weights:        [N_tokens, top_k] float -- normalised routing weights
                             from gate (already scaled by routed_scaling_factor).
        expert_up_weights:   List of 128 tensors, each [moe_intermediate, hidden_size]
                             (as stored in the checkpoint; F.linear transposes).
                             Loaded from
                             ``backbone.layers.{i}.mixer.experts.{j}.up_proj.weight``.
        expert_down_weights: List of 128 tensors, each [hidden_size, moe_intermediate].
                             Loaded from
                             ``backbone.layers.{i}.mixer.experts.{j}.down_proj.weight``.

    Returns:
        Tensor of shape [N_tokens, hidden_size] -- sum of weighted expert outputs,
        same dtype as ``hidden_states``.
    """
    n_tokens, hidden_size = hidden_states.shape
    n_experts = len(expert_up_weights)

    # Accumulator in the same dtype as the input
    final = torch.zeros_like(hidden_states)

    # Build a [n_experts, n_tokens, top_k] one-hot style mask to avoid
    # repeated scans: for each expert, find which (token, slot) pairs route to it.
    # topk_indices: [N_tokens, top_k] -> one_hot: [N_tokens, top_k, n_experts]
    # -> permute: [n_experts, N_tokens, top_k]
    one_hot = F.one_hot(topk_indices, num_classes=n_experts).permute(2, 0, 1)
    # one_hot[e]: [N_tokens, top_k] boolean mask for expert e

    for e in range(n_experts):
        # token_idx: which tokens go to expert e
        # weight_idx: which slot (0..top_k-1) carries that assignment
        token_idx, weight_idx = torch.where(one_hot[e])  # each shape [num_assigned]
        if token_idx.numel() == 0:
            continue

        # Gather the tokens assigned to this expert
        x_e = hidden_states[token_idx]  # [num_assigned, hidden_size]

        # Expert forward: relu2 MLP
        up = F.linear(x_e, expert_up_weights[e])  # [num_assigned, moe_intermediate]
        activated = F.relu(up) ** 2  # relu2 activation
        out = F.linear(activated, expert_down_weights[e])  # [num_assigned, hidden_size]

        # Gather the scalar routing weights for these (token, slot) pairs
        weights = topk_weights[token_idx, weight_idx].unsqueeze(-1)  # [num_assigned, 1]

        # Weighted accumulation into the output; cast to match final dtype (bfloat16)
        final.index_add_(0, token_idx, (out * weights).to(final.dtype))

    return final


# ---------------------------------------------------------------------------
# Mamba2Layer  (NemotronHBlock with block_type='mamba2')
# ---------------------------------------------------------------------------
# NemotronH layer 0 is a Mamba2 block (hybrid_override_pattern[0] == 'M').
# The full block forward is:
#   residual      = hidden_states
#   hidden_states = pre_norm(hidden_states)              RMSNorm
#   hidden_states = mixer.torch_forward(hidden_states)   NemotronHMamba2Mixer
#   hidden_states = residual + hidden_states
#
# Config for NemotronH-30B:
#   hidden_size      = 2688
#   num_heads        = 64    (mamba_num_heads)
#   head_dim         = 64    (mamba_head_dim)
#   intermediate_size= 4096  (= num_heads * head_dim)
#   n_groups         = 8
#   ssm_state_size   = 128
#   conv_kernel      = 4
#   chunk_size       = 128
#   time_step_limit  = (0.0, inf)   -> clamp is a no-op
#
# Weight layout (backbone.layers.0.mixer.*):
#   in_proj.weight       : [10304, 2688]   (10304 = 4096+6144+64), no bias
#   conv1d.weight        : [6144, 1, 4]
#   conv1d.bias          : [6144]
#   dt_bias              : [64]
#   A_log                : [64]
#   norm.weight          : [4096]          MambaRMSNormGated
#   D                    : [64]
#   out_proj.weight      : [2688, 4096]    no bias
#
# Pre-block norm weight: backbone.layers.0.norm.weight [2688]
# ---------------------------------------------------------------------------


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Pad sequence dimension (dim=1) with zeros."""
    if pad_size == 0:
        return input_tensor
    if input_tensor.ndim == 4:
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0)
    else:  # ndim == 3
        pad_shape = (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(
    input_tensor: torch.Tensor,
    pad_size: int,
    chunk_size: int,
) -> torch.Tensor:
    """Pad + reshape sequence dimension into chunks."""
    t = _pad_tensor_by_size(input_tensor, pad_size)
    if t.ndim == 3:
        # [B, S_padded, D] -> [B, num_chunks, chunk_size, D]
        return t.reshape(t.shape[0], -1, chunk_size, t.shape[2])
    else:
        # [B, S_padded, H, D] -> [B, num_chunks, chunk_size, H, D]
        return t.reshape(t.shape[0], -1, chunk_size, t.shape[2], t.shape[3])


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """Compute causal segment sums for SSD chunking.

    Mirrors segment_sum() from modeling_nemotron_h.py.
    """
    cs = input_tensor.size(-1)
    # [..., cs] -> [..., cs, cs]
    t = input_tensor[..., None].expand(*input_tensor.size(), cs)
    mask_lower = torch.tril(torch.ones(cs, cs, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    t = t.masked_fill(~mask_lower, 0)
    seg = torch.cumsum(t, dim=-2)
    mask_diag = torch.tril(torch.ones(cs, cs, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    seg = seg.masked_fill(~mask_diag, float("-inf"))
    return seg


def _mamba_rms_norm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    group_size: int = 512,
) -> torch.Tensor:
    """MambaRMSNormGated: element-wise gate then per-group RMSNorm then scale.

    Mirrors MambaRMSNormGated.forward(hidden_states=x, gate=z) calling
    rmsnorm_fn with norm_before_gate=False (NemotronH convention).

    With norm_before_gate=False:
      1. gate = silu(z)
      2. xg   = x * gate
      3. per-group RMSNorm on xg  (group_size = intermediate_size // n_groups)
      4. scale by weight

    Args:
        x:          [B, S, D]  -- scan output
        z:          [B, S, D]  -- gate from in_proj split
        weight:     [D]        -- learnable scale
        eps:        variance epsilon
        group_size: features per RMSNorm group (4096//8 = 512 for NemotronH-30B)

    Returns:
        Tensor [B, S, D] in the same dtype as x.
    """
    input_dtype = x.dtype
    B, S, D = x.shape

    # Gate first (norm_before_gate=False)
    gate = F.silu(z.float())
    xg = x.float() * gate  # [B, S, D]

    # Per-group RMSNorm
    xg_grouped = xg.view(B, S, -1, group_size)
    var = xg_grouped.pow(2).mean(-1, keepdim=True)
    xg_normed = xg_grouped * torch.rsqrt(var + eps)
    xg_normed = xg_normed.view(B, S, D)

    out = weight.float() * xg_normed
    return out.to(input_dtype)


def mamba2_layer(
    hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    in_proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    norm_mixer_weight: torch.Tensor,
    D: torch.Tensor,
    out_proj_weight: torch.Tensor,
    norm_eps: float = 1e-5,
    num_heads: int = 64,
    head_dim: int = 64,
    n_groups: int = 8,
    ssm_state_size: int = 128,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Full Mamba2 block (pre-norm + NemotronHMamba2Mixer + residual) for
    NemotronH-30B, prefill path (full-sequence SSD scan, no cache).

    Replicates NemotronHBlock.forward() -> NemotronHMamba2Mixer.torch_forward()
    with cache_params=None (the 'else' branch = chunked SSD naive implementation).

    Args:
        hidden_states:      Input [B, S, hidden_size] -- typically bfloat16.
        norm_weight:        Pre-block RMSNorm scale [hidden_size].
                            Loaded from 'backbone.layers.0.norm.weight'.
        in_proj_weight:     [proj_size, hidden_size] where
                            proj_size = intermediate_size + conv_dim + num_heads
                                      = 4096 + 6144 + 64 = 10304.
                            Loaded from 'backbone.layers.0.mixer.in_proj.weight'.
        conv1d_weight:      Depthwise conv weight [conv_dim, 1, conv_kernel].
                            Loaded from 'backbone.layers.0.mixer.conv1d.weight'.
        conv1d_bias:        Depthwise conv bias [conv_dim].
                            Loaded from 'backbone.layers.0.mixer.conv1d.bias'.
        dt_bias:            Time-step bias [num_heads].
                            Loaded from 'backbone.layers.0.mixer.dt_bias'.
        A_log:              Log of A [num_heads] (positive -> A < 1).
                            Loaded from 'backbone.layers.0.mixer.A_log'.
        norm_mixer_weight:  MambaRMSNormGated scale [intermediate_size].
                            Loaded from 'backbone.layers.0.mixer.norm.weight'.
        D:                  Skip-connection scale [num_heads].
                            Loaded from 'backbone.layers.0.mixer.D'.
        out_proj_weight:    Output projection [hidden_size, intermediate_size].
                            Loaded from 'backbone.layers.0.mixer.out_proj.weight'.
        norm_eps:           RMSNorm epsilon (1e-5 for NemotronH-30B).
        num_heads:          Number of Mamba2 heads (64 for NemotronH-30B).
        head_dim:           Head dimension (64 for NemotronH-30B).
        n_groups:           Number of SSM groups (8 for NemotronH-30B).
        ssm_state_size:     SSM state size (128 for NemotronH-30B).
        chunk_size:         Chunk size for SSD algorithm (128 for NemotronH-30B).

    Returns:
        Tensor [B, S, hidden_size] -- post-residual output, same dtype as
        'hidden_states'.
    """
    residual = hidden_states
    intermediate_size = num_heads * head_dim  # 4096
    conv_dim = intermediate_size + 2 * n_groups * ssm_state_size  # 6144

    # -------------------------------------------------------------------------
    # 1. Pre-block RMSNorm
    # -------------------------------------------------------------------------
    normed = layer_norm(hidden_states, norm_weight, eps=norm_eps)

    # -------------------------------------------------------------------------
    # 2. in_proj: split into [gate, hidden_states_B_C, dt]
    #    d_mlp = (10304 - 2*4096 - 2*8*128 - 64) // 2 = 0  for NemotronH-30B
    # -------------------------------------------------------------------------
    projected = F.linear(normed, in_proj_weight)  # [B, S, 10304]
    gate = projected[..., :intermediate_size]  # [B, S, 4096]
    hidden_states_B_C = projected[..., intermediate_size : intermediate_size + conv_dim]  # [B, S, 6144]
    dt = projected[..., intermediate_size + conv_dim :]  # [B, S, 64]

    # -------------------------------------------------------------------------
    # 3. Depthwise conv1d (causal, prefill path -- full sequence)
    # -------------------------------------------------------------------------
    batch_size, seq_len, _ = normed.shape
    conv_kernel = conv1d_weight.shape[-1]  # 4
    hBC_t = hidden_states_B_C.transpose(1, 2)  # [B, conv_dim, S]
    hBC_conv = F.conv1d(
        hBC_t,
        conv1d_weight,
        bias=conv1d_bias,
        padding=conv_kernel - 1,
        groups=conv_dim,
    )[
        ..., :seq_len
    ]  # [B, conv_dim, S]
    hBC_conv = F.silu(hBC_conv).transpose(1, 2)  # [B, S, conv_dim]

    # Split into x (hidden_states), B, C
    x = hBC_conv[..., :intermediate_size]  # [B, S, 4096]
    B_vec = hBC_conv[..., intermediate_size : intermediate_size + n_groups * ssm_state_size]  # [B, S, 1024]
    C_vec = hBC_conv[..., intermediate_size + n_groups * ssm_state_size :]  # [B, S, 1024]

    # -------------------------------------------------------------------------
    # 4. SSM transformation -- chunked SSD naive (no cache)
    #    Mirrors modeling_nemotron_h.py torch_forward lines 619-692.
    # -------------------------------------------------------------------------
    A = -torch.exp(A_log.float())  # [num_heads]

    # dt: softplus(dt + dt_bias), clamp(0, inf) is identity for this model
    dt_f = F.softplus(dt + dt_bias)  # [B, S, 64]

    # Reshape and cast to float32
    x_f = x.reshape(batch_size, seq_len, num_heads, head_dim).float()  # [B, S, 64, 64]
    B_f = B_vec.reshape(batch_size, seq_len, n_groups, ssm_state_size).float()  # [B, S, 8, 128]
    C_f = C_vec.reshape(batch_size, seq_len, n_groups, ssm_state_size).float()  # [B, S, 8, 128]

    # Repeat B and C groups to match num_heads: [B, S, 8, 128] -> [B, S, 64, 128]
    reps = num_heads // n_groups  # 8
    B_f = B_f.repeat_interleave(reps, dim=2)  # [B, S, 64, 128]
    C_f = C_f.repeat_interleave(reps, dim=2)  # [B, S, 64, 128]

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    # D_residual: skip connection using padded x (zeros for pad positions)
    D_residual = D[..., None].float() * _pad_tensor_by_size(x_f, pad_size)  # [B, S+pad, 64, 64]

    # Discretize x and A
    x_f = x_f * dt_f[..., None]  # [B, S, 64, 64]  x <- x * dt
    A_dt = A.to(x_f.dtype) * dt_f  # [B, S, 64]      A_dt = A * dt

    # Reshape into chunks
    x_c = _reshape_into_chunks(x_f, pad_size, chunk_size)  # [B, nc, cs, 64, 64]
    A_c = _reshape_into_chunks(A_dt, pad_size, chunk_size)  # [B, nc, cs, 64]
    B_c = _reshape_into_chunks(B_f, pad_size, chunk_size)  # [B, nc, cs, 64, 128]
    C_c = _reshape_into_chunks(C_f, pad_size, chunk_size)  # [B, nc, cs, 64, 128]

    # Permute A: [B, nc, cs, 64] -> [B, 64, nc, cs]
    A_c = A_c.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_c, dim=-1)  # [B, 64, nc, cs]

    # 1. Intra-chunk diagonal blocks
    L = torch.exp(_segment_sum(A_c))  # [B, 64, nc, cs, cs]

    # G: causal attention-weight analog  [B, nc, cs, cs, 64]
    G_inter = C_c[:, :, :, None, :, :] * B_c[:, :, None, :, :, :]  # [B, nc, cs, cs, 64, 128]
    G = G_inter.sum(dim=-1)  # [B, nc, cs, cs, 64]

    # M: gated by causal decay  [B, nc, cs, cs, 64]
    M_inter = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_inter.sum(dim=-1)  # [B, nc, cs, cs, 64]

    # Y_diag: intra-chunk output  [B, nc, cs, 64, 64]
    Y_diag = (M[..., None] * x_c[:, :, None]).sum(dim=3)

    # 2. States for inter-chunk recurrence
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [B, 64, nc, cs]
    B_decay = B_c * decay_states.permute(0, -2, -1, 1)[..., None]  # [B, nc, cs, 64, 128]
    states = (B_decay[..., None, :] * x_c[..., None]).sum(dim=2)  # [B, nc, 64, 64, 128]

    # 3. Inter-chunk SSM recurrence (no previous cache state -> zero init)
    previous_states = torch.zeros_like(states[:, :1])  # [B, 1, 64, 64, 128]
    states = torch.cat([previous_states, states], dim=1)  # [B, nc+1, 64, 64, 128]
    A_cumsum_last = A_cumsum[:, :, :, -1]  # [B, 64, nc]
    decay_chunk = torch.exp(_segment_sum(F.pad(A_cumsum_last, (1, 0))))  # [B, 64, nc+1, nc+1]
    decay_chunk = decay_chunk.transpose(1, 3)  # [B, nc+1, nc+1, 64]
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    # new_states: [B, nc+1, 64, 64, 128]
    states, _ssm_state = new_states[:, :-1], new_states[:, -1]  # [B, nc, 64, 64, 128]

    # 4. State -> output per chunk (off-diagonal blocks)
    state_decay_out = torch.exp(A_cumsum)  # [B, 64, nc, cs]
    C_times_states = C_c[..., None, :] * states[:, :, None, ...]  # [B, nc, cs, 64, 64, 128]
    state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)  # [B, nc, cs, 64]
    Y_off = C_times_states.sum(-1) * state_decay_out_perm[..., None]  # [B, nc, cs, 64, 64]

    # Combine intra + inter chunk outputs
    y = Y_diag + Y_off  # [B, nc, cs, 64, 64]
    y = y.reshape(batch_size, -1, num_heads, head_dim)  # [B, S_pad, 64, 64]

    y = y + D_residual  # add D skip connection
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    y = y.reshape(batch_size, seq_len, -1)  # [B, S, 4096]

    # -------------------------------------------------------------------------
    # 5. MambaRMSNormGated + output projection
    # -------------------------------------------------------------------------
    group_size = intermediate_size // n_groups  # 512
    scan_output = _mamba_rms_norm_gated(y, gate, norm_mixer_weight, eps=norm_eps, group_size=group_size)

    out = F.linear(scan_output.to(hidden_states.dtype), out_proj_weight)  # [B, S, 2688]

    # -------------------------------------------------------------------------
    # 6. Residual connection
    # -------------------------------------------------------------------------
    return residual + out
