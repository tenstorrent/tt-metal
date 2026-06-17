import ttnn

from .weights import Qwen35AttentionWeights


def apply_qkv_projection(hidden_states, weights: Qwen35AttentionWeights):
    """Fused Q+K+V matmul (the gate is projected separately).

    hidden_states: [B, 1, seq_len, hidden_size]  (decode: seq_len is the batch axis)
    returns fused: [B, 1, seq_len, (n_local_heads + 2*n_local_kv_heads) * head_dim]
    laid out as all-Q-heads | all-K-heads | all-V-heads, ready for nlp_create_qkv_heads.
    """
    return ttnn.linear(hidden_states, weights.wqkv)


def apply_gate_projection(hidden_states, weights: Qwen35AttentionWeights):
    """Gate matmul for Qwen3.5 gated attention — a separate projection with n_heads
    heads that the 3-way QKV head op can't produce, so it is split off the fused path.

    returns gate: [B, 1, seq_len, n_local_heads * head_dim]
    """
    return ttnn.linear(hidden_states, weights.wg)


def split_qkv_heads_prefill(xqkv_fused, num_heads, num_kv_heads):
    """Split the fused QKV into per-head q/k/v for prefill via nlp_create_qkv_heads.

    Replaces the manual reshape+transpose. transpose_k_heads=False keeps K in
    [1, num_kv_heads, S, head_dim] (same layout the RMSNorm/RoPE/SDPA path expects);
    the per-head split is numerically identical to reshape(.,(1,S,H,HD)).transpose(1,2).
        q -> [1, num_heads, S, head_dim]
        k, v -> [1, num_kv_heads, S, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def split_qkv_heads_decode(xqkv_fused, num_heads, num_kv_heads):
    """Split the fused QKV into per-head q/k/v for decode via nlp_create_qkv_heads_decode.

    The op emits q [1, B, num_heads, head_dim] and k/v [1, B, num_kv_heads, head_dim]
    in an L1 height-sharded layout. We move all three back to interleaved DRAM so the
    rest of forward_decode (per-head norm, partial RoPE, KV-cache update, SDPA-decode)
    sees the same layout the old reshape produced — i.e. only the head-creation step changes.
        q -> [1, B, num_heads, head_dim] (DRAM)
        k, v -> [1, B, num_kv_heads, head_dim] (DRAM)
    """
    # Blackhole NoC DRAM-read alignment bug (#16667) zeros odd-indexed Q rows in the
    # decode reader when the fused input is DRAM-resident; move it to L1 first (the L1
    # reader is unaffected, and this is a no-op on Wormhole). Mirrors gemma4.
    if xqkv_fused.memory_config().buffer_type == ttnn.BufferType.DRAM:
        xqkv_fused = ttnn.to_memory_config(xqkv_fused, ttnn.L1_MEMORY_CONFIG)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
    return q, k, v
