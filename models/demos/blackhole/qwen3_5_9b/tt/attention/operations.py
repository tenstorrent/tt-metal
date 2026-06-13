import ttnn

from .weights import Qwen35AttentionWeights


def apply_qkvg_projection(
    hidden_states,
    weights: Qwen35AttentionWeights,
):
    """
    hidden_states: [1, 1, seq_len, hidden_size]

    returns:
        q: [1, num_heads, seq_len, head_dim]
        k: [1, num_kv_heads, seq_len, head_dim]
        v: [1, num_kv_heads, seq_len, head_dim]
        gate: [1, num_heads, seq_len, head_dim]
    """
    # q, k, v, gate projections
    q = ttnn.linear(hidden_states, weights.wq)
    gate = ttnn.linear(hidden_states, weights.wg)
    k = ttnn.linear(hidden_states, weights.wk)
    v = ttnn.linear(hidden_states, weights.wv)

    return q, k, v, gate
