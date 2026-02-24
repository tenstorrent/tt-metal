"""
Pure-torch implementations of the gated delta rule.

Extracted from FLA (Flash Linear Attention) library:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py

These are reference implementations with no CUDA/Triton dependencies.
They serve as the mathematical specification for TTNN conversion.

Tensor layout convention (FLA style):
  q, k: [B, T, H, K]   (batch, time, heads, key_dim)
  v:    [B, T, H, V]   (batch, time, heads, value_dim)
  beta: [B, T, H]      (batch, time, heads)
  g:    [B, T, H]      (batch, time, heads) -- log-space decay
  state:[B, H, K, V]   (batch, heads, key_dim, value_dim)
"""

import torch
import torch.nn.functional as F


def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization along a given dimension."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Token-by-token recurrent gated delta rule. Used for decode (T=1).

    For each timestep t:
      1. Decay the state:  h = h * exp(g_t)
      2. Read from state:  v_read = sum_k(h * k_t)
      3. Compute delta:    delta = (v_t - v_read) * beta_t
      4. Write to state:   h = h + outer(k_t, delta)
      5. Query state:      o_t = h @ q_t

    Args:
        q: [B, T, H, K] query
        k: [B, T, H, K] key
        v: [B, T, H, V] value
        beta: [B, T, H] write strength (sigmoid output)
        g: [B, T, H] log-space decay gate
        scale: attention scale factor, defaults to 1/sqrt(K)
        initial_state: [B, H, K, V] previous recurrent state
        output_final_state: whether to return the final state
        use_qk_l2norm: apply L2 normalization to q, k

    Returns:
        output: [B, T, H, V]
        final_state: [B, H, K, V] or None
    """
    if use_qk_l2norm:
        q = l2_norm(q, dim=-1)
        k = l2_norm(k, dim=-1)

    # Transpose to [B, H, T, D] for head-first processing
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]

    B, H, T, K = k.shape
    V = v.shape[-1]

    if scale is None:
        scale = K**-0.5
    q = q * scale

    o = torch.zeros(B, H, T, V, device=v.device, dtype=v.dtype)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype)
    if initial_state is not None:
        h = initial_state.to(torch.float32)

    for i in range(T):
        b_q = q[:, :, i]  # [B, H, K]
        b_k = k[:, :, i]  # [B, H, K]
        b_v = v[:, :, i].clone()  # [B, H, V]
        b_beta = beta[:, :, i]  # [B, H]

        # 1. Decay the state
        h = h.clone() * g[:, :, i].exp()[..., None, None]

        # 2. Read from state: contract over K dimension
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)

        # 3. Scale by beta (write strength)
        b_v = b_v * b_beta[..., None]

        # 4. Write to state via outer product
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)

        # 5. Query the state
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)

    final_state = h if output_final_state else None

    # Transpose back to [B, T, H, V]
    o = o.transpose(1, 2).contiguous()
    return o, final_state


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunked gated delta rule. Used for prefill (processing full sequences).

    Processes the sequence in chunks of `chunk_size` tokens.
    Within each chunk, uses matrix operations for parallelism.
    Across chunks, propagates the recurrent state sequentially.

    Args:
        q: [B, T, H, K] query
        k: [B, T, H, K] key
        v: [B, T, H, V] value
        g: [B, T, H] log-space decay gate
        beta: [B, T, H] write strength (sigmoid output)
        chunk_size: number of tokens per chunk
        scale: attention scale factor, defaults to 1/sqrt(K)
        initial_state: [B, H, K, V] previous recurrent state
        output_final_state: whether to return the final state
        use_qk_l2norm: apply L2 normalization to q, k

    Returns:
        output: [B, T, H, V]
        final_state: [B, H, K, V] or None
    """
    if use_qk_l2norm:
        q = l2_norm(q, dim=-1)
        k = l2_norm(k, dim=-1)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Transpose to [B, H, T, D]
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]

    T = q.shape[-2]
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))

    B, H, L, K = q.shape
    V = v.shape[-1]
    q = q * scale
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]

    # Upper triangular mask for causal masking within chunks
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)

    # Reshape into chunks: [B, H, num_chunks, chunk_size, D]
    def to_chunks(x):
        return x.reshape(B, H, -1, chunk_size, x.shape[-1])

    q_c = to_chunks(q)
    k_c = to_chunks(k)
    v_c = to_chunks(v)
    k_beta_c = to_chunks(k_beta)
    v_beta_c = to_chunks(v_beta)

    # Cumulative decay within each chunk
    g_c = g.reshape(B, H, -1, chunk_size)
    decay = g_c.cumsum(dim=-1)
    decay_exp = decay.exp()[..., None]

    # Intra-chunk decay mask: L_mask[i,j] = exp(cumsum_g[i] - cumsum_g[j]) for j <= i
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()

    # Woodbury identity: resolve intra-chunk dependencies
    attn = -((k_beta_c @ k_c.transpose(-1, -2)) * L_mask).masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(
            -2
        )
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    # Corrected values and keys after resolving dependencies
    v_corrected = attn @ v_beta_c
    k_cumdecay = attn @ (k_beta_c * decay_exp)

    # Recurrent state propagation across chunks
    S = torch.zeros(B, H, K, V, device=q.device, dtype=q.dtype)
    if initial_state is not None:
        S = initial_state.to(torch.float32)

    num_chunks = L // chunk_size
    o = torch.zeros_like(v_corrected)
    mask_causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)

    for i in range(num_chunks):
        q_i = q_c[:, :, i]
        k_i = k_c[:, :, i]
        v_i = v_corrected[:, :, i]

        # Intra-chunk attention
        intra_attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask_causal, 0)

        # Cross-chunk: subtract state contribution from corrected values
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime

        # Cross-chunk: query the state
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S

        # Combine intra-chunk attention + cross-chunk state
        o[:, :, i] = o_inter + intra_attn @ v_new

        # Update state for next chunk
        S = (
            S * decay[:, :, i, -1, None, None].exp()
            + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    final_state = S if output_final_state else None

    # Reshape back, un-pad, transpose to [B, T, H, V]
    o = o.reshape(B, H, -1, V)[:, :, :T]
    o = o.transpose(1, 2).contiguous()
    return o, final_state
