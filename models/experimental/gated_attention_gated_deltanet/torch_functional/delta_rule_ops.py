# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
    Token-by-token recurrent gated delta rule -- used for decode (typically T=1).

    High-level idea
    ---------------
    We maintain a per-head associative memory (the "state") of shape [K, V]:

        h in R^{K x V}

    For each timestep t (with log-decay g_t, write strength beta_t):

      1. Decay  :  h <- h * exp(g_t)                     (forget old info)
      2. Read   :  v_read_t = h^T k_t                    (what h currently "remembers" for key k_t)
      3. Delta  :  delta_t  = (v_t - v_read_t) * beta_t  (how much to correct)
      4. Write  :  h <- h + k_t (outer) delta_t          (rank-1 update)
      5. Query  :  o_t = h^T q_t                         (read with the query)

    Unlike the chunked version, this path is purely sequential -- no
    matmul tricks, no masks. That makes it the reference implementation
    and also what we actually want at decode time when T is small (often 1).

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
        output:      [B, T, H, V]
        final_state: [B, H, K, V] or None
    """
    # ==================================================================
    # Step 1: Prepare inputs -- normalize, transpose, scale.
    # ------------------------------------------------------------------
    # - Optional L2-norm on q/k (stabilizes the dot-products).
    # - Transpose [B, T, H, D] -> [B, H, T, D] so heads are the outer
    #   dim (matches the state layout [B, H, K, V]).
    # - Apply the attention scale to q once, up-front.
    # - Promote everything to fp32 for accumulation precision.
    # ==================================================================
    if use_qk_l2norm:
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    #  Transpose [B, T, H, D] -> [B, H, T, D]
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (q, k, v, beta, g)]

    B, H, T, K = k.shape
    V = v.shape[-1]

    if scale is None:
        scale = K**-0.5
    q = q * scale

    # ==================================================================
    # Step 2: Allocate outputs and initialize the recurrent state h.
    # ------------------------------------------------------------------
    # h has shape [B, H, K, V] -- one K-by-V associative memory per
    # (batch, head). Start from zero unless the caller passed one in
    # (e.g. from the previous decode step).
    # ==================================================================
    o = torch.zeros(B, H, T, V, device=v.device, dtype=v.dtype)
    h = torch.zeros(B, H, K, V, device=v.device, dtype=v.dtype)
    if initial_state is not None:
        h = initial_state.to(torch.float32)

    # ==================================================================
    # Step 3: Walk through time, running the 5-step delta rule per token.
    # ==================================================================
    for i in range(T):
        # Slice out the t-th token for q/k/v/beta/g.
        b_q = q[:, :, i]  # [B, H, K]
        b_k = k[:, :, i]  # [B, H, K]
        b_v = v[:, :, i].clone()  # [B, H, V]
        b_beta = beta[:, :, i]  # [B, H]

        # (1) Decay the state: h <- h * exp(g_t).
        #     g is broadcast across the K and V axes via [..., None, None].
        # h: [B, H, K, V], g[:, :, i].exp()[..., None, None]: [B, H, 1, 1]
        h = h.clone() * g[:, :, i].exp()[..., None, None]

        # (2) Read from state: v_read = h^T k_t.
        #     (h * k_t[..., None]).sum(-2) contracts over the K axis,
        #     giving the current value stored under key k_t. Subtract it
        #     from v_t to get the raw delta before beta.
        # h.clone(): [B, H, K, V], b_k[..., None]: [B, H, K, 1]
        # (h.clone() * b_k[..., None]).sum(-2): [B, H, V]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)

        # (3) Scale delta by beta (the write strength / gate).
        # b_v: [B, H, V], b_beta[..., None]: [B, H, 1]
        b_v = b_v * b_beta[..., None]

        # (4) Write to state via rank-1 outer product: h <- h + k_t (outer) delta_t.
        # b_k.unsqueeze(-1): [B, H, K, 1], b_v.unsqueeze(-2): [B, H, 1, V]
        # Outer product: [B, H, K, V]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)

        # (5) Query the (just-updated) state with q_t to produce o_t.
        #     Einsum does o[..., m] = sum_d q[..., d] * h[..., d, m].
        # b_q: [B, H, K], h: [B, H, K, V], output: [B, H, V]
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)

    final_state = h if output_final_state else None

    # ==================================================================
    # Step 4: Restore output layout [B, H, T, V] -> [B, T, H, V].
    # ==================================================================
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
    Chunked gated delta rule -- used for prefill (processing full sequences).

    High-level idea
    ---------------
    The recurrent (token-by-token) delta rule has a state S of shape [K, V]:

        S_t = diag(exp(g_t)) * S_{t-1} + k_t (outer) (v_t - S_{t-1}^T k_t) * beta_t
        o_t = S_t^T q_t

    Running this token-by-token is slow. Instead we split the sequence into
    chunks of size `C` and:

      * Inside each chunk we use dense matmuls (parallel across tokens).
      * Between chunks we carry S forward sequentially (only one state
        update per chunk).

    The tricky bit is that inside a chunk the writes and reads are still
    causal and interact, so we solve a small triangular system
    (Woodbury / forward-substitution) to get "corrected" values/keys that
    can then be applied as plain matmuls.

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
        output:      [B, T, H, V]
        final_state: [B, H, K, V] or None
    """
    # ==================================================================
    # Step 1: Prepare inputs -- normalize, transpose, pad, and scale.
    # ------------------------------------------------------------------
    # - Optional L2-norm on q/k (stabilizes the dot-products).
    # - Transpose [B, T, H, D] -> [B, H, T, D] so heads are the outer
    #   dim (matches the state layout [B, H, K, V]).
    # - Pad T up to a multiple of chunk_size.
    # - Apply the attention scale to q once, up-front.
    # - Promote everything to fp32 for accumulation precision.
    # ==================================================================
    if use_qk_l2norm:  # True
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)

    if scale is None:
        scale = q.shape[-1] ** -0.5
    # Transpose [B, T, H, D] -> [B, H, T, D]
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

    # Pre-weight k and v by the write strength beta (used by the delta write).
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]

    # ==================================================================
    # Step 2: Reshape into per-chunk views and compute cumulative decay.
    # ------------------------------------------------------------------
    # Every tensor gets a new "num_chunks" axis so we can operate on all
    # chunks in parallel: [B, H, T, D] -> [B, H, num_chunks, C, D].
    # ==================================================================
    q_c = q.reshape(B, H, -1, chunk_size, q.shape[-1])
    k_c = k.reshape(B, H, -1, chunk_size, k.shape[-1])
    k_beta_c = k_beta.reshape(B, H, -1, chunk_size, k_beta.shape[-1])
    v_beta_c = v_beta.reshape(B, H, -1, chunk_size, v_beta.shape[-1])

    # Cumulative log-decay within each chunk: decay[..., i] = sum_{j<=i} g_j.
    g_c = g.reshape(B, H, -1, chunk_size)
    # Compute the cumulative sum of the log-space decay g within each chunk.
    # For each token position i in the chunk, decay[..., i] contains the sum of all g_j for j <= i.
    # This gives us the total amount of decay accumulated up to each position in the chunk,
    # which is necessary for properly scaling contributions within the chunk.
    decay = g_c.cumsum(dim=-1)  # [B, H, num_chunks, C]
    decay_exp = decay.exp()[..., None]  # [B, H, num_chunks, C, 1]

    # ==================================================================
    # Step 3: Build the masks / weight matrix used inside every chunk.
    # ------------------------------------------------------------------
    # - mask_upper  : i <= j. Zeros out non-strictly-lower entries when
    #                 resolving intra-chunk dependencies (Step 4).
    # - mask_causal : i <  j. Standard causal mask for attention scores.
    # - L_mask[i,j] : exp(cumdecay[i] - cumdecay[j]) for j <= i, else 0.
    #                 Tells us how much position j's contribution decays
    #                 by the time we read it at position i.
    # ==================================================================
    # torch.triu creates an upper-triangular matrix from the supplied input tensor.
    # Here, torch.ones(...) creates a chunk_size x chunk_size matrix of True (since dtype is bool).
    # torch.triu(..., diagonal=0) sets all elements below the main diagonal to False, keeping the diagonal and above as True.
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    mask_causal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()

    # ==================================================================
    # Step 4: Resolve intra-chunk write-before-read feedback.
    # ------------------------------------------------------------------
    # Inside one chunk each token's effective write depends on earlier
    # tokens in the same chunk (delta-rule writes at step t feed reads
    # at step t+1). We encode this as a strictly-lower matrix A and we
    # actually need attn = (I - A)^{-1}. Because A is strictly lower
    # triangular we do this by forward-substitution.
    #
    # After this step:
    #   v_corrected[c] = "corrected" values each token would write if it
    #                    only saw the state carried in from previous chunks.
    #   k_cumdecay[c]  = beta-weighted keys pre-scaled by the cumulative
    #                    decay, used to read the incoming state S.
    # ==================================================================
    attn_raw = (k_beta_c @ k_c.transpose(-1, -2)) * L_mask
    # Mask out the upper triangular part of the matrix to handle intra-chunk dependencies.
    attn = -attn_raw.masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        # Compute the feedback sum for position i in the chunk.
        prev_rows = attn[..., :i, :i]  # rows [0..i-1], columns [0..i-1]; shape: [B, H, num_chunks, i, i]
        curr_row = attn[..., i, :i]  # row i, columns [0..i-1]; shape: [B, H, num_chunks, i]
        # Sum: for token at position i, the feedback is [curr_row] + sum_j (curr_row[j] * prev_rows[j])
        feedback = (curr_row.unsqueeze(-1) * prev_rows).sum(-2)  # [B, H, num_chunks, i]
        attn[..., i, :i] = curr_row + feedback  # [B, H, num_chunks, i]
    # Add the identity matrix to 'attn' so that it becomes (I - A), where I is the identity matrix.
    # This step is required to later invert (I - A) (or solve the triangular system by forward-substitution),
    # which is mathematically equivalent to handling intra-chunk feedback in the delta rule.
    # In this context, 'attn' encodes the strictly-lower-triangular feedback weights; adding the identity
    # ensures the main diagonal has ones, as required for such recurrent/triangular systems.
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)

    v_corrected = attn @ v_beta_c
    k_cumdecay = attn @ (k_beta_c * decay_exp)

    # ==================================================================
    # Step 5: Walk chunk-by-chunk, propagating the recurrent state S.
    # ------------------------------------------------------------------
    # For each chunk we:
    #   (a) compute the intra-chunk causal attention,
    #   (b) subtract from v_corrected the part already contributed by S
    #       (gives v_new -- what this chunk actually needs to write),
    #   (c) query the incoming state S with decay-weighted queries,
    #   (d) combine (a)+(c) to form the chunk output,
    #   (e) advance S to the end of this chunk.
    # ==================================================================
    S = torch.zeros(B, H, K, V, device=q.device, dtype=q.dtype)
    if initial_state is not None:
        S = initial_state.to(torch.float32)

    num_chunks = L // chunk_size
    o = torch.zeros_like(v_corrected)  # [B, H, num_chunks, C, V]

    for i in range(num_chunks):
        q_i = q_c[:, :, i]  # [B, H, C, K]
        k_i = k_c[:, :, i]  # [B, H, C, K]
        v_i = v_corrected[:, :, i]  # [B, H, C, V]
        decay_i = decay[:, :, i]  # [B, H, C]

        # (a) Intra-chunk causal attention, scaled by the decay mask.
        intra_attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask_causal, 0)

        # (b) Residual value actually written by this chunk.
        v_new = v_i - k_cumdecay[:, :, i] @ S

        # (c) Cross-chunk read from the incoming state S.
        o_inter = (q_i * decay_i[..., None].exp()) @ S

        # (d) Chunk output = cross-chunk read + intra-chunk attention.
        o[:, :, i] = o_inter + intra_attn @ v_new

        # (e) Advance the state to the end of this chunk:
        #     - Decay the old state by exp(total chunk decay).
        #     - Add sum over t of  exp(total_decay - decay_t) * k_t (outer) v_new_t,
        #       which aligns every in-chunk contribution to the end-of-chunk timestep.
        total_decay = decay_i[:, :, -1, None]  # [B, H, 1]
        tail_weight = (total_decay - decay_i).exp()[..., None]  # [B, H, C, 1]
        S = S * total_decay[..., None].exp() + (k_i * tail_weight).transpose(-1, -2) @ v_new

    final_state = S if output_final_state else None

    # ==================================================================
    # Step 6: Un-chunk, un-pad, restore [B, T, H, V] layout.
    # ==================================================================
    o = o.reshape(B, H, -1, V)[:, :, :T]
    o = o.transpose(1, 2).contiguous()
    return o, final_state
