# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Chunkwise parallel delta rule for GDN prefill.

Ported from origin/qwen9b-p150:models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py
and adapted for 4-device TP mesh operation.

Tensor layout convention (FLA style):
  q, k: [BH, T, K]   (batch*heads, time, key_dim)
  v:    [BH, T, V]   (batch*heads, time, value_dim)
  beta: [BH, T, 1]   (batch*heads, time, 1)
  g:    [BH, T]       (batch*heads, time)
  state:[BH, K, V]   (batch*heads, key_dim, value_dim)
"""

import torch

import ttnn

_TILE = 32
_EYE_CACHE = {}


def _chunk_eye(size):
    """Return a cached [size, size] float32 identity matrix on CPU."""
    if size not in _EYE_CACHE:
        _EYE_CACHE[size] = torch.eye(size, dtype=torch.float32)
    return _EYE_CACHE[size]


def _create_triu_ones(size, device, dtype=ttnn.float32, memory_config=None):
    """Upper triangular ones matrix [size, size] on device."""
    ones = ttnn.ones(
        shape=(size, size), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    return ttnn.triu(ones, diagonal=0, memory_config=memory_config)


def _create_tril_ones(size, device, dtype=ttnn.float32, memory_config=None):
    """Lower triangular ones matrix [size, size] on device."""
    ones = ttnn.ones(
        shape=(size, size), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    return ttnn.tril(ones, diagonal=0, memory_config=memory_config)


def create_chunk_masks(chunk_size, device):
    """Pre-create mask matrices needed by chunk_gated_delta_rule.

    Call once during model init and pass as `cached_masks` to avoid
    recreating on every forward call (48 layers x every prefill).

    Returns dict with keys: triu_ones, tril_mask, lower_causal
    """
    triu_ones = _create_triu_ones(chunk_size, device, dtype=ttnn.float32)
    triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size])
    tril_mask = _create_tril_ones(chunk_size, device, dtype=ttnn.float32)
    tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size])
    lower_causal = _create_tril_ones(chunk_size, device, dtype=ttnn.float32)
    return {"triu_ones": triu_ones, "tril_mask": tril_mask, "lower_causal": lower_causal}


def l2_norm(x, dim=-1, eps=1e-6):
    """L2 normalize along last dim using TTNN ops."""
    T = x.shape[1] if len(x.shape) >= 3 else x.shape[0]
    mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
    x_sq = ttnn.multiply(x, x, memory_config=mc)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True, memory_config=mc)
    inv_norm = ttnn.rsqrt(ttnn.add(norm_sq, eps, memory_config=mc), memory_config=mc)
    return ttnn.multiply(x, inv_norm, memory_config=mc)


def softplus(x, memory_config=None):
    """softplus(x) = log(1 + exp(x)) using TTNN ops.

    Uses the numerically stable form: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    to avoid overflow for large positive x.
    Note: ttnn.softplus may exist in newer TTNN versions — if so, use it directly.
    """
    mc = memory_config
    try:
        return ttnn.softplus(x, memory_config=mc)
    except (AttributeError, RuntimeError):
        # Fallback: log(1 + exp(x)) via stable formulation
        abs_x = ttnn.abs(x, memory_config=mc)
        neg_abs = ttnn.neg(abs_x, memory_config=mc)
        exp_neg = ttnn.exp(neg_abs, memory_config=mc)
        log1p = ttnn.log1p(exp_neg, memory_config=mc)
        relu_x = ttnn.relu(x, memory_config=mc)
        return ttnn.add(relu_x, log1p, memory_config=mc)


def chunk_gated_delta_rule(
    q,  # [BH, T, K] float32 on mesh
    k,  # [BH, T, K] float32 on mesh
    v,  # [BH, T, V] float32 on mesh
    beta,  # [BH, T, 1] float32 on mesh
    g,  # [BH, T]    float32 on mesh
    chunk_size=64,
    scale=None,
    initial_state=None,  # [BH, K, V] float32 on mesh or None
    mesh_device=None,  # mesh device for Concat/Shard tensor mapping
    cached_masks=None,  # from create_chunk_masks()
):
    """
    Chunked gated delta rule using TTNN ops. Used for prefill on 4-device TP mesh.

    Processes the sequence in chunks of `chunk_size` tokens.
    Within each chunk: batched matmuls (parallel over tokens).
    Across chunks: sequential state propagation (T/chunk_size steps).

    Uses LAPACK batched triangular solve (on CPU) to resolve intra-chunk
    dependencies. Cumsum is computed via matmul with upper-triangular ones matrix.

    NOTE: L2 norm is NOT applied inside this function — it must be done by the caller.

    Args:
        q:  [BH, T, K] float32 on mesh (already L2-normed and scaled externally)
        k:  [BH, T, K] float32 on mesh (already L2-normed externally)
        v:  [BH, T, V] float32 on mesh
        beta: [BH, T, 1] float32 on mesh
        g:  [BH, T] float32 on mesh (log-space decay)
        chunk_size: int
        scale: float or None (if None, uses K**-0.5)
        initial_state: [BH, K, V] float32 on mesh or None
        mesh_device: mesh device for ConcatMeshToTensor/ShardTensorToMesh
        cached_masks: dict from create_chunk_masks() or None

    Returns:
        output: [BH, T, V] float32 on mesh
        final_state: [BH, K, V] float32 on mesh
    """
    # HiFi2 + fp32 accumulation for all chunk matmuls to maintain numerical stability.
    _hifi_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    BH = q.shape[0]
    T = q.shape[1]
    K = q.shape[2]
    V = v.shape[2]

    if scale is None:
        scale = K**-0.5

    q = ttnn.multiply(q, scale, memory_config=None)

    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    L = T + pad_len
    num_chunks = L // chunk_size
    batch = BH * num_chunks

    # beta is already [BH, T, 1], g is [BH, T]
    beta_flat = beta

    if pad_len > 0:
        q = ttnn.concat(
            [
                q,
                ttnn.zeros(
                    [BH, pad_len, K],
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        k = ttnn.concat(
            [
                k,
                ttnn.zeros(
                    [BH, pad_len, K],
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        v = ttnn.concat(
            [
                v,
                ttnn.zeros(
                    [BH, pad_len, V],
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        beta_flat = ttnn.concat(
            [
                beta_flat,
                ttnn.zeros(
                    [BH, pad_len, 1],
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        g_3d = ttnn.reshape(g, [BH, T, 1])
        ttnn.deallocate(g)
        g_3d = ttnn.concat(
            [
                g_3d,
                ttnn.zeros(
                    [BH, pad_len, 1],
                    device=mesh_device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=None,
                ),
            ],
            dim=1,
            memory_config=None,
        )
        g = ttnn.reshape(g_3d, [BH, L])
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])
    else:
        beta_flat = ttnn.reshape(beta_flat, [BH, L, 1])

    v_beta = ttnn.multiply(v, beta_flat, memory_config=None)
    k_beta = ttnn.multiply(k, beta_flat, memory_config=None)
    del beta_flat

    q_c = ttnn.reshape(q, [batch, chunk_size, K], memory_config=None)
    k_c = ttnn.reshape(k, [batch, chunk_size, K], memory_config=None)
    del v
    k_beta_c = ttnn.reshape(k_beta, [batch, chunk_size, K], memory_config=None)
    v_beta_c = ttnn.reshape(v_beta, [batch, chunk_size, V], memory_config=None)
    g_c = ttnn.reshape(g, [batch, chunk_size], memory_config=None)
    del q, v_beta, k_beta

    # Use cached masks if available, otherwise create them
    if cached_masks is not None:
        triu_ones = cached_masks["triu_ones"]
        tril_mask = cached_masks["tril_mask"]
    else:
        triu_ones = _create_triu_ones(chunk_size, mesh_device, dtype=ttnn.float32)
        triu_ones = ttnn.reshape(triu_ones, [1, chunk_size, chunk_size])
        tril_mask = _create_tril_ones(chunk_size, mesh_device, dtype=ttnn.float32)
        tril_mask = ttnn.reshape(tril_mask, [1, chunk_size, chunk_size])

    g_c_3d = ttnn.reshape(g_c, [batch, 1, chunk_size], memory_config=None)
    decay = ttnn.reshape(
        ttnn.matmul(g_c_3d, triu_ones, memory_config=None),
        [batch, chunk_size],
        memory_config=None,
    )

    # Per-chunk normalization: subtract first element so cumsum starts at 0.
    # This keeps values in [-decay_range, 0] instead of starting at a large negative.
    # Sites 2 and 3 use decay_raw (un-normalized) since they need absolute decay
    # values for correct state interaction.
    decay_offset = decay[:, 0:1]  # [batch, 1]
    decay_raw = decay  # save raw cumsum before normalization
    decay = ttnn.subtract(decay_raw, decay_offset, memory_config=None)  # normalized: starts at 0
    ttnn.deallocate(decay_offset)

    # Site 2: key scaling needs raw (absolute) decay for state correction term.
    decay_exp = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_raw, min=-20.0, max=0.0), memory_config=None),
        [batch, chunk_size, 1],
        memory_config=None,
    )

    # For large chunk sizes, force DRAM placement for intermediate tensors.
    _cmc = ttnn.DRAM_MEMORY_CONFIG if chunk_size > 64 else None

    decay_col = ttnn.reshape(decay, [batch, chunk_size, 1], memory_config=None)
    decay_row = ttnn.reshape(decay, [batch, 1, chunk_size], memory_config=None)
    L_diff = ttnn.subtract(decay_col, decay_row, memory_config=_cmc)
    del decay_col, decay_row

    # Clamp before exp to prevent overflow/underflow
    L_diff_masked = ttnn.multiply(L_diff, tril_mask, memory_config=_cmc)
    ttnn.deallocate(L_diff)
    L_diff_clamped = ttnn.clip(L_diff_masked, min=-20.0, max=0.0)
    ttnn.deallocate(L_diff_masked)
    L_mask = ttnn.multiply(ttnn.exp(L_diff_clamped, memory_config=_cmc), tril_mask, memory_config=_cmc)
    ttnn.deallocate(L_diff_clamped)

    del k
    k_c = ttnn.move(k_c)
    k_c_t = ttnn.transpose(k_c, 1, 2, memory_config=_cmc)
    kk = ttnn.matmul(k_beta_c, k_c_t, memory_config=_cmc, compute_kernel_config=_hifi_cfg)
    ttnn.deallocate(k_c_t)

    # Compute -(kk * L_mask) = lower-triangular correction matrix including diagonal.
    attn_raw = ttnn.neg(ttnn.multiply(kk, L_mask, memory_config=_cmc), memory_config=_cmc)
    ttnn.deallocate(kk)

    # Forward substitution: compute R = (I - A)^{-1} where A is lower triangular.
    # Uses LAPACK batched triangular solve on CPU.
    # For multi-device mesh: round-trip via ConcatMeshToTensor / ShardTensorToMesh
    # because each device holds different head shards.
    A = ttnn.to_torch(attn_raw, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    A = A.float()
    ttnn.deallocate(attn_raw)
    eye = _chunk_eye(chunk_size)
    I_minus_A = eye - A
    del A
    attn_cpu = torch.linalg.solve_triangular(I_minus_A, eye.expand_as(I_minus_A), upper=False)
    del I_minus_A
    attn = ttnn.from_torch(
        attn_cpu,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        memory_config=_cmc,
    )
    del attn_cpu

    v_corrected = ttnn.matmul(attn, v_beta_c, memory_config=_cmc, compute_kernel_config=_hifi_cfg)
    del v_beta_c

    k_beta_decay = ttnn.multiply(k_beta_c, decay_exp, memory_config=_cmc)
    k_cumdecay = ttnn.matmul(attn, k_beta_decay, memory_config=_cmc, compute_kernel_config=_hifi_cfg)

    q_c_4d = ttnn.reshape(q_c, [BH, num_chunks, chunk_size, K], memory_config=None)
    q_c_4d = ttnn.to_layout(q_c_4d, ttnn.TILE_LAYOUT, memory_config=None)
    k_c_4d = ttnn.reshape(k_c, [BH, num_chunks, chunk_size, K], memory_config=None)
    k_c_4d = ttnn.to_layout(k_c_4d, ttnn.TILE_LAYOUT, memory_config=None)
    v_cor_4d = ttnn.reshape(v_corrected, [BH, num_chunks, chunk_size, V], memory_config=None)
    v_cor_4d = ttnn.to_layout(v_cor_4d, ttnn.TILE_LAYOUT, memory_config=None)
    k_cum_4d = ttnn.reshape(k_cumdecay, [BH, num_chunks, chunk_size, K], memory_config=None)
    k_cum_4d = ttnn.to_layout(k_cum_4d, ttnn.TILE_LAYOUT, memory_config=None)
    L_mask_4d = ttnn.reshape(L_mask, [BH, num_chunks, chunk_size, chunk_size], memory_config=None)
    L_mask_4d = ttnn.to_layout(L_mask_4d, ttnn.TILE_LAYOUT, memory_config=None)
    decay_3d = ttnn.reshape(decay, [BH, num_chunks, chunk_size], memory_config=None)
    decay_raw_3d = ttnn.reshape(decay_raw, [BH, num_chunks, chunk_size], memory_config=None)

    decay_last_raw = ttnn.reshape(
        ttnn.sum(g_c, dim=-1, memory_config=None),
        [BH, num_chunks, 1],
        memory_config=None,
    )
    # decay_last in normalized coordinates = last element of normalized decay per chunk
    decay_last_normalized = decay_3d[:, :, -1:]  # [BH, num_chunks, 1]
    decay_last_normalized = ttnn.reshape(decay_last_normalized, [BH, num_chunks, 1], memory_config=None)

    if cached_masks is not None:
        lower_causal = cached_masks["lower_causal"]
    else:
        lower_causal = _create_tril_ones(chunk_size, mesh_device, dtype=ttnn.float32)

    S = ttnn.zeros(
        [BH, K, V],
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=None,
    )
    if initial_state is not None:
        S = ttnn.typecast(
            ttnn.reshape(initial_state, [BH, K, V], memory_config=None),
            ttnn.float32,
            memory_config=None,
        )

    # Pre-compute state-independent ops as batched 4D tensors (hoisted from the loop).

    # 1. Batched qk: [BH, num_chunks, cs, K] @ [BH, num_chunks, K, cs] -> [BH, num_chunks, cs, cs]
    k_c_4d_t = ttnn.transpose(k_c_4d, 2, 3, memory_config=_cmc)
    qk_4d = ttnn.matmul(q_c_4d, k_c_4d_t, memory_config=_cmc, compute_kernel_config=_hifi_cfg)
    ttnn.deallocate(k_c_4d_t)

    # 2. Batched intra_attn: qk * L_mask * lower_causal
    lower_causal_4d = ttnn.reshape(lower_causal, [1, 1, chunk_size, chunk_size], memory_config=None)
    combined_mask_4d = ttnn.multiply(L_mask_4d, lower_causal_4d, memory_config=_cmc)
    intra_attn_4d = ttnn.multiply(qk_4d, combined_mask_4d, memory_config=_cmc)
    ttnn.deallocate(qk_4d)
    ttnn.deallocate(combined_mask_4d)

    # 3. Batched q_decay: q * exp(clip(decay_raw)) for inter-chunk state query
    decay_raw_exp_4d = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_raw_3d, min=-20.0, max=0.0), memory_config=_cmc),
        [BH, num_chunks, chunk_size, 1],
        memory_config=_cmc,
    )
    q_decay_4d = ttnn.multiply(q_c_4d, decay_raw_exp_4d, memory_config=_cmc)
    ttnn.deallocate(decay_raw_exp_4d)

    # 4. Batched k_decay_t: k * exp(clip(decay_last - decay)) then transpose
    decay_last_norm_4d = ttnn.reshape(decay_last_normalized, [BH, num_chunks, 1], memory_config=_cmc)
    decay_diff_3d = ttnn.subtract(decay_last_norm_4d, decay_3d, memory_config=_cmc)
    decay_diff_exp_4d = ttnn.reshape(
        ttnn.exp(ttnn.clip(decay_diff_3d, min=-20.0, max=0.0), memory_config=_cmc),
        [BH, num_chunks, chunk_size, 1],
        memory_config=_cmc,
    )
    k_decay_4d = ttnn.multiply(k_c_4d, decay_diff_exp_4d, memory_config=_cmc)
    ttnn.deallocate(decay_diff_exp_4d)
    k_decay_t_4d = ttnn.transpose(k_decay_4d, 2, 3, memory_config=_cmc)
    ttnn.deallocate(k_decay_4d)

    # 5. Batched state decay factors: exp(clip(decay_last_raw))
    dl_exp_3d = ttnn.exp(ttnn.clip(decay_last_raw, min=-20.0, max=0.0), memory_config=_cmc)

    outputs = []
    for i in range(num_chunks):
        v_i = v_cor_4d[:, i]
        k_cum_i = k_cum_4d[:, i]
        intra_attn_i = intra_attn_4d[:, i]
        q_decay_i = q_decay_4d[:, i]
        k_decay_t_i = k_decay_t_4d[:, i]

        # v_prime = k_cumdecay @ S (state-dependent)
        v_prime = ttnn.matmul(k_cum_i, S, memory_config=_cmc, compute_kernel_config=_hifi_cfg)
        v_new = ttnn.subtract(v_i, v_prime, memory_config=_cmc)

        # o_inter = q_decay @ S (state-dependent)
        o_inter = ttnn.matmul(q_decay_i, S, memory_config=_cmc, compute_kernel_config=_hifi_cfg)

        # intra_v = intra_attn @ v_new (depends on v_new which depends on S)
        intra_v = ttnn.matmul(intra_attn_i, v_new, memory_config=_cmc, compute_kernel_config=_hifi_cfg)

        o_i = ttnn.add(o_inter, intra_v, memory_config=_cmc)
        outputs.append(ttnn.reshape(o_i, [BH, 1, chunk_size, V], memory_config=_cmc))

        # State update: S = S * decay_factor + k_decay_t @ v_new
        dl_i_exp = dl_exp_3d[:, i]
        S = ttnn.multiply(
            S,
            ttnn.reshape(dl_i_exp, [BH, 1, 1], memory_config=_cmc),
            memory_config=_cmc,
        )
        state_update = ttnn.matmul(k_decay_t_i, v_new, memory_config=_cmc, compute_kernel_config=_hifi_cfg)
        S = ttnn.add(S, state_update, memory_config=_cmc)

    o = ttnn.concat(outputs, dim=1, memory_config=None)
    # o shape: [BH, num_chunks, chunk_size, V]
    # Merge chunk dims to get [BH, L, V], then trim padding if needed
    o = ttnn.reshape(o, [BH, L, V], memory_config=None)

    if pad_len > 0:
        o = o[:, :T, :]
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=None)

    final_state = S
    return o, final_state
