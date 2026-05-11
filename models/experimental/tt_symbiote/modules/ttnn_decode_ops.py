# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN-native decode-step ops for Qwen3.5/3.6 linear attention.

These wrap pieces of Qwen3_5MoeGatedDeltaNet's decode forward in TTNN so the
whole layer's decode path can stay on device when the GDN kernel is active.
Together with the kernel itself, this is what makes trace capture by an outer
generator possible (every op in the decode trace region is TTNN).

Design notes:

- `ttnn_causal_conv1d_update_step` uses a circular-buffer formulation rather
  than a custom tt-lang kernel. The native PyTorch op shifts a [B, D, K] state
  every step (concat + slice + sliding-window conv, K=4 sub-tile). On TTNN
  that geometry is awkward (`ttnn.sum` over a 4-element axis, `ttnn.slice` on
  sub-tile dims). Maintaining K=4 separate [B, D] tile-aligned tensors plus an
  index pointer turns the "shift" into pointer rebinding (free) and the
  "weighted sum" into K elementwise mults + (K-1) adds — every op is
  tile-aligned and uses native TTNN primitives.

- `ttnn_gated_rms_norm` mirrors `Qwen3_5MoeRMSNormGated.forward`: y = weight *
  rms_norm(x) * silu(gate). Implemented as `ttnn.rms_norm(x, weight, eps) *
  ttnn.silu(gate)`.

- `ttnn_compute_g_beta` replaces the `g = -A_log.exp() * softplus(a + dt_bias)`
  / `beta = sigmoid(b)` PyTorch elementwise chain.
"""


import ttnn


# ============================================================
# Weight upload helpers (one-time, called from from_torch / move_weights)
# ============================================================


def upload_replicated(tensor, mesh_device, dtype=ttnn.bfloat16):
    """Upload a torch tensor to mesh, replicated across all devices."""
    if tensor is None:
        return None
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
    )


def upload_conv1d_weights(conv1d_weight, mesh_device):
    """Split conv1d weight [dim, 1, K] (or [dim, K]) into K [1, dim] TTNN tensors.

    Each tensor is the slice along the kernel axis for one lag, ready for
    elementwise multiplication against a circular-buffer slot.
    """
    w = conv1d_weight
    if w.dim() == 3:
        w = w.squeeze(1)  # [dim, K]
    K = w.shape[-1]
    return [upload_replicated(w[:, k].contiguous().unsqueeze(0), mesh_device) for k in range(K)]


# ============================================================
# Gated RMS norm (TTNN-native equivalent of Qwen3_5MoeRMSNormGated)
# ============================================================


def ttnn_gated_rms_norm(x, gate, weight, eps):
    """Compute Qwen3_5MoeRMSNormGated.forward in TTNN.

    Reference (transformers/qwen3_5_moe):
        h = x.float()
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * rsqrt(var + eps)
        h = weight * h.to(input_dtype)
        out = h * silu(gate.float())
        return out.to(input_dtype)

    Args:
      x      : ttnn.Tensor [..., head_v_dim]
      gate   : ttnn.Tensor same shape as x
      weight : ttnn.Tensor [head_v_dim] tile-padded
      eps    : float
    Returns:
      ttnn.Tensor same shape as x
    """
    normed = ttnn.rms_norm(x, weight=weight, epsilon=eps)
    gated = ttnn.silu(gate)
    return ttnn.mul(normed, gated)


# ============================================================
# Causal conv1d update via circular buffer
# ============================================================


def init_conv_slots_from_torch_state(conv_state_torch, mesh_device):
    """Convert a PyTorch conv_state [B, dim, K] -> shift-register slots.

    Returns a list of K TTNN tensors, each [1, dim], replicated across the
    mesh. slots[0] holds the OLDEST history element, slots[K-1] the NEWEST.

    Subsequent calls to `ttnn_causal_conv1d_update_step` mutate these in
    place via `ttnn.copy` (data shift), preserving the invariant.
    """
    K = conv_state_torch.shape[-1]
    return [
        upload_replicated(
            conv_state_torch[0, :, k].contiguous().unsqueeze(0),  # [1, dim]
            mesh_device,
        )
        for k in range(K)
    ]


def ttnn_causal_conv1d_update_step(x, slots, weights_per_k, bias):
    """One decode step of depthwise causal conv1d + SiLU.

    Mathematically equivalent to torch_causal_conv1d_update on a single token:
        new_state = concat(state[..., 1:], x); copy back to state
        y = silu(F.conv1d(concat(old_state, x), weight) + bias)

    Implementation: explicit data shift via `ttnn.copy`. The slot tensors are
    fixed TTNN buffers; each call performs the same op sequence:
        copy slot[1] -> slot[0]   (drop oldest; everything ages by one)
        copy slot[2] -> slot[1]
        copy slot[3] -> slot[2]
        copy x       -> slot[3]   (newest is now `x`)
        y = w[0]*slot[0] + w[1]*slot[1] + w[2]*slot[2] + w[3]*slot[3] + bias
        y = silu(y)

    The key property: both the op graph and the tensor identities are
    invariant across decode steps, so this is replayable from a captured
    trace. (An earlier circular-buffer-with-pointer-rotation version was
    not, because the slot/weight pairing rotated each step.)

    Args:
      x             : ttnn.Tensor [1, dim] - new token (channels-last single row)
      slots         : list of K ttnn.Tensors [1, dim] - persistent shift register;
                      slots[0] is OLDEST, slots[K-1] is NEWEST after the call.
      weights_per_k : list of K ttnn.Tensors [1, dim] - conv weights per lag;
                      weights_per_k[0] applies to oldest (slots[0]).
      bias          : ttnn.Tensor [1, dim] or None

    Returns:
      y : ttnn.Tensor [1, dim] - silu(conv_output). The slot tensors have been
          mutated in place; caller does not rebind them.
    """
    K = len(slots)
    if len(weights_per_k) != K:
        raise ValueError(f"weights_per_k must have length {K}, got {len(weights_per_k)}")

    # Shift register: slots[0] <- slots[1], ..., slots[K-2] <- slots[K-1], slots[K-1] <- x.
    # ttnn.copy(input_a, input_b) writes input_a's data into input_b in place.
    for k in range(K - 1):
        ttnn.copy(slots[k + 1], slots[k])
    ttnn.copy(x, slots[K - 1])

    # y = sum_{k=0..K-1} weight[k] * slots[k]
    y = ttnn.mul(weights_per_k[0], slots[0])
    for k in range(1, K):
        term = ttnn.mul(weights_per_k[k], slots[k])
        y = ttnn.add(y, term)

    if bias is not None:
        y = ttnn.add(y, bias)
    y = ttnn.silu(y)

    return y


# ============================================================
# Gating: g, beta from in_proj_a, in_proj_b outputs
# ============================================================


def ttnn_compute_g_beta(a, b, A_log_neg_exp, dt_bias):
    """Compute (g, beta) on device.

    PyTorch reference (qwen_attention.py:1675):
        beta = b.sigmoid()
        g    = -A_log.exp() * softplus(a + dt_bias)

    The first term -A_log.exp() is constant across decode steps; pass it
    pre-computed as `A_log_neg_exp` (= -exp(A_log)) to avoid the per-step exp.

    Args:
      a              : ttnn.Tensor [..., num_v_heads]
      b              : ttnn.Tensor [..., num_v_heads]
      A_log_neg_exp  : ttnn.Tensor [num_v_heads]  pre-computed -exp(A_log)
      dt_bias        : ttnn.Tensor [num_v_heads]

    Returns:
      g    : ttnn.Tensor [..., num_v_heads]
      beta : ttnn.Tensor [..., num_v_heads]
    """
    beta = ttnn.sigmoid(b)
    a_shifted = ttnn.add(a, dt_bias)
    sp = ttnn.softplus(a_shifted)
    g = ttnn.mul(A_log_neg_exp, sp)
    return g, beta
