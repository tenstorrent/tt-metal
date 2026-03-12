# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Optional, Tuple

from ..core.module import TTNNModule

# Tile size used by TTNN matmul (wormhole)
_TILE_H = 32
_TILE_W = 32


def _recurrent_outer_product_program_config(device, K, V):
    """Build MatmulMultiCoreReuseProgramConfig for the outer-product matmul."""
    grid = device.compute_with_storage_grid_size()
    if hasattr(grid, "x"):
        grid_x, grid_y = int(grid.x), int(grid.y)
    else:
        grid_x, grid_y = int(grid[0]), int(grid[1])

    M_tiles = (K + _TILE_H - 1) // _TILE_H
    N_tiles = (V + _TILE_W - 1) // _TILE_W
    K_tiles_inner = max(1, (1 + _TILE_W - 1) // _TILE_W)  # 1

    # N == per_core_N (mandatory)
    per_core_N = N_tiles
    # in0_block_w: inner dim in tiles
    in0_block_w = K_tiles_inner

    # per_core_M must divide M_tiles
    per_core_M = M_tiles
    while per_core_M > 1 and (M_tiles % per_core_M != 0 or (grid_x * grid_y) < (M_tiles // per_core_M)):
        per_core_M -= 1
    if per_core_M < 1:
        per_core_M = 1

    # out_subblock must divide per_core; profiler suggests out_subblock_h * out_subblock_w >= 2
    out_subblock_h = min(2, per_core_M) if per_core_M >= 2 else 1
    out_subblock_w = min(2, per_core_N) if per_core_N >= 2 else 1

    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def _recurrent_read_query_program_config(device, K, V):
    """Build MatmulMultiCoreReuseProgramConfig for read/query matmuls."""
    grid = device.compute_with_storage_grid_size()
    N_tiles = (V + _TILE_W - 1) // _TILE_W
    K_tiles = (K + _TILE_W - 1) // _TILE_W

    per_core_N = N_tiles
    per_core_M = 1
    in0_block_w = K_tiles
    out_subblock_h = 1
    out_subblock_w = min(2, per_core_N) if per_core_N >= 2 else 1

    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def l2_norm_ttnn(x, dim=-1, eps=1e-6):
    """L2 normalization along a given dimension."""
    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    norm_sq = ttnn.sum(x_sq, dim=dim, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    inv_norm = ttnn.rsqrt(
        ttnn.add(norm_sq, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG
    )
    return ttnn.multiply(x, inv_norm, memory_config=ttnn.L1_MEMORY_CONFIG)


def fused_decay_and_write_ttnn(
    h,
    k_t,
    delta,
    decay_t,
    beta_t,
    device=None,
    step_idx=0,
):
    """
    Logical fusion for the recurrent delta rule state update:
    h = decay * h + beta_t * (k_t ⊗ delta)
    """
    B = h.shape[0]
    H = h.shape[1]
    K = h.shape[2]
    V = h.shape[3]

    if step_idx == 0:
        print(
            f"[DEBUG FUSED {step_idx}] Input shapes - h: {h.shape}, k_t: {k_t.shape}, delta: {delta.shape}, decay_t: {decay_t.shape}, beta_t: {beta_t.shape}"
        )
        h_torch = ttnn.to_torch(h)
        k_t_torch = ttnn.to_torch(k_t)
        delta_torch = ttnn.to_torch(delta)
        print(f"[DEBUG FUSED {step_idx}] h before sample (first 3): {h_torch.flatten()[:3].tolist()}")
        print(f"[DEBUG FUSED {step_idx}] k_t sample (first 3): {k_t_torch.flatten()[:3].tolist()}")
        print(f"[DEBUG FUSED {step_idx}] delta sample (first 3): {delta_torch.flatten()[:3].tolist()}")

    # decay: [B, H] -> [B, H, 1, 1]
    decay = ttnn.typecast(decay_t, ttnn.bfloat16)
    decay = ttnn.reshape(decay, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    if step_idx == 0:
        decay_torch = ttnn.to_torch(decay)
        print(f"[DEBUG FUSED {step_idx}] decay sample (first 3): {decay_torch.flatten()[:3].tolist()}")

    # beta: [B, H] -> [B, H, 1, 1]
    beta_expanded = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    if step_idx == 0:
        beta_torch = ttnn.to_torch(beta_expanded)
        print(f"[DEBUG FUSED {step_idx}] beta_expanded sample (first 3): {beta_torch.flatten()[:3].tolist()}")

    # k_t: [B, H, K] -> [B, H, K, 1]
    k_col = ttnn.reshape(k_t, [B, H, K, 1], memory_config=ttnn.L1_MEMORY_CONFIG)

    # delta: [B, H, V] -> [B, H, 1, V]
    d_row = ttnn.reshape(delta, [B, H, 1, V], memory_config=ttnn.L1_MEMORY_CONFIG)

    k_col = ttnn.to_layout(k_col, ttnn.TILE_LAYOUT)
    d_row = ttnn.to_layout(d_row, ttnn.TILE_LAYOUT)
    k_col = ttnn.to_memory_config(k_col, ttnn.L1_MEMORY_CONFIG)
    d_row = ttnn.to_memory_config(d_row, ttnn.L1_MEMORY_CONFIG)

    matmul_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    outer = ttnn.matmul(
        k_col,
        d_row,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=matmul_compute_cfg,
        program_config=None,
    )
    if step_idx == 0:
        print(f"[DEBUG FUSED {step_idx}] After outer product - outer shape: {outer.shape}")
        outer_torch = ttnn.to_torch(outer)
        print(f"[DEBUG FUSED {step_idx}] outer sample (first 3): {outer_torch.flatten()[:3].tolist()}")
        print(
            f"[DEBUG FUSED {step_idx}] outer stats - min: {outer_torch.min().item():.6f}, max: {outer_torch.max().item():.6f}, mean: {outer_torch.mean().item():.6f}"
        )

    # apply beta
    outer = ttnn.multiply(
        outer,
        beta_expanded,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if step_idx == 0:
        outer_torch = ttnn.to_torch(outer)
        print(
            f"[DEBUG FUSED {step_idx}] After beta multiply - outer sample (first 3): {outer_torch.flatten()[:3].tolist()}"
        )
        print(
            f"[DEBUG FUSED {step_idx}] After beta multiply outer stats - min: {outer_torch.min().item():.6f}, max: {outer_torch.max().item():.6f}, mean: {outer_torch.mean().item():.6f}"
        )

    # fused-style update: decay * h + outer
    h = ttnn.multiply(h, decay)
    if step_idx == 0:
        h_torch = ttnn.to_torch(h)
        print(f"[DEBUG FUSED {step_idx}] After decay multiply - h sample (first 3): {h_torch.flatten()[:3].tolist()}")
        print(
            f"[DEBUG FUSED {step_idx}] After decay multiply h stats - min: {h_torch.min().item():.6f}, max: {h_torch.max().item():.6f}, mean: {h_torch.mean().item():.6f}"
        )

    h = ttnn.add(h, outer)
    if step_idx == 0:
        h_torch = ttnn.to_torch(h)
        print(f"[DEBUG FUSED {step_idx}] After add - h sample (first 3): {h_torch.flatten()[:3].tolist()}")
        print(
            f"[DEBUG FUSED {step_idx}] After add h stats - min: {h_torch.min().item():.6f}, max: {h_torch.max().item():.6f}, mean: {h_torch.mean().item():.6f}"
        )

    return h


def recurrent_delta_rule_step_ttnn(
    q_t,
    k_t,
    v_t,
    beta_t,
    decay_t,
    h,
    seq_len=None,
    device=None,
    step_idx=0,
):
    """Recurrent delta rule step using TTNN ops."""
    B = q_t.shape[0]
    H = q_t.shape[1]
    K = q_t.shape[2]
    V = v_t.shape[2]

    if step_idx == 0 or (seq_len is not None and step_idx == seq_len - 1):
        print(
            f"[DEBUG STEP {step_idx}] Input shapes - q_t: {q_t.shape}, k_t: {k_t.shape}, v_t: {v_t.shape}, h: {h.shape}"
        )
        q_t_torch = ttnn.to_torch(q_t)
        k_t_torch = ttnn.to_torch(k_t)
        v_t_torch = ttnn.to_torch(v_t)
        h_torch = ttnn.to_torch(h)
        print(f"[DEBUG STEP {step_idx}] q_t sample (first 3): {q_t_torch.flatten()[:3].tolist()}")
        print(f"[DEBUG STEP {step_idx}] k_t sample (first 3): {k_t_torch.flatten()[:3].tolist()}")
        print(f"[DEBUG STEP {step_idx}] v_t sample (first 3): {v_t_torch.flatten()[:3].tolist()}")
        print(f"[DEBUG STEP {step_idx}] h sample (first 3): {h_torch.flatten()[:3].tolist()}")
        print(
            f"[DEBUG STEP {step_idx}] h stats - min: {h_torch.min().item():.6f}, max: {h_torch.max().item():.6f}, mean: {h_torch.mean().item():.6f}"
        )

    h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
    h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)

    read_query_compute_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    read_query_prog_cfg = None
    if device is not None:
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(device, K, V)
        except Exception:
            pass

    k_row = ttnn.reshape(k_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    k_row = ttnn.to_layout(k_row, ttnn.TILE_LAYOUT)
    k_row = ttnn.to_memory_config(k_row, ttnn.L1_MEMORY_CONFIG)

    # Read from state: k_row @ h (contract over K dimension)
    if read_query_prog_cfg:
        v_prime = ttnn.matmul(
            k_row,
            h,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=read_query_compute_cfg,
            program_config=read_query_prog_cfg,
        )
    else:
        v_prime = ttnn.matmul(
            k_row,
            h,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=read_query_compute_cfg,
        )

    v_prime = ttnn.reshape(v_prime, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    if step_idx == 0 or (seq_len is not None and step_idx == seq_len - 1):
        print(f"[DEBUG STEP {step_idx}] After read - v_prime shape: {v_prime.shape}")
        v_prime_torch = ttnn.to_torch(v_prime)
        print(f"[DEBUG STEP {step_idx}] v_prime sample (first 3): {v_prime_torch.flatten()[:3].tolist()}")
        print(
            f"[DEBUG STEP {step_idx}] v_prime stats - min: {v_prime_torch.min().item():.6f}, max: {v_prime_torch.max().item():.6f}, mean: {v_prime_torch.mean().item():.6f}"
        )

    # Compute delta = v_t - v_prime
    delta = ttnn.subtract(v_t, v_prime, memory_config=ttnn.L1_MEMORY_CONFIG)
    if step_idx == 0 or (seq_len is not None and step_idx == seq_len - 1):
        print(f"[DEBUG STEP {step_idx}] After delta computation - delta shape: {delta.shape}")
        delta_torch = ttnn.to_torch(delta)
        print(f"[DEBUG STEP {step_idx}] delta sample (first 3): {delta_torch.flatten()[:3].tolist()}")
        print(
            f"[DEBUG STEP {step_idx}] delta stats - min: {delta_torch.min().item():.6f}, max: {delta_torch.max().item():.6f}, mean: {delta_torch.mean().item():.6f}"
        )

    # Write to state
    h = fused_decay_and_write_ttnn(h, k_t, delta, decay_t, beta_t, device=device, step_idx=step_idx)
    if step_idx == 0 or (seq_len is not None and step_idx == seq_len - 1):
        print(f"[DEBUG STEP {step_idx}] After write - h shape: {h.shape}")
        h_torch = ttnn.to_torch(h)
        print(f"[DEBUG STEP {step_idx}] h after write sample (first 3): {h_torch.flatten()[:3].tolist()}")
        print(
            f"[DEBUG STEP {step_idx}] h after write stats - min: {h_torch.min().item():.6f}, max: {h_torch.max().item():.6f}, mean: {h_torch.mean().item():.6f}"
        )

    # Read from state: q_row @ h (contract over K dimension)
    q_row = ttnn.reshape(q_t, [B, H, 1, K], memory_config=ttnn.L1_MEMORY_CONFIG)
    q_row = ttnn.to_layout(q_row, ttnn.TILE_LAYOUT)
    q_row = ttnn.to_memory_config(q_row, ttnn.L1_MEMORY_CONFIG)

    if read_query_prog_cfg:
        o_t = ttnn.matmul(
            q_row,
            h,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=read_query_compute_cfg,
            program_config=read_query_prog_cfg,
        )
    else:
        o_t = ttnn.matmul(
            q_row,
            h,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=read_query_compute_cfg,
        )

    o_t = ttnn.reshape(o_t, [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)
    if step_idx == 0 or (seq_len is not None and step_idx == seq_len - 1):
        print(f"[DEBUG STEP {step_idx}] Final o_t shape: {o_t.shape}")
        o_t_torch = ttnn.to_torch(o_t)
        print(f"[DEBUG STEP {step_idx}] o_t sample (first 3): {o_t_torch.flatten()[:3].tolist()}")
        print(
            f"[DEBUG STEP {step_idx}] o_t stats - min: {o_t_torch.min().item():.6f}, max: {o_t_torch.max().item():.6f}, mean: {o_t_torch.mean().item():.6f}"
        )

    return o_t, h


def recurrent_gated_delta_rule_ttnn(
    q,
    k,
    v,
    beta,
    g,
    initial_state=None,
    device=None,
):
    """Recurrent gated delta rule implementation."""
    B, T, H, K = q.shape
    V = v.shape[3]

    print(f"[DEBUG RECURRENT] Starting recurrent delta rule - B: {B}, T: {T}, H: {H}, K: {K}, V: {V}")

    # Initialize state
    h = ttnn.zeros(
        [B, H, K, V], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    if initial_state is not None:
        h = ttnn.typecast(
            ttnn.reshape(initial_state, [B, H, K, V], memory_config=ttnn.L1_MEMORY_CONFIG),
            ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    outputs = []
    for t in range(T):
        if t == 0 or t == T - 1:  # Debug first and last step
            print(f"[DEBUG RECURRENT] Processing timestep {t}/{T}")
        q_t = ttnn.reshape(q[:, t, :, :], [B, H, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_t = ttnn.reshape(k[:, t, :, :], [B, H, K], memory_config=ttnn.L1_MEMORY_CONFIG)
        v_t = ttnn.reshape(v[:, t, :, :], [B, H, V], memory_config=ttnn.L1_MEMORY_CONFIG)
        beta_t = ttnn.reshape(beta[:, t, :], [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)
        decay_t = ttnn.reshape(g[:, t, :], [B, H], memory_config=ttnn.L1_MEMORY_CONFIG)

        o_t, h = recurrent_delta_rule_step_ttnn(q_t, k_t, v_t, beta_t, decay_t, h, seq_len=T, device=device, step_idx=t)
        outputs.append(ttnn.reshape(o_t, [B, 1, H, V], memory_config=ttnn.L1_MEMORY_CONFIG))

    o = ttnn.concat(outputs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG RECURRENT] Final output shape: {o.shape}, final state shape: {h.shape}")
    o_torch = ttnn.to_torch(o)
    h_torch = ttnn.to_torch(h)
    print(f"[DEBUG RECURRENT] Final output sample (first 5): {o_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG RECURRENT] Final output stats - min: {o_torch.min().item():.6f}, max: {o_torch.max().item():.6f}, mean: {o_torch.mean().item():.6f}"
    )
    print(f"[DEBUG RECURRENT] Final state sample (first 5): {h_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG RECURRENT] Final state stats - min: {h_torch.min().item():.6f}, max: {h_torch.max().item():.6f}, mean: {h_torch.mean().item():.6f}"
    )
    return o, h


def chunk_gated_delta_rule_ttnn(
    q,
    k,
    v,
    beta,
    g,
    chunk_size=64,
    initial_state=None,
    device=None,
):
    """Chunked gated delta rule implementation."""
    # This would contain the chunked implementation
    # For brevity, using recurrent as fallback
    return recurrent_gated_delta_rule_ttnn(q, k, v, beta, g, initial_state, device)


def rms_norm_ttnn(x, weight, eps=1e-6):
    """RMS normalization using TTNN."""
    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    rsqrt = ttnn.rsqrt(ttnn.add(mean_sq, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG)
    normed = ttnn.multiply(x, rsqrt, memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.multiply(normed, weight, memory_config=ttnn.L1_MEMORY_CONFIG)


def rms_norm_gated_ttnn(x, gate, weight, eps=1e-6):
    """RMSNorm with SiLU gating using TTNN ops."""
    print(f"[DEBUG RMS_NORM_GATED] Input x shape: {x.shape}, gate shape: {gate.shape}, weight shape: {weight.shape}")
    x_torch = ttnn.to_torch(x)
    gate_torch = ttnn.to_torch(gate)
    weight_torch = ttnn.to_torch(weight)
    print(f"[DEBUG RMS_NORM_GATED] x sample (first 3): {x_torch.flatten()[:3].tolist()}")
    print(
        f"[DEBUG RMS_NORM_GATED] x stats - min: {x_torch.min().item():.6f}, max: {x_torch.max().item():.6f}, mean: {x_torch.mean().item():.6f}"
    )
    print(f"[DEBUG RMS_NORM_GATED] gate sample (first 3): {gate_torch.flatten()[:3].tolist()}")
    print(f"[DEBUG RMS_NORM_GATED] weight sample (first 3): {weight_torch.flatten()[:3].tolist()}")

    x_sq = ttnn.multiply(x, x, memory_config=ttnn.L1_MEMORY_CONFIG)
    variance = ttnn.mean(x_sq, dim=-1, keepdim=True, memory_config=ttnn.L1_MEMORY_CONFIG)
    variance_torch = ttnn.to_torch(variance)
    print(f"[DEBUG RMS_NORM_GATED] variance sample (first 3): {variance_torch.flatten()[:3].tolist()}")
    print(
        f"[DEBUG RMS_NORM_GATED] variance stats - min: {variance_torch.min().item():.6f}, max: {variance_torch.max().item():.6f}, mean: {variance_torch.mean().item():.6f}"
    )

    inv_rms = ttnn.rsqrt(
        ttnn.add(variance, eps, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.L1_MEMORY_CONFIG
    )
    inv_rms_torch = ttnn.to_torch(inv_rms)
    print(f"[DEBUG RMS_NORM_GATED] inv_rms sample (first 3): {inv_rms_torch.flatten()[:3].tolist()}")
    print(
        f"[DEBUG RMS_NORM_GATED] inv_rms stats - min: {inv_rms_torch.min().item():.6f}, max: {inv_rms_torch.max().item():.6f}, mean: {inv_rms_torch.mean().item():.6f}"
    )

    x_normed = ttnn.multiply(x, inv_rms, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_normed_torch = ttnn.to_torch(x_normed)
    print(
        f"[DEBUG RMS_NORM_GATED] After multiply inv_rms - x_normed sample (first 3): {x_normed_torch.flatten()[:3].tolist()}"
    )
    print(
        f"[DEBUG RMS_NORM_GATED] After multiply inv_rms - x_normed stats - min: {x_normed_torch.min().item():.6f}, max: {x_normed_torch.max().item():.6f}, mean: {x_normed_torch.mean().item():.6f}"
    )

    x_normed = ttnn.multiply(x_normed, weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    x_normed_torch = ttnn.to_torch(x_normed)
    print(
        f"[DEBUG RMS_NORM_GATED] After multiply weight - x_normed sample (first 3): {x_normed_torch.flatten()[:3].tolist()}"
    )
    print(
        f"[DEBUG RMS_NORM_GATED] After multiply weight - x_normed stats - min: {x_normed_torch.min().item():.6f}, max: {x_normed_torch.max().item():.6f}, mean: {x_normed_torch.mean().item():.6f}"
    )

    gate_act = ttnn.silu(gate, memory_config=ttnn.L1_MEMORY_CONFIG)
    gate_act_torch = ttnn.to_torch(gate_act)
    print(f"[DEBUG RMS_NORM_GATED] gate_act (silu) sample (first 3): {gate_act_torch.flatten()[:3].tolist()}")
    print(
        f"[DEBUG RMS_NORM_GATED] gate_act stats - min: {gate_act_torch.min().item():.6f}, max: {gate_act_torch.max().item():.6f}, mean: {gate_act_torch.mean().item():.6f}"
    )

    result = ttnn.multiply(x_normed, gate_act, memory_config=ttnn.L1_MEMORY_CONFIG)
    result_torch = ttnn.to_torch(result)
    print(f"[DEBUG RMS_NORM_GATED] Final result sample (first 3): {result_torch.flatten()[:3].tolist()}")
    print(
        f"[DEBUG RMS_NORM_GATED] Final result stats - min: {result_torch.min().item():.6f}, max: {result_torch.max().item():.6f}, mean: {result_torch.mean().item():.6f}"
    )
    return result


def causal_conv1d_ttnn(x, weight, bias, kernel_size, device):
    """Causal convolution1d using TTNN."""
    B, T, D = x.shape

    # For now, return input as-is (placeholder implementation)
    # In a full implementation, this would perform actual causal conv1d + SiLU
    # The actual implementation should maintain sequence length T
    return x


def gated_deltanet_forward_ttnn(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    a_proj_weight,
    b_proj_weight,
    o_proj_weight,
    q_conv_weight,
    k_conv_weight,
    v_conv_weight,
    q_conv_bias,
    k_conv_bias,
    v_conv_bias,
    A_log,
    dt_bias,
    o_norm_weight,
    g_proj_weight=None,
    num_heads=4,
    num_v_heads=None,
    head_k_dim=256,
    head_v_dim=512,
    conv_kernel_size=4,
    use_gate=True,
    allow_neg_eigval=False,
    norm_eps=1e-5,
    device=None,
    recurrent_state=None,
    mode="recurrent",
    chunk_size=64,
):
    """TTNN forward pass for the Gated DeltaNet layer."""
    if num_v_heads is None:
        num_v_heads = num_heads

    B = hidden_states.shape[0]
    T = hidden_states.shape[1]

    print(f"[DEBUG DELTANET] Input hidden_states shape: {hidden_states.shape}")
    hidden_states_torch = ttnn.to_torch(hidden_states)
    print(f"[DEBUG DELTANET] Input hidden_states sample (first 5): {hidden_states_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] Input hidden_states stats - min: {hidden_states_torch.min().item():.6f}, max: {hidden_states_torch.max().item():.6f}, mean: {hidden_states_torch.mean().item():.6f}"
    )

    # 1. Linear projections
    q = ttnn.linear(hidden_states, q_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(hidden_states, k_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(hidden_states, v_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG DELTANET] After linear proj - q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
    q_torch = ttnn.to_torch(q)
    print(f"[DEBUG DELTANET] After linear proj q sample (first 5): {q_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] After linear proj q stats - min: {q_torch.min().item():.6f}, max: {q_torch.max().item():.6f}, mean: {q_torch.mean().item():.6f}"
    )

    # 2. Causal conv1d + SiLU
    q = causal_conv1d_ttnn(q, q_conv_weight, q_conv_bias, conv_kernel_size, device)
    k = causal_conv1d_ttnn(k, k_conv_weight, k_conv_bias, conv_kernel_size, device)
    v = causal_conv1d_ttnn(v, v_conv_weight, v_conv_bias, conv_kernel_size, device)
    print(f"[DEBUG DELTANET] After causal conv - q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
    q_torch = ttnn.to_torch(q)
    print(f"[DEBUG DELTANET] After causal conv q sample (first 5): {q_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] After causal conv q stats - min: {q_torch.min().item():.6f}, max: {q_torch.max().item():.6f}, mean: {q_torch.mean().item():.6f}"
    )

    # 3. Reshape to multi-head
    q = ttnn.reshape(q, [B, T, num_heads, head_k_dim])
    k = ttnn.reshape(k, [B, T, num_heads, head_k_dim])
    v = ttnn.reshape(v, [B, T, num_v_heads, head_v_dim])
    print(f"[DEBUG DELTANET] After reshape - q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
    q_torch = ttnn.to_torch(q)
    print(f"[DEBUG DELTANET] After reshape q sample (first 5): {q_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] After reshape q stats - min: {q_torch.min().item():.6f}, max: {q_torch.max().item():.6f}, mean: {q_torch.mean().item():.6f}"
    )

    # GVA: repeat q,k
    if num_v_heads > num_heads:
        repeats = num_v_heads // num_heads
        q = ttnn.repeat_interleave(q, repeats, dim=2)
        k = ttnn.repeat_interleave(k, repeats, dim=2)

    # Apply L2 normalization to q and k (matching PyTorch use_qk_l2norm=True)
    q = l2_norm_ttnn(q, dim=-1)
    k = l2_norm_ttnn(k, dim=-1)
    print(f"[DEBUG DELTANET] After L2 norm - q shape: {q.shape}, k shape: {k.shape}")
    q_torch = ttnn.to_torch(q)
    print(f"[DEBUG DELTANET] After L2 norm q sample (first 5): {q_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] After L2 norm q stats - min: {q_torch.min().item():.6f}, max: {q_torch.max().item():.6f}, mean: {q_torch.mean().item():.6f}"
    )

    # 4. Compute beta and g
    beta = ttnn.sigmoid(
        ttnn.linear(hidden_states, b_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    if allow_neg_eigval:
        beta = ttnn.multiply(beta, 2.0, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG DELTANET] After beta computation - beta shape: {beta.shape}")
    beta_torch = ttnn.to_torch(beta)
    print(f"[DEBUG DELTANET] Beta sample (first 5): {beta_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] Beta stats - min: {beta_torch.min().item():.6f}, max: {beta_torch.max().item():.6f}, mean: {beta_torch.mean().item():.6f}"
    )

    a = ttnn.linear(hidden_states, a_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    # g = -A * softplus(a + dt_bias)
    a_biased = ttnn.add(a, dt_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    sp = ttnn.softplus(a_biased, memory_config=ttnn.L1_MEMORY_CONFIG)
    A = ttnn.exp(A_log, memory_config=ttnn.L1_MEMORY_CONFIG)
    A_neg = ttnn.neg(A, memory_config=ttnn.L1_MEMORY_CONFIG)
    g = ttnn.multiply(A_neg, sp, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG DELTANET] After g computation - g shape: {g.shape}")
    g_torch = ttnn.to_torch(g)
    print(f"[DEBUG DELTANET] G sample (first 5): {g_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] G stats - min: {g_torch.min().item():.6f}, max: {g_torch.max().item():.6f}, mean: {g_torch.mean().item():.6f}"
    )

    # 5. Gated delta rule (recurrent or chunked)
    print(f"[DEBUG DELTANET] Before delta rule - q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
    if mode == "chunk":
        o, new_state = chunk_gated_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            chunk_size=chunk_size,
            initial_state=recurrent_state,
            device=device,
        )
    else:
        o, new_state = recurrent_gated_delta_rule_ttnn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            g=g,
            initial_state=recurrent_state,
            device=device,
        )
    print(f"[DEBUG DELTANET] After delta rule - o shape: {o.shape}, new_state shape: {new_state.shape}")
    o_torch = ttnn.to_torch(o)
    print(f"[DEBUG DELTANET] After delta rule o sample (first 5): {o_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] After delta rule o stats - min: {o_torch.min().item():.6f}, max: {o_torch.max().item():.6f}, mean: {o_torch.mean().item():.6f}"
    )

    # 6. Output normalization
    print(f"[DEBUG DELTANET] Before normalization - o shape: {o.shape}")
    o_before_norm_torch = ttnn.to_torch(o)
    print(f"[DEBUG DELTANET] Before normalization o sample (first 5): {o_before_norm_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] Before normalization o stats - min: {o_before_norm_torch.min().item():.6f}, max: {o_before_norm_torch.max().item():.6f}, mean: {o_before_norm_torch.mean().item():.6f}"
    )

    if use_gate and g_proj_weight is not None:
        gate = ttnn.linear(hidden_states, g_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.reshape(gate, [B, T, num_v_heads, head_v_dim])
        print(f"[DEBUG DELTANET] Gate shape: {gate.shape}")
        gate_torch = ttnn.to_torch(gate)
        print(f"[DEBUG DELTANET] Gate sample (first 5): {gate_torch.flatten()[:5].tolist()}")
        print(
            f"[DEBUG DELTANET] Gate stats - min: {gate_torch.min().item():.6f}, max: {gate_torch.max().item():.6f}, mean: {gate_torch.mean().item():.6f}"
        )
        print(f"[DEBUG DELTANET] o_norm_weight shape: {o_norm_weight.shape}")
        o_norm_weight_torch = ttnn.to_torch(o_norm_weight)
        print(f"[DEBUG DELTANET] o_norm_weight sample (first 5): {o_norm_weight_torch.flatten()[:5].tolist()}")
        o = rms_norm_gated_ttnn(o, gate, o_norm_weight, eps=norm_eps)
    else:
        o = rms_norm_ttnn(o, o_norm_weight, eps=norm_eps)
    print(f"[DEBUG DELTANET] After normalization - o shape: {o.shape}")
    o_torch = ttnn.to_torch(o)
    print(f"[DEBUG DELTANET] After normalization o sample (first 5): {o_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] After normalization o stats - min: {o_torch.min().item():.6f}, max: {o_torch.max().item():.6f}, mean: {o_torch.mean().item():.6f}"
    )

    # 7. Reshape and project output
    o = ttnn.reshape(o, [B, T, num_v_heads * head_v_dim])
    o = ttnn.linear(o, o_proj_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"[DEBUG DELTANET] Final output - o shape: {o.shape}")
    o_torch = ttnn.to_torch(o)
    print(f"[DEBUG DELTANET] Final output o sample (first 5): {o_torch.flatten()[:5].tolist()}")
    print(
        f"[DEBUG DELTANET] Final output o stats - min: {o_torch.min().item():.6f}, max: {o_torch.max().item():.6f}, mean: {o_torch.mean().item():.6f}"
    )

    return o, new_state


class TTNNRecurrentDeltaNet(TTNNModule):
    """TTNN-accelerated Recurrent DeltaNet layer."""

    def __init__(
        self,
        num_heads: int = 4,
        num_v_heads: Optional[int] = None,
        head_k_dim: int = 256,
        head_v_dim: int = 512,
        conv_kernel_size: int = 4,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        mode: str = "recurrent",
        chunk_size: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads or num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps
        self.mode = mode
        self.chunk_size = chunk_size

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        """Create TTNNRecurrentDeltaNet from PyTorch module."""
        new_layer = cls(*args, **kwargs)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    def preprocess_weights_impl(self):
        """Preprocess weights for TTNN operations."""
        if self.torch_layer is None:
            return

        # Extract weights from PyTorch layer
        state_dict = self.torch_layer.state_dict()

        # Preprocess linear projection weights
        self.q_proj_weight = self._preprocess_linear_weight(state_dict.get("q_proj.weight"))
        self.k_proj_weight = self._preprocess_linear_weight(state_dict.get("k_proj.weight"))
        self.v_proj_weight = self._preprocess_linear_weight(state_dict.get("v_proj.weight"))
        self.a_proj_weight = self._preprocess_linear_weight(state_dict.get("a_proj.weight"))
        self.b_proj_weight = self._preprocess_linear_weight(state_dict.get("b_proj.weight"))
        self.o_proj_weight = self._preprocess_linear_weight(state_dict.get("o_proj.weight"))

        if self.use_gate:
            self.g_proj_weight = self._preprocess_linear_weight(state_dict.get("g_proj.weight"))

        # Preprocess convolution weights
        self.q_conv_weight = self._preprocess_conv_weight(state_dict.get("q_conv.weight"))
        self.k_conv_weight = self._preprocess_conv_weight(state_dict.get("k_conv.weight"))
        self.v_conv_weight = self._preprocess_conv_weight(state_dict.get("v_conv.weight"))

        # Preprocess biases
        self.q_conv_bias = self._preprocess_bias(state_dict.get("q_conv.bias"))
        self.k_conv_bias = self._preprocess_bias(state_dict.get("k_conv.bias"))
        self.v_conv_bias = self._preprocess_bias(state_dict.get("v_conv.bias"))

        # Preprocess other parameters
        self.A_log = self._preprocess_param(state_dict.get("A_log"))
        self.dt_bias = self._preprocess_param(state_dict.get("dt_bias"))
        self.o_norm_weight = self._preprocess_param(state_dict.get("o_norm.weight"))

    def _preprocess_linear_weight(self, weight):
        """Preprocess linear weight for TTNN."""
        if weight is None:
            return None
        # Transpose for TTNN convention [in_features, out_features]
        weight_t = weight.transpose(0, 1)
        return ttnn.from_torch(weight_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _preprocess_conv_weight(self, weight):
        """Preprocess convolution weight for TTNN."""
        if weight is None:
            return None
        return ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _preprocess_bias(self, bias):
        """Preprocess bias for TTNN."""
        if bias is None:
            return None
        return ttnn.from_torch(bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _preprocess_param(self, param):
        """Preprocess parameter for TTNN."""
        if param is None:
            return None
        return ttnn.from_torch(param, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        if self.q_proj_weight is not None:
            self.q_proj_weight = ttnn.to_device(self.q_proj_weight, self.device)
        if self.k_proj_weight is not None:
            self.k_proj_weight = ttnn.to_device(self.k_proj_weight, self.device)
        if self.v_proj_weight is not None:
            self.v_proj_weight = ttnn.to_device(self.v_proj_weight, self.device)
        if self.a_proj_weight is not None:
            self.a_proj_weight = ttnn.to_device(self.a_proj_weight, self.device)
        if self.b_proj_weight is not None:
            self.b_proj_weight = ttnn.to_device(self.b_proj_weight, self.device)
        if self.o_proj_weight is not None:
            self.o_proj_weight = ttnn.to_device(self.o_proj_weight, self.device)
        if self.use_gate and self.g_proj_weight is not None:
            self.g_proj_weight = ttnn.to_device(self.g_proj_weight, self.device)
        if self.q_conv_weight is not None:
            self.q_conv_weight = ttnn.to_device(self.q_conv_weight, self.device)
        if self.k_conv_weight is not None:
            self.k_conv_weight = ttnn.to_device(self.k_conv_weight, self.device)
        if self.v_conv_weight is not None:
            self.v_conv_weight = ttnn.to_device(self.v_conv_weight, self.device)
        if self.q_conv_bias is not None:
            self.q_conv_bias = ttnn.to_device(self.q_conv_bias, self.device)
        if self.k_conv_bias is not None:
            self.k_conv_bias = ttnn.to_device(self.k_conv_bias, self.device)
        if self.v_conv_bias is not None:
            self.v_conv_bias = ttnn.to_device(self.v_conv_bias, self.device)
        if self.A_log is not None:
            self.A_log = ttnn.to_device(self.A_log, self.device)
        if self.dt_bias is not None:
            self.dt_bias = ttnn.to_device(self.dt_bias, self.device)
        if self.o_norm_weight is not None:
            self.o_norm_weight = ttnn.to_device(self.o_norm_weight, self.device)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        recurrent_state: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass through Recurrent DeltaNet.

        Args:
            hidden_states: [B, T, hidden_size] input tensor
            recurrent_state: [B, H, K, V] optional initial state

        Returns:
            output: [B, T, hidden_size] output tensor
            new_state: [B, H, K, V] updated recurrent state
        """
        # Ensure input is in correct layout
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Call the TTNN implementation
        output, new_state = gated_deltanet_forward_ttnn(
            hidden_states=hidden_states,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            a_proj_weight=self.a_proj_weight,
            b_proj_weight=self.b_proj_weight,
            o_proj_weight=self.o_proj_weight,
            q_conv_weight=self.q_conv_weight,
            k_conv_weight=self.k_conv_weight,
            v_conv_weight=self.v_conv_weight,
            q_conv_bias=self.q_conv_bias,
            k_conv_bias=self.k_conv_bias,
            v_conv_bias=self.v_conv_bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            o_norm_weight=self.o_norm_weight,
            g_proj_weight=getattr(self, "g_proj_weight", None),
            num_heads=self.num_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_kernel_size,
            use_gate=self.use_gate,
            allow_neg_eigval=self.allow_neg_eigval,
            norm_eps=self.norm_eps,
            device=self.device,
            recurrent_state=recurrent_state,
            mode=self.mode,
            chunk_size=self.chunk_size,
        )

        return output, new_state
