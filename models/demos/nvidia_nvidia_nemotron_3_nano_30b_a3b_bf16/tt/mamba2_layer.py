# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mamba2Layer — TP=4 on QB 4-chip Blackhole, S=1 decode path only.

SSM recurrence implemented via standard TTNN ops (softplus, exp, mul, matmul).
nemotron3_mamba2_decode_owned hangs on 4-chip MeshDevice (investigated with
both FABRIC_1D and FABRIC_1D_RING); this path is fully device-resident and
trace-compatible.

For S=1 with zero initial state, causal conv1d simplifies to:
  out[c] = conv_weight[c, 0, -1] * hBC[c] + conv_bias[c]  (elementwise, no padding needed)

Weight shapes (bfloat16 unless noted):
  norm.weight     : [2688]
  in_proj.weight  : [10304, 2688]
  conv1d.weight   : [6144, 1, 4]
  conv1d.bias     : [6144]
  dt_bias         : [64]   bf16
  A_log           : [64]   fp32 in ckpt
  mixer norm.w    : [4096] bf16  (MambaRMSNormGated scale)
  D               : [64]   fp32 in ckpt
  out_proj.weight : [2688, 4096]
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _rep, _rep_keyed

NUM_HEADS = 64
HEAD_DIM = 64
N_GROUPS = 8
SSM_STATE_SIZE = 128
INTERMEDIATE_SIZE = NUM_HEADS * HEAD_DIM  # 4096
CONV_DIM = INTERMEDIATE_SIZE + 2 * N_GROUPS * SSM_STATE_SIZE  # 6144
NORM_EPS = 1e-5


def mamba2_layer_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [B, 1, 2688] bf16 on device
    norm_weight: torch.Tensor,  # [2688] bf16 CPU
    in_proj_weight: torch.Tensor,  # [10304, 2688] bf16 CPU
    conv1d_weight: torch.Tensor,  # [6144, 1, 4] bf16 CPU
    conv1d_bias: torch.Tensor,  # [6144] bf16 CPU
    dt_bias: torch.Tensor,  # [64] bf16 CPU
    A_log: torch.Tensor,  # [64] fp32 CPU
    norm_mixer_weight: torch.Tensor,  # [4096] bf16 CPU
    D: torch.Tensor,  # [64] fp32 CPU
    out_proj_weight: torch.Tensor,  # [2688, 4096] bf16 CPU
    norm_eps: float = NORM_EPS,
    ssm_state: ttnn.Tensor | None = None,  # [B, H, D, N] bf16 on device, None → zero-init
    conv_state: tuple | None = None,  # (h_tm3, h_tm2, h_tm1) each [B,1,6144] bf16 on device
) -> tuple:
    """S=1 decode path.

    Returns (output [B,1,2688], ssm_state_new [B,H,D,N], conv_state_new) bf16 on device.
    conv_state_new is (h_tm2, h_tm1, hBC_current) if conv_state provided, else None.

    ssm_state=None uses zero initial state (first token / non-stateful mode).
    conv_state=None uses zero-padded conv (only current token contributes via w[-1]).
    For stateful generation pass both from DecoderState.
    """
    residual = hidden_states
    B = hidden_states.shape[0]

    # 1. Pre-block RMSNorm
    w_tt = _rep_keyed(id(norm_weight), norm_weight.bfloat16().unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(hidden_states, epsilon=norm_eps, weight=w_tt)

    # 2. in_proj: [B, 1, 2688] → [B, 1, 10304]
    ip_tt = _rep(in_proj_weight, mesh_device)
    projected_tt = ttnn.linear(normed_tt, ip_tt, transpose_b=True)

    # 3. Split projected: gate [B,1,4096] | hBC [B,1,6144] | dt [B,1,64]
    gate_tt = ttnn.slice(projected_tt, [0, 0, 0], [B, 1, INTERMEDIATE_SIZE])
    hBC_tt = ttnn.slice(projected_tt, [0, 0, INTERMEDIATE_SIZE], [B, 1, INTERMEDIATE_SIZE + CONV_DIM])
    dt_slice_tt = ttnn.slice(
        projected_tt, [0, 0, INTERMEDIATE_SIZE + CONV_DIM], [B, 1, INTERMEDIATE_SIZE + CONV_DIM + NUM_HEADS]
    )

    # 4. Causal conv1d S=1.
    #    With full history: out[c] = sum_k(weight[c,0,k] * h[t-3+k][c]) + bias[c]
    #    With zero-padded (conv_state=None): out[c] = weight[c,0,3] * hBC[c] + bias[c]
    conv_b_tt = _rep_keyed(id(conv1d_bias), conv1d_bias.bfloat16().unsqueeze(0).unsqueeze(0).contiguous(), mesh_device)
    if conv_state is not None:
        h_tm3, h_tm2, h_tm1 = conv_state
        conv_w = [
            _rep_keyed(
                ("conv_w", id(conv1d_weight), k),
                conv1d_weight[:, 0, k].bfloat16().unsqueeze(0).unsqueeze(0).contiguous(),
                mesh_device,
            )
            for k in range(4)
        ]
        hBC_conv_tt = ttnn.add(
            ttnn.add(
                ttnn.add(ttnn.mul(h_tm3, conv_w[0]), ttnn.mul(h_tm2, conv_w[1])),
                ttnn.add(ttnn.mul(h_tm1, conv_w[2]), ttnn.mul(hBC_tt, conv_w[3])),
            ),
            conv_b_tt,
        )
        conv_state_new = (h_tm2, h_tm1, hBC_tt)
    else:
        conv_w_tt = _rep_keyed(
            ("conv_w", id(conv1d_weight), 3),
            conv1d_weight[:, 0, 3].bfloat16().unsqueeze(0).unsqueeze(0).contiguous(),
            mesh_device,
        )
        hBC_conv_tt = ttnn.add(ttnn.mul(hBC_tt, conv_w_tt), conv_b_tt)
        conv_state_new = None

    # 5. Silu activation
    hBC_silu_tt = ttnn.silu(hBC_conv_tt)  # [B, 1, 6144]

    # 6. Split hBC_silu: x [B,1,4096] | B_vec [B,1,1024] | C_vec [B,1,1024]
    x_flat_tt = ttnn.slice(hBC_silu_tt, [0, 0, 0], [B, 1, INTERMEDIATE_SIZE])
    b_flat_tt = ttnn.slice(
        hBC_silu_tt, [0, 0, INTERMEDIATE_SIZE], [B, 1, INTERMEDIATE_SIZE + N_GROUPS * SSM_STATE_SIZE]
    )
    c_flat_tt = ttnn.slice(hBC_silu_tt, [0, 0, INTERMEDIATE_SIZE + N_GROUPS * SSM_STATE_SIZE], [B, 1, CONV_DIM])

    # 7. Reshape for SSM decode
    #    x:    [B, 1, 4096] → [B, 64, 64]
    #    B_in: [B, 1, 1024] → [B, 8, 128]
    #    C_in: [B, 1, 1024] → [B, 8, 128]
    #    dt:   [B, 1, 64]   → [B, 64]
    x_tt = ttnn.reshape(x_flat_tt, [B, NUM_HEADS, HEAD_DIM])
    B_in_tt = ttnn.reshape(b_flat_tt, [B, N_GROUPS, SSM_STATE_SIZE])
    C_in_tt = ttnn.reshape(c_flat_tt, [B, N_GROUPS, SSM_STATE_SIZE])
    dt_tt = ttnn.reshape(dt_slice_tt, [B, NUM_HEADS])

    # 8. SSM scalar weights — upload as [1, N] for tile compatibility
    dt_bias_tt = _rep_keyed(id(dt_bias), dt_bias.bfloat16().unsqueeze(0), mesh_device)
    A_log_tt = _rep_keyed(id(A_log), A_log.float().bfloat16().unsqueeze(0), mesh_device)
    D_tt = _rep_keyed(id(D), D.float().bfloat16().unsqueeze(0), mesh_device)

    # 9–10. S=1 SSM decode using standard TTNN ops.
    #
    # nemotron3_mamba2_decode_owned hangs on 4-chip MeshDevice (investigated:
    # reproducible with both FABRIC_1D and FABRIC_1D_RING — pure local-compute
    # kernel, root cause unknown at time of writing, likely multi-chip dispatch
    # incompatibility).  This TTNN-native path produces identical math and is
    # fully trace-compatible.
    #
    # Mamba2 S=1 recurrence (zero initial state):
    #   dt_eff     = softplus(dt + dt_bias)              [B, H]
    #   decay      = exp(-exp(A_log) * dt_eff)           [B, H]
    #   x_dt       = x * dt_eff                          [B, H, D]
    #   B_exp/C_exp = repeat B/C groups → full heads     [B, H, N]
    #   state_new  = outer(x_dt, B_exp)                  [B, H, D, N]  (zero-init state)
    #   y_ssm      = state_new @ C_exp                   [B, H, D]
    #   y          = y_ssm + D_scalar * x                [B, H, D]
    HEADS_PER_GROUP = NUM_HEADS // N_GROUPS  # 8

    # dt_eff [B, H]
    dt_eff_tt = ttnn.softplus(ttnn.add(dt_tt, dt_bias_tt))

    # decay  [B, H]
    A_neg_tt = ttnn.neg(ttnn.exp(A_log_tt))
    decay_tt = ttnn.exp(ttnn.mul(A_neg_tt, dt_eff_tt))

    # x_dt [B, H, D]: scale x by dt_eff (broadcast over head_dim)
    dt_eff_3d = ttnn.reshape(dt_eff_tt, [B, NUM_HEADS, 1])
    x_dt_tt = ttnn.mul(x_tt, dt_eff_3d)

    # Expand B and C from N_GROUPS to NUM_HEADS by replicating each group slice
    B_slices, C_slices = [], []
    for g in range(N_GROUPS):
        b_g = ttnn.slice(B_in_tt, [0, g, 0], [B, g + 1, SSM_STATE_SIZE])
        c_g = ttnn.slice(C_in_tt, [0, g, 0], [B, g + 1, SSM_STATE_SIZE])
        for _ in range(HEADS_PER_GROUP):
            B_slices.append(b_g)
            C_slices.append(c_g)
    B_exp_tt = ttnn.concat(B_slices, dim=1)  # [B, H, N]
    C_exp_tt = ttnn.concat(C_slices, dim=1)  # [B, H, N]

    # Outer product state update: new_contrib = outer(x_dt, B_exp) [B, H, D, N]
    x_dt_4d = ttnn.reshape(x_dt_tt, [B, NUM_HEADS, HEAD_DIM, 1])
    B_exp_4d = ttnn.reshape(B_exp_tt, [B, NUM_HEADS, 1, SSM_STATE_SIZE])
    new_contrib_tt = ttnn.mul(x_dt_4d, B_exp_4d)  # [B, H, D, N]

    # Full recurrence: state_new = decay * state_prev + new_contrib
    if ssm_state is not None:
        decay_4d_tt = ttnn.reshape(decay_tt, [B, NUM_HEADS, 1, 1])  # broadcast over D, N
        state_new_tt = ttnn.add(ttnn.mul(decay_4d_tt, ssm_state), new_contrib_tt)
    else:
        state_new_tt = new_contrib_tt  # zero initial state

    # y_ssm = state_new @ C_exp  (matmul over state_size dim)
    C_exp_4d = ttnn.reshape(C_exp_tt, [B, NUM_HEADS, SSM_STATE_SIZE, 1])
    y_4d_tt = ttnn.matmul(state_new_tt, C_exp_4d)  # [B, H, D, 1]
    y_ssm_tt = ttnn.reshape(y_4d_tt, [B, NUM_HEADS, HEAD_DIM])  # [B, H, D]

    # D skip: y = y_ssm + D * x
    D_3d_tt = ttnn.reshape(D_tt, [1, NUM_HEADS, 1])
    y_tt = ttnn.add(y_ssm_tt, ttnn.mul(D_3d_tt, x_tt))  # [B, H, D]

    # 11. MambaRMSNormGated — gate-first, per-group RMS, scale
    #     y:    [B, 64, 64] → [B, 1, 4096]
    #     gate: [B, 1, 4096] (already sliced above as gate_tt)
    y_flat_tt = ttnn.reshape(y_tt, [B, 1, INTERMEDIATE_SIZE])
    gate_silu_tt = ttnn.silu(gate_tt)  # [B, 1, 4096]
    xg_tt = ttnn.mul(y_flat_tt, gate_silu_tt)  # [B, 1, 4096]

    # Per-group RMS: group_size = 4096 / 8 = 512
    GROUP_SIZE = INTERMEDIATE_SIZE // N_GROUPS
    xg_grouped_tt = ttnn.reshape(xg_tt, [B, 1, N_GROUPS, GROUP_SIZE])  # [B, 1, 8, 512]
    xg_sq_tt = ttnn.pow(xg_grouped_tt, 2)
    var_tt = ttnn.mean(xg_sq_tt, dim=3, keepdim=True)  # [B, 1, 8, 1]
    xg_normed_tt = ttnn.mul(xg_grouped_tt, ttnn.rsqrt(ttnn.add(var_tt, norm_eps)))
    xg_normed_flat_tt = ttnn.reshape(xg_normed_tt, [B, 1, INTERMEDIATE_SIZE])

    norm_w_tt = _rep_keyed(id(norm_mixer_weight), norm_mixer_weight.bfloat16().unsqueeze(0).unsqueeze(0), mesh_device)
    scan_out_tt = ttnn.mul(xg_normed_flat_tt, norm_w_tt)  # [B, 1, 4096]

    # 12. out_proj: [B, 1, 4096] → [B, 1, 2688]
    op_tt = _rep(out_proj_weight, mesh_device)
    out_tt = ttnn.linear(scan_out_tt, op_tt, transpose_b=True)

    # 13. Residual
    return ttnn.add(residual, out_tt), state_new_tt, conv_state_new
