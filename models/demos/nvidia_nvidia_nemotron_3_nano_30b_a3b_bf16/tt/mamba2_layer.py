# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mamba2Layer — TP=4 on QB 4-chip Blackhole, S=1 decode path only.

Uses ttnn.experimental.nemotron3_mamba2_decode_owned for the SSM recurrence.
Kernel boundary: POST conv1d/silu/split, PRE MambaRMSNormGated/out_proj.

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

from .tp import _rep

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
) -> ttnn.Tensor:
    """S=1 decode path. Returns [B, 1, 2688] bfloat16 on device."""
    residual = hidden_states
    B = hidden_states.shape[0]

    # 1. Pre-block RMSNorm
    w_tt = _rep(norm_weight.unsqueeze(0), mesh_device)
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

    # 4. Causal conv1d for S=1 with zero state:
    #    out[c] = weight[c, 0, -1] * hBC[c] + bias[c]  (elementwise)
    conv_w_last = conv1d_weight[:, 0, -1].contiguous().bfloat16()  # [6144] CPU
    conv_w_tt = _rep(conv_w_last.unsqueeze(0).unsqueeze(0), mesh_device)  # [1, 1, 6144]
    conv_b_tt = _rep(conv1d_bias.bfloat16().unsqueeze(0).unsqueeze(0), mesh_device)  # [1, 1, 6144]
    hBC_conv_tt = ttnn.add(ttnn.mul(hBC_tt, conv_w_tt), conv_b_tt)

    # 5. Silu activation
    hBC_silu_tt = ttnn.silu(hBC_conv_tt)  # [B, 1, 6144]

    # 6. Split hBC_silu: x [B,1,4096] | B_vec [B,1,1024] | C_vec [B,1,1024]
    x_flat_tt = ttnn.slice(hBC_silu_tt, [0, 0, 0], [B, 1, INTERMEDIATE_SIZE])
    b_flat_tt = ttnn.slice(
        hBC_silu_tt, [0, 0, INTERMEDIATE_SIZE], [B, 1, INTERMEDIATE_SIZE + N_GROUPS * SSM_STATE_SIZE]
    )
    c_flat_tt = ttnn.slice(hBC_silu_tt, [0, 0, INTERMEDIATE_SIZE + N_GROUPS * SSM_STATE_SIZE], [B, 1, CONV_DIM])

    # 7. Reshape inputs for decode kernel
    #    x:    [B, 1, 4096] → [B, 64, 64]
    #    z:    gate [B, 1, 4096] → [B, 64, 64]  (gating tensor, passed through by kernel)
    #    B_in: [B, 1, 1024] → [B, 8, 128]
    #    C_in: [B, 1, 1024] → [B, 8, 128]
    #    dt:   [B, 1, 64]   → [B, 64]
    x_tt = ttnn.reshape(x_flat_tt, [B, NUM_HEADS, HEAD_DIM])
    z_tt = ttnn.reshape(gate_tt, [B, NUM_HEADS, HEAD_DIM])
    B_in_tt = ttnn.reshape(b_flat_tt, [B, N_GROUPS, SSM_STATE_SIZE])
    C_in_tt = ttnn.reshape(c_flat_tt, [B, N_GROUPS, SSM_STATE_SIZE])
    dt_tt = ttnn.reshape(dt_slice_tt, [B, NUM_HEADS])

    # 8. SSM scalar weights — upload as [1, N] for tile compatibility
    dt_bias_tt = _rep(dt_bias.bfloat16().unsqueeze(0), mesh_device)  # [1, 64]
    A_log_tt = _rep(A_log.float().bfloat16().unsqueeze(0), mesh_device)  # [1, 64]
    D_tt = _rep(D.float().bfloat16().unsqueeze(0), mesh_device)  # [1, 64]

    # 9. Zero initial SSM state [B, num_heads, head_dim, state_size] fp32
    ssm_state = ttnn.from_torch(
        torch.zeros(B, NUM_HEADS, HEAD_DIM, SSM_STATE_SIZE),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # 10. Decode kernel → (ssm_state_out [discarded], y [B, 64, 64])
    _, y_tt = ttnn.experimental.nemotron3_mamba2_decode_owned(
        x_tt, z_tt, dt_tt, dt_bias_tt, A_log_tt, D_tt, B_in_tt, C_in_tt, ssm_state
    )

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

    norm_w_tt = _rep(norm_mixer_weight.unsqueeze(0).unsqueeze(0), mesh_device)  # [1, 1, 4096]
    scan_out_tt = ttnn.mul(xg_normed_flat_tt, norm_w_tt)  # [B, 1, 4096]

    # 12. out_proj: [B, 1, 4096] → [B, 1, 2688]
    op_tt = _rep(out_proj_weight, mesh_device)
    out_tt = ttnn.linear(scan_out_tt, op_tt, transpose_b=True)

    # 13. Residual
    return ttnn.add(residual, out_tt)
