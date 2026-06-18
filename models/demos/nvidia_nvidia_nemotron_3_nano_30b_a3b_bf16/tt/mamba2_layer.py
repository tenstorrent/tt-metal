# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mamba2Layer — TP=4 on QB 4-chip Blackhole, S=1 decode + S>1 prefill.

Dispatch:
  S == 1  →  mamba2_layer_forward()        (this file, trace-compatible decode)
  S  > 1  →  mamba2_prefill_layer_forward() (mamba2_prefill.py, chunked SSD scan)

SSM recurrence implemented via standard TTNN ops (softplus, exp, mul, matmul).
nemotron3_mamba2_decode_owned hangs on 4-chip MeshDevice (investigated with
both FABRIC_1D and FABRIC_1D_RING); this path is fully device-resident and
trace-compatible.

All TILE reshapes that change the last-two-dim tile structure route through a
ROW_MAJOR intermediate (_rr helper) to avoid a Blackhole relayout-kernel hang
that triggers on certain DRAM buffer addresses when the tile must be re-padded.
Shapes confirmed to hang directly: [1,1,N]→[B,G,S], [B,H,D]→[B,H,D,1],
[B,H,N]→[B,H,1,N] and similar singleton-dim inserts at larger tensor sizes.

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

from .tp import _col, _rep_keyed, _row, all_gather, all_reduce

NUM_HEADS = 64
HEAD_DIM = 64
N_GROUPS = 8
SSM_STATE_SIZE = 128
INTERMEDIATE_SIZE = NUM_HEADS * HEAD_DIM  # 4096
CONV_DIM = INTERMEDIATE_SIZE + 2 * N_GROUPS * SSM_STATE_SIZE  # 6144
NORM_EPS = 1e-5

_RM = ttnn.ROW_MAJOR_LAYOUT
_TL = ttnn.TILE_LAYOUT


def _rr(t: ttnn.Tensor, shape: list) -> ttnn.Tensor:
    """Reshape a TILE tensor safely via ROW_MAJOR intermediate.

    Direct TILE→TILE relayout for shapes that change the last-two-dim tile
    structure (e.g. adding/removing singleton dims) triggers a Blackhole
    device-side kernel that deadlocks on certain DRAM buffer addresses.
    Going TILE→RM→reshape→TILE avoids that kernel entirely.
    """
    return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(t, _RM), shape), _TL)


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
    _dbg: bool = False,  # kept for backward compat (model.py passes _dbg=debug_sync)
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

    # 2. in_proj: column-parallel → partial [B, 1, 2576]/device, then all_gather → [B, 1, 10304].
    # Column sharding halves on-device weight DRAM (55.5 MB → 13.9 MB per layer/device).
    # At S=1 the all_gather tensor is only 20 KB — negligible CCL cost.
    ip_tt = _col(in_proj_weight, mesh_device)  # [2576, 2688]/device
    _proj_partial = ttnn.linear(normed_tt, ip_tt, transpose_b=True)  # [B, 1, 2576]/device
    projected_tt = all_gather(_proj_partial, dim=2)  # [B, 1, 10304]
    _proj_partial.deallocate(True)

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

    # 7. Reshape for SSM decode — all via _rr (TILE→RM→reshape→TILE) to avoid
    #    a Blackhole relayout-kernel bug that deadlocks on certain DRAM addresses.
    #    x:    [B, 1, 4096] → [B, 64, 64]
    #    B_in: [B, 1, 1024] → [B, 8, 128]
    #    C_in: [B, 1, 1024] → [B, 8, 128]
    #    dt:   [B, 1, 64]   → [B, 64]  (last-2-dims [1,64]→[1,64]: safe, no _rr needed)
    x_tt = _rr(x_flat_tt, [B, NUM_HEADS, HEAD_DIM])
    B_in_tt = _rr(b_flat_tt, [B, N_GROUPS, SSM_STATE_SIZE])
    C_in_tt = _rr(c_flat_tt, [B, N_GROUPS, SSM_STATE_SIZE])
    dt_tt = ttnn.reshape(dt_slice_tt, [B, NUM_HEADS])

    # 8. SSM scalar weights — upload as [1, N] for tile compatibility.
    #    Previously these used explicit memory_config=L1, but 23 layers × 3 weights
    #    × 4KB (TILE) = 276KB of PERSISTENT L1 (cached in _DERIVED_CACHE forever).
    #    At ISL=512 the ttnn.slice([1,512,2688]) kernel compiles with a larger CB
    #    region [0,746240] that extends over those tensors, causing a CB clash.
    #    _rep_keyed already has DRAM-corruption detection + RM-L1 fallback: weights
    #    uploaded during warmup (before heavy DRAM recycling) land in safe DRAM TILE;
    #    if corruption is ever detected, _rep_keyed heals to RM-L1 = 128 B each.
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
    # Mamba2 S=1 recurrence:
    #   dt_eff     = softplus(dt + dt_bias)              [B, H]
    #   decay      = exp(-exp(A_log) * dt_eff)           [B, H]
    #   x_dt       = x * dt_eff                          [B, H, D]
    #   B_exp/C_exp = repeat B/C groups → full heads     [B, H, N]
    #   state_new  = decay * state + outer(x_dt, B_exp)  [B, H, D, N]
    #   y_ssm      = state_new @ C_exp                   [B, H, D]
    #   y          = y_ssm + D_scalar * x                [B, H, D]
    HEADS_PER_GROUP = NUM_HEADS // N_GROUPS  # 8

    # dt_eff [B, H] and decay [B, H]
    dt_eff_tt = ttnn.softplus(ttnn.add(dt_tt, dt_bias_tt))
    A_neg_tt = ttnn.neg(ttnn.exp(A_log_tt))
    decay_tt = ttnn.exp(ttnn.mul(A_neg_tt, dt_eff_tt))

    # x_dt [B, H, D]: scale x by dt_eff (broadcast over head_dim)
    dt_eff_3d = _rr(dt_eff_tt, [B, NUM_HEADS, 1])  # [1,64]→[1,64,1] via RM
    x_dt_tt = ttnn.mul(x_tt, dt_eff_3d)  # [B, H, D]

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
    # Use _rr for 4D reshapes (same TILE-relayout hang class as step 7).
    # Use matmul([B,H,D,1] @ [B,H,1,N]) for the outer product instead of broadcast mul.
    x_dt_4d = _rr(x_dt_tt, [B, NUM_HEADS, HEAD_DIM, 1])  # [1,64,64]→[1,64,64,1]
    B_exp_4d = _rr(B_exp_tt, [B, NUM_HEADS, 1, SSM_STATE_SIZE])  # [1,64,128]→[1,64,1,128]
    new_contrib_tt = ttnn.matmul(x_dt_4d, B_exp_4d)  # [B,H,D,1]@[B,H,1,N]=[B,H,D,N]

    # Full recurrence: state_new = decay * state_prev + new_contrib
    if ssm_state is not None:
        decay_4d_tt = _rr(decay_tt, [B, NUM_HEADS, 1, 1])  # [1,64]→[1,64,1,1] via RM
        state_new_tt = ttnn.add(ttnn.mul(decay_4d_tt, ssm_state), new_contrib_tt)
    else:
        state_new_tt = new_contrib_tt  # zero initial state

    # y_ssm = state_new @ C_exp  (matmul over state_size dim)
    C_exp_4d = _rr(C_exp_tt, [B, NUM_HEADS, SSM_STATE_SIZE, 1])  # [1,64,128]→[1,64,128,1]
    y_4d_tt = ttnn.matmul(state_new_tt, C_exp_4d)  # [B, H, D, 1]
    y_ssm_tt = _rr(y_4d_tt, [B, NUM_HEADS, HEAD_DIM])  # [1,64,64,1]→[1,64,64]

    # D skip: y = y_ssm + D * x
    D_3d_tt = _rr(D_tt, [1, NUM_HEADS, 1])  # [1,64]→[1,64,1] via RM
    y_tt = ttnn.add(y_ssm_tt, ttnn.mul(D_3d_tt, x_tt))  # [B, H, D]

    # 11. MambaRMSNormGated — gate-first, per-group RMS, scale
    #     y:    [B, 64, 64] → [B, 1, 4096]
    #     gate: [B, 1, 4096] (already sliced above as gate_tt)
    y_flat_tt = _rr(y_tt, [B, 1, INTERMEDIATE_SIZE])  # [1,64,64]→[1,1,4096] via RM
    gate_silu_tt = ttnn.silu(gate_tt)  # [B, 1, 4096]
    xg_tt = ttnn.mul(y_flat_tt, gate_silu_tt)  # [B, 1, 4096]

    # Per-group RMS: group_size = 4096 / 8 = 512
    GROUP_SIZE = INTERMEDIATE_SIZE // N_GROUPS
    xg_grouped_tt = _rr(xg_tt, [B, 1, N_GROUPS, GROUP_SIZE])  # [1,1,4096]→[1,1,8,512]
    xg_sq_tt = ttnn.pow(xg_grouped_tt, 2)
    var_tt = ttnn.mean(xg_sq_tt, dim=3, keepdim=True)  # [B, 1, 8, 1]
    xg_normed_tt = ttnn.mul(xg_grouped_tt, ttnn.rsqrt(ttnn.add(var_tt, norm_eps)))
    xg_normed_flat_tt = _rr(xg_normed_tt, [B, 1, INTERMEDIATE_SIZE])  # [1,1,8,512]→[1,1,4096]

    norm_w_tt = _rep_keyed(id(norm_mixer_weight), norm_mixer_weight.bfloat16().unsqueeze(0).unsqueeze(0), mesh_device)
    scan_out_tt = ttnn.mul(xg_normed_flat_tt, norm_w_tt)  # [B, 1, 4096]

    # 12. out_proj: row-parallel → partial [B, 1, 2688]/device, then all_reduce → full.
    # Row sharding: [2688, 4096] → [2688, 1024]/device (22 MB → 5.5 MB per layer/device).
    op_tt = _row(out_proj_weight, mesh_device)  # [2688, 1024]/device
    _out_partial = ttnn.linear(scan_out_tt, op_tt, transpose_b=True)  # [B, 1, 2688] partial
    out_tt = all_reduce(_out_partial)  # [B, 1, 2688] full
    _out_partial.deallocate(True)

    # 13. Residual
    return ttnn.add(residual, out_tt), state_new_tt, conv_state_new


def mamba2_layer_forward_dispatch(
    mesh_device,
    hidden_states,  # [B, S, 2688]
    norm_weight,
    in_proj_weight,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A_log,
    norm_mixer_weight,
    D,
    out_proj_weight,
    norm_eps=NORM_EPS,
    ssm_state=None,
    conv_state=None,
    _dbg=False,
):
    """Dispatch to prefill (S>1) or decode (S==1) based on sequence length."""
    S = hidden_states.shape[1]
    if S == 1:
        return mamba2_layer_forward(
            mesh_device,
            hidden_states,
            norm_weight,
            in_proj_weight,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A_log,
            norm_mixer_weight,
            D,
            out_proj_weight,
            norm_eps,
            ssm_state,
            conv_state,
            _dbg,
        )
    # S > 1 — chunked SSD prefill
    from .mamba2_prefill import mamba2_prefill_layer_forward

    return mamba2_prefill_layer_forward(
        mesh_device,
        hidden_states,
        norm_weight,
        in_proj_weight,
        conv1d_weight,
        conv1d_bias,
        dt_bias,
        A_log,
        norm_mixer_weight,
        D,
        out_proj_weight,
        norm_eps,
        ssm_state,
        conv_state,
    )
