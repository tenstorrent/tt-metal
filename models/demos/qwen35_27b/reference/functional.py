# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Pure-PyTorch (float32) reference implementations for GDN prefill.

Matches the math in gdn_fused.cpp / gdn_prefill.cpp exactly:
  Phase 1: L2-norm Q, scale
  Phase 2: L2-norm K
  Phase 3: K^T (column vector)
  Phase 4: beta = sigmoid(b), g = neg_exp_A * softplus(a + dt_bias)
  Phase 5: DeltaNet recurrence
             state *= exp(g)
             kv_mem  = k_row @ state
             delta   = beta * (v - kv_mem)
             state  += outer(k_col, delta)
             out     = q @ state

Inputs follow the same shape conventions as gdn_prefill_fused:
  conv_out  : [1, N, qkv_dim_tp]   — post-conv1d+SiLU
  a_all     : [1, N, Nv_TP]
  b_all     : [1, N, Nv_TP]
  neg_exp_A : [1, 1, Nv_TP]
  dt_bias   : [1, 1, Nv_TP]
  norm_w    : [1, 1, Dv]            (for RMS-norm post-processing, not used here)
  scale     : float                  = Dk ** -0.5
"""

import torch
import torch.nn.functional as F


def gdn_recurrence_step(
    q: torch.Tensor,  # [num_pairs, Dk]
    k: torch.Tensor,  # [num_pairs, Dk]
    v: torch.Tensor,  # [num_pairs, Dv]
    g: torch.Tensor,  # [num_pairs]  log-space decay (negative)
    beta: torch.Tensor,  # [num_pairs]  beta gate in (0, 1)
    state: torch.Tensor,  # [num_pairs, Dk, Dv]  modified in-place
) -> torch.Tensor:
    """Single DeltaNet recurrence step. Updates state in-place, returns output."""
    decay = torch.exp(g)  # [num_pairs]

    # 1. state *= exp(g)
    state.mul_(decay[:, None, None])

    # 2. kv_mem = k_row @ state  →  [num_pairs, Dv]
    kv_mem = torch.bmm(k.unsqueeze(1), state).squeeze(1)

    # 3. delta = beta * (v - kv_mem)
    delta = beta[:, None] * (v - kv_mem)  # [num_pairs, Dv]

    # 4. state += outer(k_col, delta)  →  [num_pairs, Dk, Dv]
    state.add_(torch.bmm(k.unsqueeze(2), delta.unsqueeze(1)))

    # 5. out = q @ state  →  [num_pairs, Dv]
    return torch.bmm(q.unsqueeze(1), state).squeeze(1)


def gdn_prefill_ref(
    conv_out: torch.Tensor,  # [1, N, qkv_dim_tp]
    a_all: torch.Tensor,  # [1, N, Nv_TP]
    b_all: torch.Tensor,  # [1, N, Nv_TP]
    neg_exp_A: torch.Tensor,  # [1, 1, Nv_TP]
    dt_bias: torch.Tensor,  # [1, 1, Nv_TP]
    scale: float,
    Dk: int,
    Dv: int,
    Nk_TP: int,
    Nv_TP: int,
    repeat_factor: int,
    key_dim_tp: int,
    init_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch GDN prefill over N tokens.

    Returns:
        outputs     : [num_pairs, N, Dv]   per-token recurrence outputs
        final_state : [num_pairs, Dk, Dv]  state after last token
    """
    N = conv_out.shape[1]
    num_pairs = Nv_TP  # B=1 for prefill

    state = (
        init_state.clone().float() if init_state is not None else torch.zeros(num_pairs, Dk, Dv, dtype=torch.float32)
    )

    # Use float32 for reference accuracy
    conv_out = conv_out.float().squeeze(0)  # [N, qkv_dim_tp]
    a_all = a_all.float().squeeze(0)  # [N, Nv_TP]
    b_all = b_all.float().squeeze(0)  # [N, Nv_TP]
    neg_exp_A = neg_exp_A.float()[0, 0]  # [Nv_TP]
    dt_bias = dt_bias.float()[0, 0]  # [Nv_TP]

    outputs = []
    for t in range(N):
        conv_t = conv_out[t]  # [qkv_dim_tp]

        # Split Q / K / V
        q_raw = conv_t[:key_dim_tp].view(Nk_TP, Dk)
        k_raw = conv_t[key_dim_tp : 2 * key_dim_tp].view(Nk_TP, Dk)
        v_raw = conv_t[2 * key_dim_tp :].view(Nv_TP, Dv)

        # L2 norm Q and K (per head)
        q_n = F.normalize(q_raw, p=2, dim=-1)  # [Nk_TP, Dk]
        k_n = F.normalize(k_raw, p=2, dim=-1)  # [Nk_TP, Dk]

        # Scale Q and expand K/Q from Nk_TP to Nv_TP
        q_scaled = q_n * scale  # [Nk_TP, Dk]
        q_exp = q_scaled.repeat_interleave(repeat_factor, dim=0)  # [Nv_TP, Dk]
        k_exp = k_n.repeat_interleave(repeat_factor, dim=0)  # [Nv_TP, Dk]

        # Gates
        a_t = a_all[t]  # [Nv_TP]
        b_t = b_all[t]  # [Nv_TP]
        beta = torch.sigmoid(b_t)  # [Nv_TP]
        g = neg_exp_A * F.softplus(a_t + dt_bias)  # [Nv_TP]

        out_t = gdn_recurrence_step(q_exp, k_exp, v_raw, g, beta, state)
        outputs.append(out_t)

    return torch.stack(outputs, dim=1), state  # ([num_pairs, N, Dv], [num_pairs, Dk, Dv])
