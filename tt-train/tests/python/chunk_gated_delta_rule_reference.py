# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""float64 reference for the ttml chunked gated delta rule tests.

The oracle is ttnn's recurrent_gated_delta_rule golden (FLA's naive rule, token by token). The intermediates the
backward consumes follow from their definitions on the golden's states, not from the chunked algorithm the op runs.
"""

from __future__ import annotations

import torch

from ttnn.operations.transformer_golden import recurrent_gated_delta_rule

# Scale of g = logsigmoid(randn) * scale, about -0.82 * scale per token.
#   WEAK_GATE: a chunk keeps most of the state, so initial_state and the carry along the scan matter.
#   DECAY_GATE: a chunk keeps e^-13 .. e^-26 of the state; the within-chunk decay terms dominate.
#   STRONG_GATE: |cumsum(g)| in the hundreds inside a chunk.
WEAK_GATE = 0.02
DECAY_GATE = 0.5
STRONG_GATE = 4.0

# (B, T, HK, HV, K, V), chunk_size: Qwen3.8 GatedDeltaNet on one chip of 4-way tensor parallelism, as the LoRA run
# trains it.
QWEN_TP4_SHAPE = ((1, 1024, 4, 12, 128, 128), 64)


def reference_forward(q, k, v, g, beta, *, chunk_size, initial_state=None, scale=None):
    """The op's outputs in float64, head-major (row b * HV + h).

    q, k [B, T, HK, K]; v [B, T, HV, V]; g, beta [B, T, HV]; initial_state [B, HV, K, V]. Value head h reads key head
    h // (HV / HK). Returns o [B*HV, T, V], final_state [B, HV, K, V], h [B*HV, NC, K, V] (the state entering each
    chunk), v_new [B*HV, T, V] and A [B*HV, T, C] (row r of its chunk's (I + N)^-1).
    """
    f64 = torch.float64
    q, k, v, g, beta = (x.to(f64) for x in (q, k, v, g, beta))
    B, T, HK, K = q.shape
    HV, V = v.shape[2:]
    C = chunk_size
    NC = T // C

    state = torch.zeros(B, HV, K, V, dtype=f64) if initial_state is None else initial_state.to(f64)
    outs, states = [], []
    for i in range(NC):
        states.append(state)
        t = slice(i * C, (i + 1) * C)
        o_i, state = recurrent_gated_delta_rule(
            q[:, t],
            k[:, t],
            v[:, t],
            beta[:, t],
            g[:, t],
            scale=scale,
            initial_state=state,
            output_final_state=True,
            dtype=f64,
        )
        outs.append(o_i)

    per_chunk = lambda x: x.transpose(1, 2).reshape(B, HV, NC, C, -1)  # [B, T, HV, D] -> [B, HV, NC, C, D]
    k_c = per_chunk(k.repeat_interleave(HV // HK, dim=2))
    v_c = per_chunk(v)
    beta_c = per_chunk(beta[..., None])
    gc = per_chunk(g[..., None]).squeeze(-1).cumsum(-1)  # [B, HV, NC, C]
    h = torch.stack(states, dim=2)  # [B, HV, NC, K, V]

    # N = strict_lower((k * beta) k^T * L), L[i, j] = exp(gc_i - gc_j); v_new = A (v * beta - (k * beta * exp(gc)) h).
    lower = torch.ones(C, C, dtype=torch.bool).tril()
    decay = torch.where(lower, gc[..., :, None] - gc[..., None, :], -torch.inf).exp()
    n = ((k_c * beta_c) @ k_c.transpose(-1, -2) * decay).tril(-1)
    a = torch.linalg.solve_triangular(torch.eye(C, dtype=f64) + n, torch.eye(C, dtype=f64), upper=False)
    v_new = a @ (v_c * beta_c - (k_c * beta_c * gc[..., None].exp()) @ h)

    BH = B * HV
    return (
        torch.cat(outs, dim=1).transpose(1, 2).reshape(BH, T, V),
        state,
        h.reshape(BH, NC, K, V),
        v_new.reshape(BH, T, V),
        a.reshape(BH, T, C),
    )
