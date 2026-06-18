# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mamba2 S>1 prefill for NemotronH-30B — chunked SSD scan on QB TP=4.

Why this exists
---------------
The S=1 decode path in mamba2_layer.py processes one token at a time: each
token costs ~45 ms → a 256 k-token context costs ~3.3 hours. For prefill we
can process all S tokens in a *single* forward pass by reformulating the SSM
recurrence as a blocked parallel prefix scan (SSD: Structured State Space
Duality, Dao & Gu 2024).

Algorithm (SSD chunked scan)
----------------------------
For a chunk of C consecutive tokens, the Mamba2 linear recurrence

    h[t] = decay[t] * h[t-1]  +  outer(x_dt[t], B[t])          [H, D, N]
    y[t] = C[t] @ h[t]  +  D_skip * x[t]                        [H, D]

can be split into:

    (a) Intra-chunk parallel part
        Decay matrix  L[i, s] = exp(Σ_{u=s+1}^{i} log decay[u])  (lower-tri)
        Similarity    Q_K[i,s] = C[i] · B[s]                       [C, C]
        Intra output  y_intra  = (L ⊙ Q_K) @ (x_dt)               [C, H, D]

    (b) Cross-chunk carry from previous state h_prev
        Cumulative decay  γ[t] = exp(Σ_{u=0}^{t} log decay[u])
        y_cross[t]  = γ[t] * (C[t] @ h_prev)                      [H, D]

    (c) State update for the NEXT chunk
        h_next = γ[C-1] * h_prev  +  Σ_s γ_from_s_to_C * outer(x_dt[s], B[s])

References to GDN chunked prefill
----------------------------------
Same chunk-scan pattern used by ``chunk_gated_delta_rule_ttnn`` in
``models/demos/qwen3_6_galaxy_v2/tt/qwen35_chunk_delta_rule_ops.py``
(branch ssinghal/qwen36_vlm).  GDN uses the DeltaNet recurrence; this
file implements the SSD (diagonal-A Mamba2) recurrence.

TT-Lang integration
-------------------
The per-chunk dt_eff / decay / x_dt preprocessing chain is fused into one
tt-lang kernel (``kernels/mamba2_ssm_inputs_ttlang.py``).  The heavy matmuls
(in_proj, causal conv, out_proj, Q_K attention) use standard TTNN ops.

Entry point
-----------
    mamba2_prefill_layer_forward(mesh_device, hidden_states [B, S, 2688], ...)
        → (output [B, S, 2688], ssm_state_new [B, H, D, N], conv_state_new)

``mamba2_layer.py`` calls this for S > 1 and the existing decode path for S == 1.
"""

from __future__ import annotations

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _col, _rep_keyed, all_gather

# ---------------------------------------------------------------------------
# Constants (must match mamba2_layer.py exactly)
# ---------------------------------------------------------------------------
NUM_HEADS = 64
HEAD_DIM = 64
N_GROUPS = 8
SSM_STATE_SIZE = 128
INTERMEDIATE_SIZE = NUM_HEADS * HEAD_DIM  # 4096
CONV_DIM = INTERMEDIATE_SIZE + 2 * N_GROUPS * SSM_STATE_SIZE  # 6144
NORM_EPS = 1e-5
HEADS_PER_GROUP = NUM_HEADS // N_GROUPS  # 8
CONV_KERNEL = 4  # causal conv1d kernel size
CHUNK_SIZE = 64  # must be multiple of 32

_RM = ttnn.ROW_MAJOR_LAYOUT
_TL = ttnn.TILE_LAYOUT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rr(t: ttnn.Tensor, shape: list) -> ttnn.Tensor:
    """TILE → RM → reshape → TILE (avoids BH relayout-kernel deadlock)."""
    return ttnn.to_layout(ttnn.reshape(ttnn.to_layout(t, _RM), shape), _TL)


def _expand_groups(
    flat: ttnn.Tensor,  # [B, S, N_GROUPS, SSM_STATE_SIZE]
) -> ttnn.Tensor:
    """Repeat each group slice HEADS_PER_GROUP times → [B, S, NUM_HEADS, N]."""
    B, S, G, N = flat.shape[0], flat.shape[1], flat.shape[2], flat.shape[3]
    slices = []
    for g in range(G):
        g_slice = ttnn.slice(flat, [0, 0, g, 0], [B, S, g + 1, N])
        for _ in range(HEADS_PER_GROUP):
            slices.append(g_slice)
    return ttnn.concat(slices, dim=2)  # [B, S, H, N]


# ---------------------------------------------------------------------------
# Causal conv1d for prefill
# ---------------------------------------------------------------------------


def _causal_conv1d_prefill(
    hBC: ttnn.Tensor,  # [B, S, CONV_DIM] bf16 on device
    conv_weight: torch.Tensor,  # [CONV_DIM, 1, CONV_KERNEL] CPU
    conv_bias: torch.Tensor,  # [CONV_DIM] CPU
    mesh_device: MeshDevice,
    conv_state: tuple | None = None,  # (h_tm3, h_tm2, h_tm1) each [B,1,CONV_DIM]
) -> tuple[ttnn.Tensor, tuple]:
    """Depthwise causal conv1d with kernel_size=4 for a full S-token sequence.

    Implements:  out[s, c] = Σ_k w[c, k] * hBC[s - (K-1) + k, c] + bias[c]
    with causal zero-padding at the start.

    When conv_state is provided (tokens from a previous forward pass), the
    three history tokens are prepended before the causal pad.

    Returns (hBC_conv [B, S, CONV_DIM], conv_state_new).
    conv_state_new = (hBC[S-3], hBC[S-2], hBC[S-1]) for the next decode step.
    """
    B = hBC.shape[0]
    S = hBC.shape[1]

    # Build the 4-token history + current sequence:
    # history: (h_tm3, h_tm2, h_tm1) from conv_state, else zeros
    if conv_state is not None:
        h_tm3, h_tm2, h_tm1 = conv_state
        hist = ttnn.concat([h_tm3, h_tm2, h_tm1], dim=1)  # [B, 3, CONV_DIM]
    else:
        hist = ttnn.zeros(
            [B, CONV_KERNEL - 1, CONV_DIM],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=_TL,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Padded input: [B, S+3, CONV_DIM]
    padded = ttnn.concat([hist, hBC], dim=1)

    # Depthwise conv1d: out[s] = Σ_k w[:,k] * padded[:,s+k]   for k in [0..3]
    # Upload per-tap weights to device (replicated).
    # Keys match the decode path in mamba2_layer.py so prefill gets cache hits
    # instead of re-uploading the same weights — duplicate uploads risk landing
    # on device-2's defective DRAM pages and accumulating persistent L1 RM fallbacks.
    out = None
    for k in range(CONV_KERNEL):
        w_k = _rep_keyed(
            ("conv_w", id(conv_weight), k),  # same key as decode path
            conv_weight[:, 0, k].bfloat16().unsqueeze(0).unsqueeze(0).contiguous(),
            mesh_device,
        )
        tap = ttnn.slice(padded, [0, k, 0], [B, k + S, CONV_DIM])
        contribution = ttnn.mul(tap, w_k)
        out = contribution if out is None else ttnn.add(out, contribution)

    bias_tt = _rep_keyed(
        id(conv_bias),  # same key as decode path
        conv_bias.bfloat16().unsqueeze(0).unsqueeze(0).contiguous(),
        mesh_device,
    )
    out = ttnn.add(out, bias_tt)  # [B, S, CONV_DIM]

    # Save last 3 tokens as conv_state for subsequent decode steps
    if S >= 3:
        conv_state_new = (
            ttnn.slice(hBC, [0, S - 3, 0], [B, S - 2, CONV_DIM]),
            ttnn.slice(hBC, [0, S - 2, 0], [B, S - 1, CONV_DIM]),
            ttnn.slice(hBC, [0, S - 1, 0], [B, S, CONV_DIM]),
        )
    else:
        # Very short sequences: pad history with zeros
        zeros_needed = 3 - S
        z = ttnn.zeros(
            [B, zeros_needed, CONV_DIM],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=_TL,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        full_hist = ttnn.concat([z, hBC], dim=1)  # [B, 3, CONV_DIM]
        conv_state_new = (
            ttnn.slice(full_hist, [0, 0, 0], [B, 1, CONV_DIM]),
            ttnn.slice(full_hist, [0, 1, 0], [B, 2, CONV_DIM]),
            ttnn.slice(full_hist, [0, 2, 0], [B, 3, CONV_DIM]),
        )

    return out, conv_state_new


# ---------------------------------------------------------------------------
# SSD chunked scan
# ---------------------------------------------------------------------------


def _mamba2_ssd_chunk(
    decay_chunk: ttnn.Tensor,  # [B, C, H]  — per-step decay for this chunk
    x_dt_chunk: ttnn.Tensor,  # [B, C, H, D]
    B_chunk: ttnn.Tensor,  # [B, C, H, N]
    C_chunk: ttnn.Tensor,  # [B, C, H, N]
    D_tt: ttnn.Tensor,  # [1, 1, H, 1] or [B, 1, H, 1]
    x_chunk: ttnn.Tensor,  # [B, C, H, D]  raw x for D-skip
    h_prev: ttnn.Tensor | None,  # [B, H, D, N]  state before this chunk
    mesh_device: MeshDevice,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Process one chunk of C tokens.

    Returns (y_chunk [B, C, H, D], h_next [B, H, D, N]).
    """
    B = decay_chunk.shape[0]
    C = decay_chunk.shape[1]

    # --- Log-decay cumsum → γ[t] = exp(Σ_{u=0}^{t} log_decay[u]) -------
    log_decay = ttnn.log(ttnn.clamp(decay_chunk, min=1e-6))  # [B, C, H]
    log_decay_cum = ttnn.cumsum(log_decay, dim=1)  # [B, C, H]

    # --- Build lower-triangular decay matrix L[i, s] = exp(Σ_{s+1}^{i} log_decay) ----
    # Compute directly in log space: L[i, s] = exp(log_cum[i] - log_cum[s]).
    # Do NOT compute gamma=exp(log_cum) and divide — when cumulative decays are large
    # and negative (A_log ≥ 1), gamma underflows to 0 in BF16, making L[i,s]=0/0=garbage
    # for nearby (i, s) pairs including the diagonal (which should be 1).
    log_cum_t = ttnn.permute(log_decay_cum, [0, 2, 1])  # [B, H, C]
    log_cum_col = _rr(log_cum_t, [B, NUM_HEADS, C, 1])  # [B, H, C, 1] — log_cum[i]
    log_cum_row = _rr(log_cum_t, [B, NUM_HEADS, 1, C])  # [B, H, 1, C] — log_cum[s]
    # log_L[i, s] = log_cum[i] - log_cum[s]: negative for lower tri, 0 diagonal, positive for upper tri.
    # Clamp to max=0 before exp: upper-tri positive values would overflow to +inf, and
    # +inf * 0 (from causal_mask) = NaN. Clamping to 0 gives exp(0)=1 for upper tri, which
    # then becomes 0 after causal masking. Lower tri / diagonal values are ≤ 0 so unaffected.
    log_L = ttnn.clamp(ttnn.sub(log_cum_col, log_cum_row), max=0.0)  # [B, H, C, C]
    L_raw = ttnn.exp(log_L)  # lower tri: exp(neg)=correct; diagonal: 1; upper tri: 1 → masked to 0

    # Gamma needed for cross-chunk carry and state update.
    # Underflow to 0 is CORRECT there: tiny cumulative decay → near-zero carry.
    gamma = ttnn.exp(log_decay_cum)  # [B, C, H]
    gamma_t = ttnn.permute(gamma, [0, 2, 1])  # [B, H, C]

    # Causal lower-triangular mask (ones on/below diagonal, zeros above)
    ones_cpu = torch.tril(torch.ones(C, C, dtype=torch.bfloat16))
    causal_mask = ttnn.from_torch(
        ones_cpu.unsqueeze(0).unsqueeze(0).expand(B, NUM_HEADS, C, C),
        dtype=ttnn.bfloat16,
        layout=_TL,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    L = ttnn.mul(L_raw, causal_mask)  # [B, H, C, C]  — lower-tri decay matrix

    # --- Intra-chunk output ------------------------------------------------
    # Q_K[i, s] = C[i, h, :] · B[s, h, :]   →  [B, H, C, C]
    # C_chunk: [B, C, H, N] → [B, H, C, N] for batch matmul
    C_perm = ttnn.permute(C_chunk, [0, 2, 1, 3])  # [B, H, C, N]
    B_perm = ttnn.permute(B_chunk, [0, 2, 1, 3])  # [B, H, C, N]
    B_perm_T = ttnn.permute(B_perm, [0, 1, 3, 2])  # [B, H, N, C]
    Q_K = ttnn.matmul(C_perm, B_perm_T)  # [B, H, C, C]

    # Weighted: (L ⊙ Q_K) — element-wise, same [B, H, C, C]
    L_QK = ttnn.mul(L, Q_K)  # [B, H, C, C]

    # x_dt_chunk: [B, C, H, D] → [B, H, C, D]
    xdt_perm = ttnn.permute(x_dt_chunk, [0, 2, 1, 3])  # [B, H, C, D]

    # y_intra = L_QK @ x_dt: [B, H, C, C] @ [B, H, C, D] = [B, H, C, D]
    y_intra = ttnn.matmul(L_QK, xdt_perm)  # [B, H, C, D]

    # --- Cross-chunk carry -------------------------------------------------
    if h_prev is not None:
        # gamma at each position: [B, H, C]
        # y_cross[t] = gamma[t] * (C[t] @ h_prev)  [B, H, D]
        # C_perm: [B, H, C, N], h_prev: [B, H, D, N] → need [B, H, N, D]
        h_prev_T = ttnn.permute(h_prev, [0, 1, 3, 2])  # [B, H, N, D]
        # C_perm @ h_prev_T: [B, H, C, N] @ [B, H, N, D] = [B, H, C, D]
        y_cross_base = ttnn.matmul(C_perm, h_prev_T)  # [B, H, C, D]
        # Scale by gamma: [B, H, C] → [B, H, C, 1] for broadcast
        gamma_4d = _rr(gamma_t, [B, NUM_HEADS, C, 1])  # [B, H, C, 1]
        y_cross = ttnn.mul(y_cross_base, gamma_4d)  # [B, H, C, D]
        y_perm = ttnn.add(y_intra, y_cross)  # [B, H, C, D]
    else:
        y_perm = y_intra

    # --- D-skip connection: y += D * x_raw --------------------------------
    x_perm = ttnn.permute(x_chunk, [0, 2, 1, 3])  # [B, H, C, D]
    y_perm = ttnn.add(y_perm, ttnn.mul(D_tt, x_perm))  # [B, H, C, D]

    # Reshape back: [B, H, C, D] → [B, C, H, D]
    y_chunk = ttnn.permute(y_perm, [0, 2, 1, 3])  # [B, C, H, D]

    # --- State update: h_next = γ[C-1] * h_prev + Σ_s δ_s ⊗ outer(x_dt_s, B_s) ----
    # Total decay for the chunk: gamma[:, C-1, :] = [B, H]
    gamma_last = ttnn.slice(gamma_t, [0, 0, C - 1], [B, NUM_HEADS, C])  # [B, H, 1]
    gamma_last = _rr(gamma_last, [B, NUM_HEADS])  # [B, H]
    gamma_last_4d = _rr(gamma_last, [B, NUM_HEADS, 1, 1])  # [B, H, 1, 1]

    # Delta per step: gamma[s] / gamma[C-1] = decay prod from s to C-1
    # = gamma_last_at_each_s: gamma[C-1] / gamma[s] * decay[s]  simplified as
    # delta[s] = exp(log_decay_cum[C-1] - log_decay_cum[s])
    log_decay_cum_last = ttnn.slice(log_decay_cum, [0, C - 1, 0], [B, C, NUM_HEADS])  # [B, 1, H]
    log_decay_cum_last = ttnn.permute(log_decay_cum_last, [0, 2, 1])  # [B, H, 1]
    log_delta = ttnn.sub(
        _rr(log_decay_cum_last, [B, NUM_HEADS, 1]),  # [B, H, 1] — broadcast
        log_cum_t,  # [B, H, C] — reuse from L computation above
    )
    delta_s = ttnn.exp(log_delta)  # [B, H, C]

    # Accumulate: state_delta = Σ_s delta[s] * outer(x_dt[s], B[s])
    # x_dt: [B, H, C, D], delta_s: [B, H, C] → scale each step
    delta_s_4d = _rr(delta_s, [B, NUM_HEADS, C, 1])  # [B, H, C, 1]
    xdt_scaled = ttnn.mul(xdt_perm, delta_s_4d)  # [B, H, C, D]

    # Sum-outer: Σ_s outer(xdt_s, Bs) = xdt_T @ B_perm  [B, H, D, N]
    #   xdt_scaled: [B, H, C, D] → [B, H, D, C]
    xdt_scaled_T = ttnn.permute(xdt_scaled, [0, 1, 3, 2])  # [B, H, D, C]
    state_delta = ttnn.matmul(xdt_scaled_T, B_perm)  # [B, H, D, N]

    if h_prev is not None:
        h_next = ttnn.add(
            ttnn.mul(gamma_last_4d, h_prev),  # [B, H, 1, 1] * [B, H, D, N]
            state_delta,
        )
    else:
        h_next = state_delta

    return y_chunk, h_next


# ---------------------------------------------------------------------------
# Main prefill entry point
# ---------------------------------------------------------------------------


def mamba2_prefill_layer_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [B, S, 2688] bf16 on device
    norm_weight: torch.Tensor,
    in_proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    norm_mixer_weight: torch.Tensor,
    D: torch.Tensor,
    out_proj_weight: torch.Tensor,
    norm_eps: float = NORM_EPS,
    ssm_state: ttnn.Tensor | None = None,  # [B, H, D, N] — initial state
    conv_state: tuple | None = None,  # (h_tm3, h_tm2, h_tm1) from prior pass
) -> tuple:
    """Mamba2 forward for S > 1 tokens (prefill).

    Returns (output [B, S, 2688], ssm_state_new [B, H, D, N], conv_state_new).
    """
    B = hidden_states.shape[0]
    S = hidden_states.shape[1]

    # For ISL > 65536 (ISL=262K: S≈257K), projected=[B,S,10304] is 5.3 GB —
    # too large to fit alongside model weights (~26 GB) + state (1.57 GB).
    # Split into _S_M_OUTER-token outer chunks so each projected ≤ 1.35 GB.
    # h_prev and conv_state thread sequentially through chunks.
    #
    # Shape bucketing (cf. gpt_oss get_padded_prefill_len): the last partial chunk
    # is right-padded with zeros to _S_M_OUTER so every recursive call compiles
    # the same kernel shapes as full chunks — no new unique L1 binaries from the
    # remainder.  The SSM output at real positions [0:_chunk_S] is causally correct
    # (no future tokens involved), so trimming back is safe for the output tensor.
    # The SSM STATE after a right-padded run is A^pad_len * correct_state (decayed
    # toward zero by the zero inputs), so the padded state is wrong.  Fix: run the
    # last chunk a second time WITHOUT padding to get the correct final state and
    # discard its output.  ssm_state/conv_state are read-only inside the function,
    # so both runs safely share the same pre-chunk state tensors.
    _S_M_OUTER = 65536
    if S > _S_M_OUTER:
        _out_chunks = []
        _hs = ssm_state
        _cs = conv_state
        for _s in range(0, S, _S_M_OUTER):
            _e = min(_s + _S_M_OUTER, S)
            _chunk_S = _e - _s
            _hc = ttnn.slice(hidden_states, [0, _s, 0], [B, _e, hidden_states.shape[2]])
            if _chunk_S < _S_M_OUTER:
                # Last partial chunk: right-pad to _S_M_OUTER for kernel-shape reuse.
                _pad_len = _S_M_OUTER - _chunk_S
                _hc_rm = ttnn.to_layout(_hc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                _zeros = ttnn.zeros(
                    [B, _pad_len, hidden_states.shape[2]],
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                _hc_pad_rm = ttnn.concat([_hc_rm, _zeros], dim=1)
                _hc_rm.deallocate(True)
                _zeros.deallocate(True)
                _hc_pad = ttnn.to_layout(_hc_pad_rm, _TL, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                _hc_pad_rm.deallocate(True)
                # Padded run: output[:_chunk_S] is causally correct; state is decayed — discard.
                _oc_full, _hs_wrong, _cs_wrong = mamba2_prefill_layer_forward(
                    mesh_device,
                    _hc_pad,
                    norm_weight,
                    in_proj_weight,
                    conv1d_weight,
                    conv1d_bias,
                    dt_bias,
                    A_log,
                    norm_mixer_weight,
                    D,
                    out_proj_weight,
                    norm_eps=norm_eps,
                    ssm_state=_hs,
                    conv_state=_cs,
                )
                _hc_pad.deallocate(True)
                _oc = ttnn.slice(_oc_full, [0, 0, 0], [B, _chunk_S, hidden_states.shape[2]])
                _oc_full.deallocate(True)
                _hs_wrong.deallocate(True)
                for _t in _cs_wrong:
                    _t.deallocate(True)
                # Unpadded run: correct final SSM + conv state; output discarded.
                _out_unused, _hs, _cs = mamba2_prefill_layer_forward(
                    mesh_device,
                    _hc,
                    norm_weight,
                    in_proj_weight,
                    conv1d_weight,
                    conv1d_bias,
                    dt_bias,
                    A_log,
                    norm_mixer_weight,
                    D,
                    out_proj_weight,
                    norm_eps=norm_eps,
                    ssm_state=_hs,
                    conv_state=_cs,
                )
                _out_unused.deallocate(True)
            else:
                _oc, _hs, _cs = mamba2_prefill_layer_forward(
                    mesh_device,
                    _hc,
                    norm_weight,
                    in_proj_weight,
                    conv1d_weight,
                    conv1d_bias,
                    dt_bias,
                    A_log,
                    norm_mixer_weight,
                    D,
                    out_proj_weight,
                    norm_eps=norm_eps,
                    ssm_state=_hs,
                    conv_state=_cs,
                )
            _hc.deallocate(True)
            _out_chunks.append(_oc)
        _result = ttnn.concat(_out_chunks, dim=1)
        for _oc in _out_chunks:
            _oc.deallocate(True)
        return _result, _hs, _cs

    residual = hidden_states
    # ---- 1. Pre-block RMSNorm ----------------------------------------
    w_tt = _rep_keyed(id(norm_weight), norm_weight.bfloat16().unsqueeze(0), mesh_device)  # same key as decode path
    normed = ttnn.rms_norm(hidden_states, epsilon=norm_eps, weight=w_tt)

    # ---- 2. in_proj: column-parallel → partial [B, S, 2576]/device → [B, S, 10304] ----
    ip_tt = _col(in_proj_weight, mesh_device)  # [2576, 2688]/device
    _proj_partial = ttnn.linear(normed, ip_tt, transpose_b=True)  # [B, S, 2576]/device
    projected = all_gather(_proj_partial, dim=2)  # [B, S, 10304]
    _proj_partial.deallocate(True)
    normed.deallocate(True)  # no longer needed; frees [B, S, 2688] (0.7 GB at ISL=131K)

    # ---- 3. Split projected ----------------------------------------
    gate = ttnn.slice(projected, [0, 0, 0], [B, S, INTERMEDIATE_SIZE])
    hBC = ttnn.slice(projected, [0, 0, INTERMEDIATE_SIZE], [B, S, INTERMEDIATE_SIZE + CONV_DIM])
    dt_slice = ttnn.slice(
        projected, [0, 0, INTERMEDIATE_SIZE + CONV_DIM], [B, S, INTERMEDIATE_SIZE + CONV_DIM + NUM_HEADS]
    )
    projected.deallocate(True)  # 5.4 GB freed; gate/hBC/dt_slice are copies

    # ---- 4. Causal conv1d ------------------------------------------
    hBC_conv, conv_state_new = _causal_conv1d_prefill(hBC, conv1d_weight, conv1d_bias, mesh_device, conv_state)
    hBC.deallocate(True)  # hBC_conv is the output copy; hBC no longer needed

    # ---- 5. SiLU --------------------------------------------------
    hBC_silu = ttnn.silu(hBC_conv)  # [B, S, 6144]
    hBC_conv.deallocate(True)

    # ---- 6. Split hBC_silu ----------------------------------------
    x_flat = ttnn.slice(hBC_silu, [0, 0, 0], [B, S, INTERMEDIATE_SIZE])
    b_flat = ttnn.slice(hBC_silu, [0, 0, INTERMEDIATE_SIZE], [B, S, INTERMEDIATE_SIZE + N_GROUPS * SSM_STATE_SIZE])
    c_flat = ttnn.slice(hBC_silu, [0, 0, INTERMEDIATE_SIZE + N_GROUPS * SSM_STATE_SIZE], [B, S, CONV_DIM])
    hBC_silu.deallocate(True)  # 3.22 GB freed; x_flat/b_flat/c_flat are copies

    # ---- 7. Reshape for SSM ----------------------------------------
    x_4d = _rr(x_flat, [B, S, NUM_HEADS, HEAD_DIM])  # [B, S, H, D]
    x_flat.deallocate(True)
    B_4d = _rr(b_flat, [B, S, N_GROUPS, SSM_STATE_SIZE])  # [B, S, G, N]
    b_flat.deallocate(True)
    C_4d = _rr(c_flat, [B, S, N_GROUPS, SSM_STATE_SIZE])  # [B, S, G, N]
    c_flat.deallocate(True)

    # ---- 8. Pre-compute dt_eff, decay, x_dt (fused via tt-lang) -----
    dt_bias_tt = _rep_keyed(("pf_dtb", id(dt_bias)), dt_bias.bfloat16().unsqueeze(0).unsqueeze(0), mesh_device)
    A_log_tt = _rep_keyed(("pf_alog", id(A_log)), A_log.float().bfloat16().unsqueeze(0).unsqueeze(0), mesh_device)

    # Import tt-lang fused kernel (falls back to TTNN ops if ttl unavailable)
    from .kernels.mamba2_ssm_inputs_ttlang import compute_ssm_inputs

    decay, x_dt = compute_ssm_inputs(dt_slice, dt_bias_tt, A_log_tt, x_4d, mesh_device)
    # decay: [B, S, H],  x_dt: [B, S, H, D]
    dt_slice.deallocate(True)

    # ---- 9. D skip scalar -------------------------------------------
    D_tt = _rep_keyed(("pf_D", id(D)), D.float().bfloat16().view(1, 1, NUM_HEADS, 1), mesh_device)
    D_tt = _rr(D_tt, [1, NUM_HEADS, 1, 1])  # [1, H, 1, 1]

    # ---- 10. Chunked SSD scan ----------------------------------------
    # B/C are kept in N_GROUPS shape ([B, S, N_GROUPS, N]) and expanded to NUM_HEADS
    # per chunk inside the scan loop.  This avoids ever materialising full-sequence
    # B_exp/C_exp tensors ([B, S, NUM_HEADS, N]) which are 4.3 GB each at ISL=262K.
    # Per-chunk expansion cost: N_GROUPS slices + 1 concat over CHUNK_SIZE=64 tokens → trivial.
    #
    # Pad S to a multiple of CHUNK_SIZE.
    # B_4d/C_4d are padded at N_GROUPS width (8x smaller than the old B_exp/C_exp padding).
    pad_S = (-S) % CHUNK_SIZE
    if pad_S > 0:
        _z_hd = ttnn.zeros(
            [B, pad_S, NUM_HEADS, HEAD_DIM],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=_TL,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _z_gn = ttnn.zeros(
            [B, pad_S, N_GROUPS, SSM_STATE_SIZE],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=_TL,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _z_h = ttnn.zeros(
            [B, pad_S, NUM_HEADS],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=_TL,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        decay_pad = ttnn.concat([decay, _z_h], dim=1)
        decay.deallocate(True)
        _z_h.deallocate(True)

        x_dt_pad = ttnn.concat([x_dt, _z_hd], dim=1)
        x_dt.deallocate(True)

        B_pad = ttnn.concat([B_4d, _z_gn], dim=1)  # [B, S_pad, N_GROUPS, N]
        B_4d.deallocate(True)

        C_pad = ttnn.concat([C_4d, _z_gn], dim=1)  # [B, S_pad, N_GROUPS, N]
        C_4d.deallocate(True)
        _z_gn.deallocate(True)

        x_pad = ttnn.concat([x_4d, _z_hd], dim=1)
        x_4d.deallocate(True)
        _z_hd.deallocate(True)
    else:
        decay_pad = decay
        x_dt_pad = x_dt
        B_pad = B_4d  # [B, S, N_GROUPS, N] — no full-sequence expand needed
        C_pad = C_4d
        x_pad = x_4d

    S_pad = S + pad_S
    num_chunks = S_pad // CHUNK_SIZE

    y_chunks = []
    h_prev = ssm_state  # None → zero initial state (handled in _mamba2_ssd_chunk)

    for c in range(num_chunks):
        t0 = c * CHUNK_SIZE
        t1 = t0 + CHUNK_SIZE

        decay_c = ttnn.slice(decay_pad, [0, t0, 0], [B, t1, NUM_HEADS])
        x_dt_c = ttnn.slice(x_dt_pad, [0, t0, 0, 0], [B, t1, NUM_HEADS, HEAD_DIM])
        _B_g_c = ttnn.slice(B_pad, [0, t0, 0, 0], [B, t1, N_GROUPS, SSM_STATE_SIZE])
        B_c = _expand_groups(_B_g_c)  # [B, C, NUM_HEADS, N] — 64-token expand, ~1 MB
        _C_g_c = ttnn.slice(C_pad, [0, t0, 0, 0], [B, t1, N_GROUPS, SSM_STATE_SIZE])
        C_c = _expand_groups(_C_g_c)  # [B, C, NUM_HEADS, N]
        x_c = ttnn.slice(x_pad, [0, t0, 0, 0], [B, t1, NUM_HEADS, HEAD_DIM])

        y_c, h_prev = _mamba2_ssd_chunk(decay_c, x_dt_c, B_c, C_c, D_tt, x_c, h_prev, mesh_device)
        y_chunks.append(y_c)

    ssm_state_new = h_prev  # [B, H, D, N]

    # Free SSM input arrays — no longer needed after the scan loop.
    # B_pad/C_pad are now [B, S_pad, N_GROUPS, N] (8x smaller than the old B_exp/C_exp).
    # When pad_S==0 they alias B_4d/C_4d directly; when pad_S>0 originals were freed
    # during pad creation above — only the _pad tensors remain here.
    for _t in [B_pad, C_pad, x_dt_pad, x_pad, decay_pad]:
        _t.deallocate(True)

    # ---- 12. Concatenate chunk outputs --------------------------------
    y_full = ttnn.concat(y_chunks, dim=1)  # [B, S_pad, H, D]
    del y_chunks  # individual chunk tensors freed by reference counting
    if pad_S > 0:
        _y_padded = y_full
        y_full = ttnn.slice(_y_padded, [0, 0, 0, 0], [B, S, NUM_HEADS, HEAD_DIM])
        _y_padded.deallocate(True)

    # Flatten: [B, S, H, D] → [B, S, 4096]
    y_flat = _rr(y_full, [B, S, INTERMEDIATE_SIZE])
    y_full.deallocate(True)

    # ---- 13. MambaRMSNormGated ----------------------------------------
    # Eagerly deallocate each large intermediate (~1 GB at ISL=131K) as soon as it
    # is consumed — without these frees, 7 × 1 GB tensors pile up simultaneously and
    # fragment the remaining DRAM enough to block the final reshape at line 523.
    gate_silu = ttnn.silu(gate)  # [B, S, 4096]
    gate.deallocate(True)
    xg = ttnn.mul(y_flat, gate_silu)  # [B, S, 4096]
    y_flat.deallocate(True)
    gate_silu.deallocate(True)

    GROUP_SIZE = INTERMEDIATE_SIZE // N_GROUPS  # 512
    xg_grouped = _rr(xg, [B, S, N_GROUPS, GROUP_SIZE])  # [B, S, 8, 512]
    xg.deallocate(True)
    xg_sq = ttnn.pow(xg_grouped, 2)
    var = ttnn.mean(xg_sq, dim=3, keepdim=True)  # [B, S, 8, 1]
    xg_sq.deallocate(True)
    xg_normed = ttnn.mul(xg_grouped, ttnn.rsqrt(ttnn.add(var, norm_eps)))
    xg_grouped.deallocate(True)
    xg_normed_flat = _rr(xg_normed, [B, S, INTERMEDIATE_SIZE])
    xg_normed.deallocate(True)

    nw_tt = _rep_keyed(
        id(norm_mixer_weight),
        norm_mixer_weight.bfloat16().unsqueeze(0).unsqueeze(0),
        mesh_device,  # same key as decode path
    )
    scan_out = ttnn.mul(xg_normed_flat, nw_tt)  # [B, S, 4096]
    xg_normed_flat.deallocate(True)

    # ---- 14. out_proj: column-parallel → partial [B, S, 672]/device → full via all_gather ----
    # scan_out is replicated (SSM used the gathered in_proj output), so column-parallel
    # is correct: each device computes a different slice of the output rows.
    # Weight: [2688, 4096] → [672, 4096]/device (22 MB → 5.5 MB per layer/device).
    op_tt = _col(out_proj_weight, mesh_device)  # [672, 4096]/device
    _out_partial = ttnn.linear(scan_out, op_tt, transpose_b=True)  # [B, S, 672]/device
    out = all_gather(_out_partial, dim=2)  # [B, S, 2688] full
    _out_partial.deallocate(True)

    # ---- 15. Residual -------------------------------------------------
    return ttnn.add(residual, out), ssm_state_new, conv_state_new
