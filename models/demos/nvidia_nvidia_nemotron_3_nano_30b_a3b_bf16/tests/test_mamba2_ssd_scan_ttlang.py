# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sim-only test for the tt-lang Mamba2 SSD scan kernel.

Run:
    cd /home/ttuser/ssinghal/tt-lang && \
    /home/ttuser/ssinghal/tt-lang-venv/bin/python -m pytest \
        /home/ttuser/ssinghal/tt-metal/models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_mamba2_ssd_scan_ttlang.py \
        -v -s
"""
import sys

sys.path.insert(0, "/home/ttuser/ssinghal/tt-lang/python")

import torch
import torch.nn.functional as F
from sim.ttnnsim import TILE_LAYOUT
from sim.ttnnsim import Tensor as SimTensor

from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_ttlang import (
    make_mamba2_ssd_scan_kernel,
)

# ── Constants ────────────────────────────────────────────────────────────────
TILE = 32
NUM_HEADS = 64
D = 64  # HEAD_DIM
N = 128  # SSM_STATE_SIZE
G = 8  # N_GROUPS
C = 64  # CHUNK_SIZE


# ── PyTorch reference (copied from test_mamba2_ssd_scan.py) ──────────────────


def _segment_sum_torch(x: torch.Tensor) -> torch.Tensor:
    cs = x.size(-1)
    t = x[..., None].expand(*x.size(), cs)
    mask_lower = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool), diagonal=-1)
    t = t.masked_fill(~mask_lower, 0)
    seg = torch.cumsum(t, dim=-2)
    mask_diag = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool), diagonal=0)
    return seg.masked_fill(~mask_diag, float("-inf"))


def _ssd_scan_ref(x_dt, B_in, C_in, x_raw, log_decay, D_skip, chunk_size=C):
    """Pure-PyTorch reference. Returns (y [B,S_pad,H,D], h_next [B,H,D,N])."""
    B, S, H, D_dim = x_dt.shape
    G_dim, N_dim = B_in.shape[2], B_in.shape[3]
    reps = H // G_dim
    B_f = B_in.repeat_interleave(reps, dim=2)
    C_f = C_in.repeat_interleave(reps, dim=2)
    pad_size = (chunk_size - S % chunk_size) % chunk_size
    x_pad = F.pad(x_raw, (0, 0, 0, 0, 0, pad_size))
    D_residual = D_skip[None, None, :, None] * x_pad

    def _chunk4(t):
        t = F.pad(t, (0, 0, 0, 0, 0, pad_size))
        return t.reshape(B, -1, chunk_size, t.shape[2], t.shape[3])

    def _chunk3(t):
        t = F.pad(t, (0, 0, 0, pad_size))
        return t.reshape(B, -1, chunk_size, t.shape[2])

    x_c = _chunk4(x_dt)
    B_c = _chunk4(B_f)
    C_c = _chunk4(C_f)
    A_c = _chunk3(log_decay)
    A_c_h = A_c.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_c_h, dim=-1)

    L = torch.exp(_segment_sum_torch(A_c_h))
    G_mat = (C_c[:, :, :, None, :, :] * B_c[:, :, None, :, :, :]).sum(dim=-1)
    M = G_mat * L.permute(0, 2, 3, 4, 1)
    Y_diag = (M[..., None] * x_c[:, :, None]).sum(dim=3)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B_c * decay_states.permute(0, 2, 3, 1)[..., None]
    states = (B_decay[..., None, :] * x_c[..., None]).sum(dim=2)

    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    A_cumsum_last = A_cumsum[:, :, :, -1]
    decay_chunk = torch.exp(_segment_sum_torch(F.pad(A_cumsum_last, (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, h_next = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = (C_c[..., None, :] * states[:, :, None, ...]).sum(-1)
    Y_off = Y_off * state_decay_out.permute(0, 2, 3, 1)[..., None]
    y = (Y_diag + Y_off).reshape(B, -1, H, D_dim) + D_residual
    return y, h_next


# ── SimTensor construction helper ─────────────────────────────────────────────


def _build_per_head_sim_tensors(h_idx, x_dt_raw, B_in_raw, C_in_raw, x_raw, log_decay_raw, D_skip_raw, n_chunks):
    """
    Build all SimTensors (TILE_LAYOUT) for one head h_idx.

    Args (all float32, batch dim removed):
        x_dt_raw:    [S, H, D]
        B_in_raw:    [S, G, N]
        C_in_raw:    [S, G, N]
        x_raw:       [S, H, D]
        log_decay_raw: [S, H]
        D_skip_raw:  [H]

    Returns dict with SimTensors ready to pass to the tt-lang kernel.
    """
    S = n_chunks * C
    g_idx = h_idx * G // NUM_HEADS  # group index for this head

    # Per-head raw values, reshaped to [n_chunks, C, dim]
    xdt_h = x_dt_raw[:S, h_idx, :].reshape(n_chunks, C, D).to(torch.bfloat16)
    B_h = B_in_raw[:S, g_idx, :].reshape(n_chunks, C, N).to(torch.bfloat16)
    C_h = C_in_raw[:S, g_idx, :].reshape(n_chunks, C, N).to(torch.bfloat16)
    x_h = x_raw[:S, h_idx, :].reshape(n_chunks, C, D).to(torch.bfloat16)
    logd_h = log_decay_raw[:S, h_idx].reshape(n_chunks, C).float()

    # Intra-chunk cumulative sum of log_decay → [n_chunks, C]
    A_cumsum = torch.cumsum(logd_h, dim=1)  # [n_chunks, C]

    # ── log_L: [n_chunks*C, C] element shape ───────────────────────────────
    # log_L[c, i, s] = A_cumsum[c,i] - A_cumsum[c,s] for s<=i, else -inf
    i_idx_t = torch.arange(C).unsqueeze(1).float()  # [C, 1]
    s_idx_t = torch.arange(C).unsqueeze(0).float()  # [1, C]
    causal = s_idx_t <= i_idx_t  # [C, C]

    log_L_elem = torch.zeros(n_chunks * C, C, dtype=torch.bfloat16)
    for ci in range(n_chunks):
        diff = A_cumsum[ci].unsqueeze(1) - A_cumsum[ci].unsqueeze(0)  # [C, C]
        masked = torch.where(causal, diff, torch.tensor(float("-inf")))
        log_L_elem[ci * C : (ci + 1) * C, :] = masked.to(torch.bfloat16)

    # ── x_dt: [n_chunks*C, D] ─────────────────────────────────────────────
    xdt_elem = xdt_h.reshape(n_chunks * C, D)

    # ── B, C matrices: [n_chunks*C, N] ────────────────────────────────────
    B_elem = B_h.reshape(n_chunks * C, N)
    C_elem = C_h.reshape(n_chunks * C, N)
    x_elem = x_h.reshape(n_chunks * C, D)

    # ── log_gamma (column vec, fill across TILE cols): [n_chunks*C, TILE] ─
    log_gamma_elem = torch.zeros(n_chunks * C, TILE, dtype=torch.bfloat16)
    for ci in range(n_chunks):
        for row in range(C):
            log_gamma_elem[ci * C + row, :] = A_cumsum[ci, row]

    # ── log_delta (column vec): [n_chunks*C, TILE] ────────────────────────
    log_delta_elem = torch.zeros(n_chunks * C, TILE, dtype=torch.bfloat16)
    for ci in range(n_chunks):
        A_last = A_cumsum[ci, -1]
        for row in range(C):
            log_delta_elem[ci * C + row, :] = A_last - A_cumsum[ci, row]

    # ── log_gscalar (per-chunk scalar tile): [n_chunks*TILE, TILE] ────────
    log_gscalar_elem = torch.zeros(n_chunks * TILE, TILE, dtype=torch.bfloat16)
    for ci in range(n_chunks):
        scalar = A_cumsum[ci, -1].item()
        log_gscalar_elem[ci * TILE : (ci + 1) * TILE, :] = scalar

    # ── h_in: zeros [D, N] (h_prev = 0 at start) ─────────────────────────
    h_in_elem = torch.zeros(D, N, dtype=torch.bfloat16)

    # ── D_skip: scalar tile [TILE, TILE] ─────────────────────────────────
    d_skip_val = float(D_skip_raw[h_idx].item())
    d_skip_elem = torch.full((TILE, TILE), d_skip_val, dtype=torch.bfloat16)

    # ── Output tensors (pre-allocated zeros) ─────────────────────────────
    y_out_elem = torch.zeros(n_chunks * C, D, dtype=torch.bfloat16)
    h_out_elem = torch.zeros(D, N, dtype=torch.bfloat16)

    def _st(t):
        return SimTensor(t.clone(), TILE_LAYOUT)

    return {
        "log_L": _st(log_L_elem),
        "x_dt": _st(xdt_elem),
        "B": _st(B_elem),
        "C_mat": _st(C_elem),
        "x": _st(x_elem),
        "log_gamma": _st(log_gamma_elem),
        "log_delta": _st(log_delta_elem),
        "log_gscalar": _st(log_gscalar_elem),
        "h_in": _st(h_in_elem),
        "D_skip_t": _st(d_skip_elem),
        "y_out": _st(y_out_elem),
        "h_out": _st(h_out_elem),
        # raw tensors for output extraction
        "_y_out_data": y_out_elem,
        "_h_out_data": h_out_elem,
    }


# ── Test ─────────────────────────────────────────────────────────────────────


def test_mamba2_ssd_scan_ttlang_sim_n2():
    """n_chunks=2 (ISL=128): tt-lang sim output matches PyTorch reference."""
    n_chunks = 2
    S = n_chunks * C
    torch.manual_seed(0)

    # Raw inputs (float32 for reference, converted to bf16 in helper)
    x_dt_raw = torch.randn(1, S, NUM_HEADS, D) * 0.01
    B_in_raw = torch.randn(1, S, G, N) * 0.1
    C_in_raw = torch.randn(1, S, G, N) * 0.1
    x_raw_in = torch.randn(1, S, NUM_HEADS, D)
    logd_raw = torch.randn(1, S, NUM_HEADS) * 0.01 - 0.5  # negative log-decay
    D_skip_raw = torch.ones(NUM_HEADS)

    # Reference (float32)
    y_ref, h_ref = _ssd_scan_ref(
        x_dt=x_dt_raw.float(),
        B_in=B_in_raw.float(),
        C_in=C_in_raw.float(),
        x_raw=x_raw_in.float(),
        log_decay=logd_raw.float(),
        D_skip=D_skip_raw.float(),
    )
    # y_ref: [1, S_pad, H, D]  h_ref: [1, H, D, N]

    kernel = make_mamba2_ssd_scan_kernel(n_chunks)

    y_ttlang = torch.zeros(S, NUM_HEADS, D, dtype=torch.bfloat16)
    h_ttlang = torch.zeros(NUM_HEADS, D, N, dtype=torch.bfloat16)

    for h_idx in range(NUM_HEADS):
        t = _build_per_head_sim_tensors(
            h_idx,
            x_dt_raw[0].float(),
            B_in_raw[0].float(),
            C_in_raw[0].float(),
            x_raw_in[0].float(),
            logd_raw[0].float(),
            D_skip_raw.float(),
            n_chunks,
        )
        kernel(
            t["log_L"],
            t["x_dt"],
            t["B"],
            t["C_mat"],
            t["x"],
            t["log_gamma"],
            t["log_delta"],
            t["log_gscalar"],
            t["h_in"],
            t["D_skip_t"],
            t["y_out"],
            t["h_out"],
        )
        # Extract output from SimTensor underlying torch tensor
        y_ttlang[:, h_idx, :] = t["y_out"]._tensor[:S, :D]
        h_ttlang[h_idx, :, :] = t["h_out"]._tensor[:D, :N]

    y_ref_s = y_ref[0, :S].float()  # [S, H, D]
    h_ref_s = h_ref[0].float()  # [H, D, N]

    assert torch.allclose(
        y_ttlang.float(), y_ref_s, atol=1e-2, rtol=1e-2
    ), f"y mismatch: max_diff={( y_ttlang.float()-y_ref_s).abs().max():.4f}"
    assert torch.allclose(
        h_ttlang.float(), h_ref_s, atol=1e-2, rtol=1e-2
    ), f"h mismatch: max_diff={(h_ttlang.float()-h_ref_s).abs().max():.4f}"
