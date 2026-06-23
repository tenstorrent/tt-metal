# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sim-only test for the tt-lang Mamba2 SSD scan kernel.

Run:
    cd /home/ttuser/ssinghal/tt-lang && \
    /home/ttuser/ssinghal/tt-lang-venv/bin/python -m pytest \
        /home/ttuser/ssinghal/tt-metal/models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_mamba2_ssd_scan_ttlang.py \
        -v -s
"""
import os as _os
import sys as _sys

_tt_lang_path = _os.environ.get("TT_LANG_PYTHON_PATH", "/home/ttuser/ssinghal/tt-lang/python")
if _tt_lang_path and _tt_lang_path not in _sys.path:
    _sys.path.insert(0, _tt_lang_path)

import pytest
import torch
import torch.nn.functional as F

sim_mod = pytest.importorskip("sim", reason="tt-lang sim not available (set TT_LANG_PYTHON_PATH)")
ttl = sim_mod.ttl
ttnn = sim_mod.ttnn

from sim.ttnnsim import TILE_LAYOUT
from sim.ttnnsim import Tensor as SimTensor  # noqa: E402

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


# ── All-heads SimTensor builder ───────────────────────────────────────────────


def _build_all_heads_sim_tensors(x_dt_raw, B_in_raw, C_in_raw, x_raw, log_decay_raw, D_skip_raw, n_chunks):
    """Build all-heads SimTensors by stacking 64 per-head tensors along tile-row dim.

    Args (all float32, batch dim removed):
        x_dt_raw:     [S, NUM_HEADS, D]
        B_in_raw:     [S, G, N]
        C_in_raw:     [S, G, N]
        x_raw:        [S, NUM_HEADS, D]
        log_decay_raw:[S, NUM_HEADS]
        D_skip_raw:   [NUM_HEADS]
        n_chunks:     int

    Returns dict of SimTensors with keys matching the kernel's parameter names.
    Shapes (elements):
        log_L        [NUM_HEADS * n_chunks * C, C]
        x_dt         [NUM_HEADS * n_chunks * C, D]
        B            [NUM_HEADS * n_chunks * C, N]  (B expanded G→NUM_HEADS)
        C_mat        [NUM_HEADS * n_chunks * C, N]
        x            [NUM_HEADS * n_chunks * C, D]
        log_gamma    [NUM_HEADS * n_chunks * C, TILE]
        log_delta    [NUM_HEADS * n_chunks * C, TILE]
        log_gscalar  [NUM_HEADS * n_chunks * TILE, TILE]
        h_in         [NUM_HEADS * D, N]
        D_skip_t     [NUM_HEADS * TILE, TILE]
        y_out        [NUM_HEADS * n_chunks * C, D]
        h_out        [NUM_HEADS * D, N]
    """
    S = n_chunks * C
    i_idx_t = torch.arange(C).unsqueeze(1).float()  # [C,1]
    s_idx_t = torch.arange(C).unsqueeze(0).float()  # [1,C]
    causal = s_idx_t <= i_idx_t  # [C,C]

    all_log_L = []
    all_xdt = []
    all_B = []
    all_C_mat = []
    all_x = []
    all_lg = []
    all_ld = []
    all_lgs = []
    all_h_in = []
    all_dskip = []
    all_y_out = []
    all_h_out = []

    for h_idx in range(NUM_HEADS):
        g_idx = h_idx * G // NUM_HEADS

        xdt_h = x_dt_raw[:S, h_idx, :].reshape(n_chunks, C, D).to(torch.bfloat16)
        B_h = B_in_raw[:S, g_idx, :].reshape(n_chunks, C, N).to(torch.bfloat16)
        C_h = C_in_raw[:S, g_idx, :].reshape(n_chunks, C, N).to(torch.bfloat16)
        x_h = x_raw[:S, h_idx, :].reshape(n_chunks, C, D).to(torch.bfloat16)
        logd_h = log_decay_raw[:S, h_idx].reshape(n_chunks, C).float()

        A_cumsum = torch.cumsum(logd_h, dim=1)  # [n_chunks, C]

        # log_L: [n_chunks*C, C]
        log_L_h = torch.zeros(n_chunks * C, C, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            diff = A_cumsum[ci].unsqueeze(1) - A_cumsum[ci].unsqueeze(0)  # [C,C]
            masked = torch.where(causal, diff, torch.tensor(float("-inf")))
            log_L_h[ci * C : (ci + 1) * C, :] = masked.to(torch.bfloat16)

        # log_gamma: [n_chunks*C, TILE] — each row filled with A_cumsum[ci, row]
        log_gamma_h = torch.zeros(n_chunks * C, TILE, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            for row in range(C):
                log_gamma_h[ci * C + row, :] = A_cumsum[ci, row]

        # log_delta: [n_chunks*C, TILE] — each row filled with A_last - A_cumsum[ci, row]
        log_delta_h = torch.zeros(n_chunks * C, TILE, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            A_last = A_cumsum[ci, -1]
            for row in range(C):
                log_delta_h[ci * C + row, :] = A_last - A_cumsum[ci, row]

        # log_gscalar: [n_chunks*TILE, TILE] — each TILE-row block filled with A_cumsum[ci,-1]
        log_gscalar_h = torch.zeros(n_chunks * TILE, TILE, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            log_gscalar_h[ci * TILE : (ci + 1) * TILE, :] = A_cumsum[ci, -1].item()

        all_log_L.append(log_L_h)  # [n_chunks*C, C]
        all_xdt.append(xdt_h.reshape(n_chunks * C, D))  # [n_chunks*C, D]
        all_B.append(B_h.reshape(n_chunks * C, N))  # [n_chunks*C, N]
        all_C_mat.append(C_h.reshape(n_chunks * C, N))  # [n_chunks*C, N]
        all_x.append(x_h.reshape(n_chunks * C, D))  # [n_chunks*C, D]
        all_lg.append(log_gamma_h)  # [n_chunks*C, TILE]
        all_ld.append(log_delta_h)  # [n_chunks*C, TILE]
        all_lgs.append(log_gscalar_h)  # [n_chunks*TILE, TILE]
        all_h_in.append(torch.zeros(D, N, dtype=torch.bfloat16))  # [D, N]
        all_dskip.append(
            torch.full((TILE, TILE), float(D_skip_raw[h_idx].item()), dtype=torch.bfloat16)
        )  # [TILE, TILE]
        all_y_out.append(torch.zeros(n_chunks * C, D, dtype=torch.bfloat16))  # [n_chunks*C, D]
        all_h_out.append(torch.zeros(D, N, dtype=torch.bfloat16))  # [D, N]

    def _st(t):
        return SimTensor(t.clone(), TILE_LAYOUT)

    return {
        "log_L": _st(torch.cat(all_log_L, dim=0)),  # [H*n_chunks*C, C]
        "x_dt": _st(torch.cat(all_xdt, dim=0)),  # [H*n_chunks*C, D]
        "B": _st(torch.cat(all_B, dim=0)),  # [H*n_chunks*C, N]
        "C_mat": _st(torch.cat(all_C_mat, dim=0)),  # [H*n_chunks*C, N]
        "x": _st(torch.cat(all_x, dim=0)),  # [H*n_chunks*C, D]
        "log_gamma": _st(torch.cat(all_lg, dim=0)),  # [H*n_chunks*C, TILE]
        "log_delta": _st(torch.cat(all_ld, dim=0)),  # [H*n_chunks*C, TILE]
        "log_gscalar": _st(torch.cat(all_lgs, dim=0)),  # [H*n_chunks*TILE, TILE]
        "h_in": _st(torch.cat(all_h_in, dim=0)),  # [H*D, N]
        "D_skip_t": _st(torch.cat(all_dskip, dim=0)),  # [H*TILE, TILE]
        "y_out": _st(torch.cat(all_y_out, dim=0)),  # [H*n_chunks*C, D]
        "h_out": _st(torch.cat(all_h_out, dim=0)),  # [H*D, N]
    }


# ── Multi-core sim test ───────────────────────────────────────────────────────


@pytest.mark.parametrize("n_chunks", [2, 64, 128, 256])
def test_mamba2_ssd_scan_ttlang_multicore_sim(n_chunks):
    """8×8 multi-core kernel processes all 64 heads in one call (sim)."""
    S = n_chunks * C
    torch.manual_seed(0)

    x_dt_raw = torch.randn(1, S, NUM_HEADS, D) * 0.01
    B_in_raw = torch.randn(1, S, G, N) * 0.1
    C_in_raw = torch.randn(1, S, G, N) * 0.1
    x_raw_in = torch.randn(1, S, NUM_HEADS, D)
    logd_raw = torch.randn(1, S, NUM_HEADS) * 0.01 - 0.5
    D_skip_raw = torch.ones(NUM_HEADS)

    y_ref, h_ref = _ssd_scan_ref(
        x_dt=x_dt_raw.float(),
        B_in=B_in_raw.float(),
        C_in=C_in_raw.float(),
        x_raw=x_raw_in.float(),
        log_decay=logd_raw.float(),
        D_skip=D_skip_raw.float(),
    )

    kernel = make_mamba2_ssd_scan_kernel(n_chunks, num_heads=NUM_HEADS)
    t = _build_all_heads_sim_tensors(
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

    # Extract per-head outputs from the stacked tensors
    y_ttlang = torch.zeros(S, NUM_HEADS, D, dtype=torch.bfloat16)
    h_ttlang = torch.zeros(NUM_HEADS, D, N, dtype=torch.bfloat16)
    for h in range(NUM_HEADS):
        y_row0 = h * n_chunks * C
        y_row1 = (h + 1) * n_chunks * C
        h_row0 = h * D
        h_row1 = (h + 1) * D
        y_ttlang[:, h, :] = t["y_out"]._tensor[y_row0:y_row1, :D]
        h_ttlang[h, :, :] = t["h_out"]._tensor[h_row0:h_row1, :N]

    y_ref_s = y_ref[0, :S].float()
    h_ref_s = h_ref[0].float()

    assert torch.allclose(
        y_ttlang.float(), y_ref_s, atol=1e-4, rtol=1e-4
    ), f"n_chunks={n_chunks}: y max_diff={(y_ttlang.float()-y_ref_s).abs().max():.6f}"
    assert torch.allclose(
        h_ttlang.float(), h_ref_s, atol=1e-4, rtol=1e-4
    ), f"n_chunks={n_chunks}: h max_diff={(h_ttlang.float()-h_ref_s).abs().max():.6f}"
