# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hardware PCC test for the tt-lang multi-core Mamba2 SSD scan kernel.

Runs ONLY on hardware with python_env/bin/python (Python 3.10 + real ttnn).

Run:
    cd <tt-metal-repo> && \
    TT_LANG_PYTHON_PATH=<tt-lang-repo>/python \
    <tt-lang-venv>/bin/python -m pytest \
        models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_mamba2_ssd_scan_ttlang_hw.py \
        --noconftest -v -s
"""
import os as _os
import sys as _sys

# Add tt-lang to path so the kernel module can import ttl
_tt_lang_path = _os.environ.get("TT_LANG_PYTHON_PATH", "")
if _tt_lang_path and _tt_lang_path not in _sys.path:
    _sys.path.insert(0, _tt_lang_path)

import torch
import torch.nn.functional as F

import ttnn
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


# ── Hardware tensor builder ───────────────────────────────────────────────────


def _build_all_heads_hw_tensors(device, x_dt_raw, B_in_raw, C_in_raw, x_raw, log_decay_raw, D_skip_raw, n_chunks):
    """Build all-heads ttnn device tensors for the hardware kernel.

    Same logic as _build_all_heads_sim_tensors in the sim test, but returns
    ttnn.Tensor objects on device instead of SimTensors.

    Args (all float32, batch dim removed):
        device:        ttnn device
        x_dt_raw:      [S, NUM_HEADS, D]
        B_in_raw:      [S, G, N]
        C_in_raw:      [S, G, N]
        x_raw:         [S, NUM_HEADS, D]
        log_decay_raw: [S, NUM_HEADS]
        D_skip_raw:    [NUM_HEADS]
        n_chunks:      int

    Returns dict with ttnn.Tensor values on device.
    """
    S = n_chunks * C
    i_idx_t = torch.arange(C).unsqueeze(1).float()
    s_idx_t = torch.arange(C).unsqueeze(0).float()
    causal = s_idx_t <= i_idx_t

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
        A_cumsum = torch.cumsum(logd_h, dim=1)

        log_L_h = torch.zeros(n_chunks * C, C, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            diff = A_cumsum[ci].unsqueeze(1) - A_cumsum[ci].unsqueeze(0)
            masked = torch.where(causal, diff, torch.tensor(float("-inf")))
            log_L_h[ci * C : (ci + 1) * C, :] = masked.to(torch.bfloat16)

        log_gamma_h = torch.zeros(n_chunks * C, TILE, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            for row in range(C):
                log_gamma_h[ci * C + row, :] = A_cumsum[ci, row]

        log_delta_h = torch.zeros(n_chunks * C, TILE, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            A_last = A_cumsum[ci, -1]
            for row in range(C):
                log_delta_h[ci * C + row, :] = A_last - A_cumsum[ci, row]

        log_gscalar_h = torch.zeros(n_chunks * TILE, TILE, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            log_gscalar_h[ci * TILE : (ci + 1) * TILE, :] = A_cumsum[ci, -1].item()

        all_log_L.append(log_L_h)
        all_xdt.append(xdt_h.reshape(n_chunks * C, D))
        all_B.append(B_h.reshape(n_chunks * C, N))
        all_C_mat.append(C_h.reshape(n_chunks * C, N))
        all_x.append(x_h.reshape(n_chunks * C, D))
        all_lg.append(log_gamma_h)
        all_ld.append(log_delta_h)
        all_lgs.append(log_gscalar_h)
        all_h_in.append(torch.zeros(D, N, dtype=torch.bfloat16))
        all_dskip.append(torch.full((TILE, TILE), float(D_skip_raw[h_idx].item()), dtype=torch.bfloat16))
        all_y_out.append(torch.zeros(n_chunks * C, D, dtype=torch.bfloat16))
        all_h_out.append(torch.zeros(D, N, dtype=torch.bfloat16))

    def _to_device(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.clone(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return {
        "log_L": _to_device(torch.cat(all_log_L, dim=0)),
        "x_dt": _to_device(torch.cat(all_xdt, dim=0)),
        "B": _to_device(torch.cat(all_B, dim=0)),
        "C_mat": _to_device(torch.cat(all_C_mat, dim=0)),
        "x": _to_device(torch.cat(all_x, dim=0)),
        "log_gamma": _to_device(torch.cat(all_lg, dim=0)),
        "log_delta": _to_device(torch.cat(all_ld, dim=0)),
        "log_gscalar": _to_device(torch.cat(all_lgs, dim=0)),
        "h_in": _to_device(torch.cat(all_h_in, dim=0)),
        "D_skip_t": _to_device(torch.cat(all_dskip, dim=0)),
        "y_out": _to_device(torch.cat(all_y_out, dim=0)),
        "h_out": _to_device(torch.cat(all_h_out, dim=0)),
    }


# ── Hardware PCC test ─────────────────────────────────────────────────────────


def test_mamba2_ssd_scan_ttlang_hw_pcc():
    """Multi-core tt-lang SSD scan on hardware matches PyTorch reference."""
    n_chunks = 2
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

    device = ttnn.open_device(device_id=0)
    try:
        t = _build_all_heads_hw_tensors(
            device,
            x_dt_raw[0].float(),
            B_in_raw[0].float(),
            C_in_raw[0].float(),
            x_raw_in[0].float(),
            logd_raw[0].float(),
            D_skip_raw.float(),
            n_chunks,
        )

        kernel = make_mamba2_ssd_scan_kernel(n_chunks, num_heads=NUM_HEADS)
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

        y_result = ttnn.to_torch(t["y_out"])  # [NUM_HEADS*n_chunks*C, D]
        h_result = ttnn.to_torch(t["h_out"])  # [NUM_HEADS*D, N]

        # Extract per-head outputs and compare
        y_ttlang = torch.zeros(S, NUM_HEADS, D, dtype=torch.bfloat16)
        h_ttlang = torch.zeros(NUM_HEADS, D, N, dtype=torch.bfloat16)
        for h in range(NUM_HEADS):
            y_row0, y_row1 = h * n_chunks * C, (h + 1) * n_chunks * C
            h_row0, h_row1 = h * D, (h + 1) * D
            y_ttlang[:, h, :] = y_result[y_row0:y_row1, :D].to(torch.bfloat16)
            h_ttlang[h, :, :] = h_result[h_row0:h_row1, :N].to(torch.bfloat16)

        y_ref_s = y_ref[0, :S].float()
        h_ref_s = h_ref[0].float()

        assert torch.allclose(
            y_ttlang.float(), y_ref_s, atol=1e-2, rtol=1e-2
        ), f"y max_diff={(y_ttlang.float()-y_ref_s).abs().max():.4f}"
        assert torch.allclose(
            h_ttlang.float(), h_ref_s, atol=1e-2, rtol=1e-2
        ), f"h max_diff={(h_ttlang.float()-h_ref_s).abs().max():.4f}"

    finally:
        ttnn.close_device(device)
