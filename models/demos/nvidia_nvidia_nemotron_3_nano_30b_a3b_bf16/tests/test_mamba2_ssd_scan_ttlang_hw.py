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

import pytest
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


# ── PyTorch reference ─────────────────────────────────────────────────────────


def _ssd_scan_ref(x_dt, B_in, C_in, x_raw, log_decay, D_skip, chunk_size=C):
    """Pure-PyTorch reference, iterative over chunks — O(C²·H) memory per step.

    The original batched formulation allocates O(n_chunks²·H·D·N) for the
    cross-chunk state accumulation, which OOMs at n_chunks≥512.  This version
    loops over chunks and updates h incrementally, keeping peak memory small.
    """
    B, S, H, D_dim = x_dt.shape
    G_dim, N_dim = B_in.shape[2], B_in.shape[3]
    reps = H // G_dim
    B_f = B_in.repeat_interleave(reps, dim=2)  # [B, S, H, N]
    C_f = C_in.repeat_interleave(reps, dim=2)

    pad_size = (chunk_size - S % chunk_size) % chunk_size
    n_chunks = (S + pad_size) // chunk_size

    xdt_p = F.pad(x_dt, (0, 0, 0, 0, 0, pad_size))
    B_p = F.pad(B_f, (0, 0, 0, 0, 0, pad_size))
    C_p = F.pad(C_f, (0, 0, 0, 0, 0, pad_size))
    x_p = F.pad(x_raw, (0, 0, 0, 0, 0, pad_size))
    ld_p = F.pad(log_decay, (0, 0, 0, pad_size))

    causal = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool))

    y_out = torch.zeros(B, n_chunks * chunk_size, H, D_dim, dtype=x_dt.dtype)
    h = torch.zeros(B, H, D_dim, N_dim, dtype=x_dt.dtype)  # running state [B,H,D,N]

    for ci in range(n_chunks):
        s, e = ci * chunk_size, (ci + 1) * chunk_size
        xdt_c = xdt_p[:, s:e]  # [B, C, H, D]
        B_c = B_p[:, s:e]  # [B, C, H, N]
        C_c = C_p[:, s:e]  # [B, C, H, N]
        x_c = x_p[:, s:e]  # [B, C, H, D]
        ld_c = ld_p[:, s:e]  # [B, C, H]

        A_cum = torch.cumsum(ld_c, dim=1)  # [B, C, H]
        A_last = A_cum[:, -1, :]  # [B, H]

        # Intra-chunk: L[i,s] = exp(A_cum[i]-A_cum[s]) lower-triangular causal
        diff = A_cum[:, :, None, :] - A_cum[:, None, :, :]  # [B, C_i, C_s, H]
        diff.masked_fill_(~causal[None, :, :, None], float("-inf"))
        L = torch.exp(diff)  # [B, C_i, C_s, H]
        CB = (C_c[:, :, None] * B_c[:, None]).sum(-1)  # [B, C_i, C_s, H]
        # y_intra[b,i,h,d] = sum_s (L*CB)[b,i,s,h] * xdt_c[b,s,h,d]
        LCB = (L * CB).permute(0, 3, 1, 2)  # [B, H, C_i, C_s]
        y_intra = torch.einsum("bhis,bshd->bihd", LCB, xdt_c)  # [B, C, H, D]

        # Cross-chunk: exp(A_cum[i]) * C_c[i] @ h
        gamma = torch.exp(A_cum)  # [B, C, H]
        y_cross = torch.einsum("bchn,bhdn->bchd", C_c, h) * gamma[:, :, :, None]

        y_out[:, s:e] = y_intra + y_cross + D_skip[None, None, :, None] * x_c

        # State update: h = exp(A_last)*h + sum_s exp(A_last-A_cum[s])*xdt_c[s]⊗B_c[s]
        delta = torch.exp(A_last[:, None, :] - A_cum)  # [B, C, H]
        xdt_sc = xdt_c * delta[:, :, :, None]  # [B, C, H, D]
        dh = torch.einsum("bchd,bchn->bhdn", xdt_sc, B_c)  # [B, H, D, N]
        h = torch.exp(A_last)[:, :, None, None] * h + dh

    return y_out, h


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

    # Build B/C in group format [G*n_chunks*C, N] (8x less DRAM than head-expanded)
    for g_idx in range(G):
        B_g = B_in_raw[:S, g_idx, :].reshape(n_chunks, C, N).to(torch.bfloat16)
        C_g = C_in_raw[:S, g_idx, :].reshape(n_chunks, C, N).to(torch.bfloat16)
        all_B.append(B_g.reshape(n_chunks * C, N))
        all_C_mat.append(C_g.reshape(n_chunks * C, N))

    # Build per-head tensors (log_L, x_dt, x, log_gamma, log_delta, log_gscalar, h_in, D_skip, outputs)
    for h_idx in range(NUM_HEADS):
        xdt_h = x_dt_raw[:S, h_idx, :].reshape(n_chunks, C, D).to(torch.bfloat16)
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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = (a.pow(2).sum() * b.pow(2).sum()).sqrt()
    return ((a * b).sum() / denom).item() if denom.item() != 0.0 else 1.0


@pytest.mark.parametrize("n_chunks", [2, 16, 64, 128, 256, 512, 1024, 2048, 4096])
def test_mamba2_ssd_scan_ttlang_hw_pcc(n_chunks):
    """Multi-core tt-lang SSD scan on hardware matches PyTorch reference."""
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

        kernel = make_mamba2_ssd_scan_kernel(n_chunks, num_heads=NUM_HEADS, n_groups=G)
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

        y_pcc = _pcc(y_ttlang, y_ref_s)
        h_pcc = _pcc(h_ttlang, h_ref_s)
        y_max = (y_ttlang.float() - y_ref_s).abs().max().item()
        h_max = (h_ttlang.float() - h_ref_s).abs().max().item()
        print(
            f"\nn_chunks={n_chunks:4d}  ISL={S:6d}  "
            f"y_PCC={y_pcc:.6f}  h_PCC={h_pcc:.6f}  "
            f"y_maxdiff={y_max:.4f}  h_maxdiff={h_max:.4f}  "
            f"{'PASS' if y_pcc > 0.99 and h_pcc > 0.99 else 'FAIL'}",
            flush=True,
        )

        assert y_pcc > 0.99, f"n_chunks={n_chunks}: y_PCC={y_pcc:.6f} < 0.99"
        assert h_pcc > 0.99, f"n_chunks={n_chunks}: h_PCC={h_pcc:.6f} < 0.99"

    finally:
        ttnn.close_device(device)
