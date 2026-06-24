#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SSD-scan perf comparison: ttnn op-dispatch path vs tt-lang fused kernel.

Run:
    TT_LANG_PYTHON_PATH=/home/ttuser/ssinghal/tt-lang/build/python_packages \
    python_env/bin/python \
        models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/bench_ssd_scan.py
"""

import os
import sys
import time

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if p not in sys.path:
        sys.path.insert(0, p)

_tt_lang_path = os.environ.get("TT_LANG_PYTHON_PATH", "")
if _tt_lang_path and _tt_lang_path not in sys.path:
    sys.path.insert(0, _tt_lang_path)

import torch

import ttnn

# ── Constants ────────────────────────────────────────────────────────────────
NUM_HEADS = 64
HEAD_DIM = 64
SSM_STATE_SIZE = 128
N_GROUPS = 8
CHUNK_SIZE = 64
TILE = 32

_TL = ttnn.TILE_LAYOUT
_L1 = ttnn.L1_MEMORY_CONFIG
_DR = ttnn.DRAM_MEMORY_CONFIG


def _to_tt(t: torch.Tensor, dev, mc=None) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.bfloat16().contiguous(), dtype=ttnn.bfloat16, layout=_TL, device=dev, memory_config=mc or _DR
    )


def _rr(t, shape, mc=None):
    mc = mc or _DR
    return ttnn.to_layout(
        ttnn.reshape(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc), shape), _TL, memory_config=mc
    )


# ── ttnn-based SSD scan (adapted from mamba2_prefill._mamba2_ssd_chunk) ──────


def _ttnn_ssd_all_chunks(log_decay, x_dt, B_mat, C_mat, D_skip, x_raw, n_chunks, dev):
    """Run the full SSD scan (all n_chunks) using ttnn ops.

    Inputs (all torch, float32, single head for simplicity — we replicate to H=64):
        log_decay: [n_chunks, C]
        x_dt:      [n_chunks, C, D]
        B_mat:     [n_chunks, C, N]
        C_mat:     [n_chunks, C, N]
        D_skip:    scalar float
        x_raw:     [n_chunks, C, D]

    We stack them to [1, H=64, n_chunks*C, dim] for batched ttnn ops.
    Returns total elapsed time in seconds (synchronize-to-synchronize).
    """
    C = CHUNK_SIZE
    H = NUM_HEADS
    D = HEAD_DIM
    N = SSM_STATE_SIZE

    # Build B=1 H=64 tensors by repeating
    # log_decay: [n_chunks*C, H] (each col same value)
    logd_flat = log_decay.reshape(n_chunks * C)  # [S]
    logd_hd = logd_flat.unsqueeze(1).expand(-1, H)  # [S, H]
    logd_tt = _to_tt(logd_hd.unsqueeze(0), dev, _L1)  # [1, S, H]

    xdt_flat = x_dt.reshape(n_chunks * C, D)
    xdt_tt = _to_tt(xdt_flat.unsqueeze(0).unsqueeze(0).expand(1, H, -1, -1), dev, _DR)  # [1, H, S, D]

    B_flat = B_mat.reshape(n_chunks * C, N)
    B_tt = _to_tt(B_flat.unsqueeze(0).unsqueeze(0).expand(1, H, -1, -1), dev, _DR)  # [1, H, S, N]

    C_flat = C_mat.reshape(n_chunks * C, N)
    C_tt = _to_tt(C_flat.unsqueeze(0).unsqueeze(0).expand(1, H, -1, -1), dev, _DR)  # [1, H, S, N]

    x_flat = x_raw.reshape(n_chunks * C, D)
    x_tt = _to_tt(x_flat.unsqueeze(0).unsqueeze(0).expand(1, H, -1, -1), dev, _DR)  # [1, H, S, D]

    D_tt = _to_tt(torch.full((1, H, 1, 1), D_skip), dev, _L1)

    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()

    h_prev = None
    for ci in range(n_chunks):
        s0, s1 = ci * C, (ci + 1) * C

        logd_chunk = ttnn.slice(logd_tt, [0, s0, 0], [1, s1, H], memory_config=_L1)
        xdt_c = ttnn.slice(xdt_tt, [0, 0, s0, 0], [1, H, s1, D])
        B_c = ttnn.slice(B_tt, [0, 0, s0, 0], [1, H, s1, N])
        C_c = ttnn.slice(C_tt, [0, 0, s0, 0], [1, H, s1, N])
        x_c = ttnn.slice(x_tt, [0, 0, s0, 0], [1, H, s1, D])

        # cumsum → γ
        log_cum = ttnn.cumsum(logd_chunk, dim=1, memory_config=_L1)  # [1, C, H]
        log_cum_t = ttnn.permute(log_cum, [0, 2, 1], memory_config=_L1)  # [1, H, C]
        log_cum_col = _rr(log_cum_t, [1, H, C, 1], mc=_L1)
        log_cum_row = _rr(log_cum_t, [1, H, 1, C], mc=_L1)
        log_diff = ttnn.sub(log_cum_col, log_cum_row, memory_config=_L1)
        log_L = ttnn.clamp(log_diff, max=0.0, memory_config=_L1)
        L_raw = ttnn.exp(log_L, memory_config=_L1)

        causal_mask_cpu = (
            torch.tril(torch.ones(C, C, dtype=torch.bfloat16)).unsqueeze(0).unsqueeze(0).expand(1, H, C, C).contiguous()
        )
        causal_mask = _to_tt(causal_mask_cpu, dev, _DR)
        L = ttnn.mul(L_raw, causal_mask, memory_config=_L1)

        # Intra-chunk
        B_perm_T = ttnn.permute(B_c, [0, 1, 3, 2], memory_config=_L1)
        Q_K = ttnn.matmul(C_c, B_perm_T, memory_config=_L1)
        L_QK = ttnn.mul(L, Q_K, memory_config=_L1)
        y_intra = ttnn.matmul(L_QK, xdt_c, memory_config=_L1)

        # Cross-chunk
        gamma = ttnn.exp(log_cum_t, memory_config=_L1)  # [1, H, C]
        if h_prev is not None:
            h_prev_T = ttnn.permute(h_prev, [0, 1, 3, 2], memory_config=_L1)
            y_cross = ttnn.matmul(C_c, h_prev_T, memory_config=_L1)
            gamma_4d = _rr(gamma, [1, H, C, 1], mc=_L1)
            y_perm = ttnn.add(y_intra, ttnn.mul(y_cross, gamma_4d, memory_config=_L1), memory_config=_L1)
        else:
            y_perm = y_intra

        # D-skip
        y_out = ttnn.add(y_perm, ttnn.mul(D_tt, x_c, memory_config=_L1), memory_config=_L1)

        # State update
        gamma_last = _rr(ttnn.slice(log_cum_t, [0, 0, C - 1], [1, H, C], memory_config=_L1), [1, H, 1, 1], mc=_L1)
        gamma_last = ttnn.exp(gamma_last, memory_config=_L1)
        log_last = _rr(ttnn.slice(log_cum_t, [0, 0, C - 1], [1, H, C], memory_config=_L1), [1, H, 1], mc=_L1)
        log_delta = ttnn.sub(_rr(log_last, [1, H, 1], mc=_L1), log_cum_t, memory_config=_L1)
        delta_s = ttnn.exp(log_delta, memory_config=_L1)
        delta_4d = _rr(delta_s, [1, H, C, 1], mc=_L1)
        xdt_scaled = ttnn.mul(xdt_c, delta_4d, memory_config=_L1)
        xdt_scaled_T = ttnn.permute(xdt_scaled, [0, 1, 3, 2], memory_config=_L1)
        state_delta = ttnn.matmul(xdt_scaled_T, B_c, memory_config=_L1)

        if h_prev is not None:
            h_prev = ttnn.add(ttnn.mul(gamma_last, h_prev, memory_config=_L1), state_delta, memory_config=_L1)
        else:
            h_prev = state_delta

    ttnn.synchronize_device(dev)
    return time.perf_counter() - t0


# ── tt-lang fused kernel timing ───────────────────────────────────────────────


def _build_ttlang_inputs(log_decay, x_dt, B_mat, C_mat, D_skip, x_raw, n_chunks, dev):
    """Build the stacked [H*n_chunks*C, dim] inputs expected by the tt-lang kernel."""
    C = CHUNK_SIZE
    H = NUM_HEADS
    D = HEAD_DIM
    N = SSM_STATE_SIZE

    # log_L: lower-triangular decay matrix per chunk, stacked [H * n_chunks * C, C]
    logd_f32 = log_decay.float()  # [n_chunks, C]
    A_cum = torch.cumsum(logd_f32, dim=1)  # [n_chunks, C]
    i_idx = torch.arange(C).float().unsqueeze(1)  # [C, 1]
    s_idx = torch.arange(C).float().unsqueeze(0)  # [1, C]
    causal = s_idx <= i_idx
    logL_chunks = []
    for ci in range(n_chunks):
        diff = A_cum[ci].unsqueeze(1) - A_cum[ci].unsqueeze(0)  # [C, C]
        logL_chunks.append(torch.where(causal, diff, torch.tensor(float("-inf"))).bfloat16())
    logL_per_head = torch.cat(logL_chunks, dim=0)  # [n_chunks*C, C]
    logL_all = logL_per_head.unsqueeze(0).expand(H, -1, -1).reshape(H * n_chunks * C, C)

    # x_dt, B, C_mat, x_raw: [H * n_chunks*C, dim]
    xdt_flat = x_dt.reshape(n_chunks * C, D).bfloat16()
    xdt_all = xdt_flat.unsqueeze(0).expand(H, -1, -1).reshape(H * n_chunks * C, D)
    B_flat = B_mat.reshape(n_chunks * C, N).bfloat16()
    B_all = B_flat.unsqueeze(0).expand(H, -1, -1).reshape(H * n_chunks * C, N)
    C_flat = C_mat.reshape(n_chunks * C, N).bfloat16()
    C_all = C_flat.unsqueeze(0).expand(H, -1, -1).reshape(H * n_chunks * C, N)
    x_flat = x_raw.reshape(n_chunks * C, D).bfloat16()
    x_all = x_flat.unsqueeze(0).expand(H, -1, -1).reshape(H * n_chunks * C, D)

    # log_gamma [H*n_chunks*C, TILE]: A_cumsum[i] broadcast to 32 cols
    lg_row = A_cum.reshape(n_chunks * C).unsqueeze(1).expand(-1, TILE).bfloat16()  # [n_chunks*C, TILE]
    lg_elem = lg_row.unsqueeze(0).expand(H, -1, -1).reshape(H * n_chunks * C, TILE).contiguous()

    # log_delta [H*n_chunks*C, TILE]: (A_cumsum[-1] - A_cumsum[i]) broadcast
    A_last_col = A_cum[:, -1:].expand(-1, C)  # [n_chunks, C]
    ld_row = (A_last_col - A_cum).reshape(n_chunks * C).unsqueeze(1).expand(-1, TILE).bfloat16()
    ld_elem = ld_row.unsqueeze(0).expand(H, -1, -1).reshape(H * n_chunks * C, TILE).contiguous()

    # log_gscalar [H*n_chunks*TILE, TILE]: per-chunk A_cumsum[-1] tile-filled
    lgs_row = A_cum[:, -1].bfloat16().unsqueeze(1).unsqueeze(2).expand(-1, TILE, TILE).reshape(n_chunks * TILE, TILE)
    lgs_elem = lgs_row.unsqueeze(0).expand(H, -1, -1).reshape(H * n_chunks * TILE, TILE).contiguous()

    # h_in: zeros [H*D, N]
    h_in = torch.zeros(H * D, N, dtype=torch.bfloat16)

    # D_skip: [H*TILE, TILE]
    dskip_elem = torch.zeros(H * TILE, TILE, dtype=torch.bfloat16)
    for h in range(H):
        dskip_elem[h * TILE : (h + 1) * TILE, :] = float(D_skip)

    # y_out, h_out: zeros
    y_out = torch.zeros(H * n_chunks * C, D, dtype=torch.bfloat16)
    h_out = torch.zeros(H * D, N, dtype=torch.bfloat16)

    def _tt(t):
        return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=_TL, device=dev, memory_config=_DR)

    return (
        _tt(logL_all),
        _tt(xdt_all),
        _tt(B_all),
        _tt(C_all),
        _tt(x_all),
        _tt(lg_elem),
        _tt(ld_elem),
        _tt(lgs_elem),
        _tt(h_in),
        _tt(dskip_elem),
        _tt(y_out),
        _tt(h_out),
    )


def _ttlang_ssd_all_chunks(kernel, inputs, dev):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    kernel(*inputs)
    ttnn.synchronize_device(dev)
    return time.perf_counter() - t0


# ── Main ──────────────────────────────────────────────────────────────────────


def benchmark(n_chunks_list=(2, 8, 32), n_warmup=2, n_timed=5):
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_ttlang import (
        make_mamba2_ssd_scan_kernel,
    )

    dev = ttnn.open_device(device_id=0)
    print(f"\n{'ISL':>8}  {'n_chunks':>8}  {'ttnn (ms)':>12}  {'tt-lang (ms)':>14}  {'speedup':>9}")
    print("-" * 60)

    torch.manual_seed(0)

    for n_chunks in n_chunks_list:
        C = CHUNK_SIZE
        S = n_chunks * C
        H = NUM_HEADS
        D = HEAD_DIM
        N = SSM_STATE_SIZE

        log_decay = torch.randn(n_chunks, C) * 0.01 - 0.5
        x_dt = torch.randn(n_chunks, C, D) * 0.01
        B_mat = torch.randn(n_chunks, C, N) * 0.1
        C_mat = torch.randn(n_chunks, C, N) * 0.1
        D_skip = 1.0
        x_raw = torch.randn(n_chunks, C, D)

        # Build tt-lang inputs once
        kernel = make_mamba2_ssd_scan_kernel(n_chunks, num_heads=H)
        ttl_inputs = _build_ttlang_inputs(log_decay, x_dt, B_mat, C_mat, D_skip, x_raw, n_chunks, dev)

        # Warmup both
        for _ in range(n_warmup):
            _ttnn_ssd_all_chunks(log_decay, x_dt, B_mat, C_mat, D_skip, x_raw, n_chunks, dev)
            _ttlang_ssd_all_chunks(kernel, ttl_inputs, dev)

        # Time ttnn
        ttnn_times = []
        for _ in range(n_timed):
            ttnn_times.append(_ttnn_ssd_all_chunks(log_decay, x_dt, B_mat, C_mat, D_skip, x_raw, n_chunks, dev))

        # Time tt-lang
        ttlang_times = []
        for _ in range(n_timed):
            ttlang_times.append(_ttlang_ssd_all_chunks(kernel, ttl_inputs, dev))

        ttnn_ms = 1e3 * sum(ttnn_times) / len(ttnn_times)
        ttlang_ms = 1e3 * sum(ttlang_times) / len(ttlang_times)
        speedup = ttnn_ms / ttlang_ms if ttlang_ms > 0 else float("inf")

        print(f"{S:>8}  {n_chunks:>8}  {ttnn_ms:>12.1f}  {ttlang_ms:>14.1f}  {speedup:>8.1f}x")

    ttnn.close_device(dev)
    print()


if __name__ == "__main__":
    benchmark(n_chunks_list=[2, 8, 32, 64, 4096])
