# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sim test for the DYNAMIC tt-lang Mamba2 SSD scan kernel.

Validates that a SINGLE kernel object from make_mamba2_ssd_scan_kernel_dynamic()
correctly handles multiple n_chunks values (ISL=128 and ISL=256), proving the
single-ELF-for-all-ISLs concept works at the sim level.

Key assertion: _KERNEL_CACHE_DYN has exactly 1 entry after dispatching both ISLs.

Hardware status: SIM ONLY — see mamba2_ssd_scan_ttlang_dynamic.py for gaps.

Run:
    cd /home/ttuser/ssinghal/tt-lang && \\
    /home/ttuser/ssinghal/tt-lang-venv/bin/python -m pytest \\
        /home/ttuser/ssinghal/tt-metal/models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_mamba2_ssd_scan_ttlang_dynamic.py \\
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
ttnn_sim = sim_mod.ttnn

import importlib.util as _ilu
import pathlib as _pl

from sim.ttnnsim import TILE_LAYOUT
from sim.ttnnsim import Tensor as SimTensor

_dyn_path = _pl.Path(__file__).parent.parent / "tt" / "mamba2_ssd_scan_ttlang_dynamic.py"
_spec = _ilu.spec_from_file_location("mamba2_ssd_scan_ttlang_dynamic", _dyn_path)
_dyn_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dyn_mod)
_KERNEL_CACHE_DYN = _dyn_mod._KERNEL_CACHE_DYN
make_mamba2_ssd_scan_kernel_dynamic = _dyn_mod.make_mamba2_ssd_scan_kernel_dynamic

# ── Constants ────────────────────────────────────────────────────────────────
TILE = 32
NUM_HEADS = 64
D = 64
N = 128
G = 8
C = 64


# ── PyTorch reference (same as test_mamba2_ssd_scan_ttlang.py) ───────────────


def _segment_sum_torch(x: torch.Tensor) -> torch.Tensor:
    cs = x.size(-1)
    t = x[..., None].expand(*x.size(), cs)
    mask_lower = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool), diagonal=-1)
    t = t.masked_fill(~mask_lower, 0)
    seg = torch.cumsum(t, dim=-2)
    mask_diag = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool), diagonal=0)
    return seg.masked_fill(~mask_diag, float("-inf"))


def _ssd_scan_ref(x_dt, B_in, C_in, x_raw, log_decay, D_skip, chunk_size=C):
    B_b, S, H, D_dim = x_dt.shape
    G_dim = B_in.shape[2]
    reps = H // G_dim
    B_f = B_in.repeat_interleave(reps, dim=2)
    C_f = C_in.repeat_interleave(reps, dim=2)
    pad_size = (chunk_size - S % chunk_size) % chunk_size
    x_pad = F.pad(x_raw, (0, 0, 0, 0, 0, pad_size))
    D_residual = D_skip[None, None, :, None] * x_pad

    def _chunk4(t):
        t = F.pad(t, (0, 0, 0, 0, 0, pad_size))
        return t.reshape(B_b, -1, chunk_size, t.shape[2], t.shape[3])

    def _chunk3(t):
        t = F.pad(t, (0, 0, 0, pad_size))
        return t.reshape(B_b, -1, chunk_size, t.shape[2])

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
    y = (Y_diag + Y_off).reshape(B_b, -1, H, D_dim) + D_residual
    return y, h_next


# ── SimTensor builder for the dynamic kernel ─────────────────────────────────


def _build_sim_inputs_dynamic(x_dt_raw, B_in_raw, C_in_raw, x_raw, log_decay_raw, D_skip_raw, n_chunks):
    """Build SimTensors for the dynamic kernel (includes n_chunks_t as first arg).

    B and C_mat are in GROUP format [G*n_chunks*C, N] — the kernel indexes them
    via g_idx = h_idx // n_heads_per_group, so only G=8 unique rows are needed.
    All other tensors are per-HEAD format [H*..., ...].

    Returns dict with keys:
        n_chunks_t  [TILE, TILE] bf16 — element [0,0] = float(n_chunks)
        + data tensors matching the kernel signature
    """
    S = n_chunks * C
    i_idx_t = torch.arange(C).unsqueeze(1).float()
    s_idx_t = torch.arange(C).unsqueeze(0).float()
    causal = s_idx_t <= i_idx_t

    # Per-head tensors (log_L, x_dt, x, log_gamma, log_delta, log_gscalar, h_in, D_skip_t)
    all_log_L, all_xdt, all_x = [], [], []
    all_lg, all_ld, all_lgs, all_h_in, all_dskip = [], [], [], [], []

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
        log_delta_h = torch.zeros(n_chunks * C, TILE, dtype=torch.bfloat16)
        for ci in range(n_chunks):
            A_last = A_cumsum[ci, -1]
            for row in range(C):
                log_gamma_h[ci * C + row, :] = A_cumsum[ci, row]
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

    # B and C_mat in GROUP format [G*n_chunks*C, N]
    # Kernel reads: g_idx = h_idx // n_heads_per_group; B rows at [g_idx*n*C : ...]
    all_B_g, all_C_mat_g = [], []
    for g_idx in range(G):
        B_g = B_in_raw[:S, g_idx, :].reshape(n_chunks, C, N).to(torch.bfloat16)
        C_g = C_in_raw[:S, g_idx, :].reshape(n_chunks, C, N).to(torch.bfloat16)
        all_B_g.append(B_g.reshape(n_chunks * C, N))
        all_C_mat_g.append(C_g.reshape(n_chunks * C, N))

    def _st(t):
        return SimTensor(t.clone(), TILE_LAYOUT)

    nc_tile = torch.full((TILE, TILE), float(n_chunks), dtype=torch.bfloat16)

    return {
        "n_chunks_t": _st(nc_tile),
        "log_L": _st(torch.cat(all_log_L, dim=0)),
        "x_dt": _st(torch.cat(all_xdt, dim=0)),
        "B": _st(torch.cat(all_B_g, dim=0)),
        "C_mat": _st(torch.cat(all_C_mat_g, dim=0)),
        "x": _st(torch.cat(all_x, dim=0)),
        "log_gamma": _st(torch.cat(all_lg, dim=0)),
        "log_delta": _st(torch.cat(all_ld, dim=0)),
        "log_gscalar": _st(torch.cat(all_lgs, dim=0)),
        "h_in": _st(torch.cat(all_h_in, dim=0)),
        "D_skip_t": _st(torch.cat(all_dskip, dim=0)),
        "y_out": _st(torch.zeros(NUM_HEADS * n_chunks * C, D, dtype=torch.bfloat16)),
        "h_out": _st(torch.zeros(NUM_HEADS * D, N, dtype=torch.bfloat16)),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_single_cache_entry_for_multiple_isls():
    """One call to make_mamba2_ssd_scan_kernel_dynamic() serves all ISLs."""
    _KERNEL_CACHE_DYN.clear()

    k2 = make_mamba2_ssd_scan_kernel_dynamic()
    k4 = make_mamba2_ssd_scan_kernel_dynamic()

    assert k2 is k4, "Expected cache hit: same kernel object for both calls"
    assert len(_KERNEL_CACHE_DYN) == 1, f"Expected 1 cache entry, got {len(_KERNEL_CACHE_DYN)}"


@pytest.mark.parametrize("n_chunks", [2, 4])
def test_dynamic_kernel_pcc(n_chunks):
    """Dynamic kernel produces correct output for n_chunks={2,4} with same kernel object."""
    S = n_chunks * C
    torch.manual_seed(42)

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

    kernel = make_mamba2_ssd_scan_kernel_dynamic()

    t = _build_sim_inputs_dynamic(
        x_dt_raw[0].float(),
        B_in_raw[0].float(),
        C_in_raw[0].float(),
        x_raw_in[0].float(),
        logd_raw[0].float(),
        D_skip_raw.float(),
        n_chunks,
    )

    kernel(
        t["n_chunks_t"],
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

    y_out_data = t["y_out"]._tensor[: NUM_HEADS * n_chunks * C, :D]
    h_out_data = t["h_out"]._tensor[: NUM_HEADS * D, :N]

    # Reshape: y_out is stacked [H, n_chunks*C, D] → [S, H, D] via head interleave
    # The kernel stacks heads along tile-row dim: head 0 rows 0..n_chunks*2-1, etc.
    y_ttlang = torch.zeros(S, NUM_HEADS, D, dtype=torch.bfloat16)
    h_ttlang = torch.zeros(NUM_HEADS, D, N, dtype=torch.bfloat16)
    for h_idx in range(NUM_HEADS):
        row_start = h_idx * n_chunks * C
        row_end = row_start + n_chunks * C
        y_ttlang[:, h_idx, :] = y_out_data[row_start:row_end, :D]
        h_ttlang[h_idx, :, :] = h_out_data[h_idx * D : (h_idx + 1) * D, :N]

    y_ref_s = y_ref[0, :S].float()
    h_ref_s = h_ref[0].float()

    y_diff = (y_ttlang.float() - y_ref_s).abs().max().item()
    h_diff = (h_ttlang.float() - h_ref_s).abs().max().item()
    print(f"\nn_chunks={n_chunks}: y max_diff={y_diff:.4f}  h max_diff={h_diff:.4f}")

    assert y_diff < 1e-2, f"n_chunks={n_chunks}: y max_diff={y_diff:.4f} exceeds 1e-2"
    assert h_diff < 1e-2, f"n_chunks={n_chunks}: h max_diff={h_diff:.4f} exceeds 1e-2"


def test_dynamic_kernel_same_object_for_both_isls():
    """SAME kernel object handles ISL=128 then ISL=256 sequentially (no re-compilation)."""
    _KERNEL_CACHE_DYN.clear()
    kernel = make_mamba2_ssd_scan_kernel_dynamic()
    torch.manual_seed(7)

    for n_chunks in [2, 4]:
        S = n_chunks * C
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

        t = _build_sim_inputs_dynamic(
            x_dt_raw[0].float(),
            B_in_raw[0].float(),
            C_in_raw[0].float(),
            x_raw_in[0].float(),
            logd_raw[0].float(),
            D_skip_raw.float(),
            n_chunks,
        )

        kernel(
            t["n_chunks_t"],
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

        y_out_data = t["y_out"]._tensor[: NUM_HEADS * n_chunks * C, :D]
        y_ttlang = torch.zeros(S, NUM_HEADS, D, dtype=torch.bfloat16)
        for h_idx in range(NUM_HEADS):
            r0 = h_idx * n_chunks * C
            y_ttlang[:, h_idx, :] = y_out_data[r0 : r0 + n_chunks * C, :D]

        y_diff = (y_ttlang.float() - y_ref[0, :S].float()).abs().max().item()
        assert y_diff < 1e-2, f"n_chunks={n_chunks}: y max_diff={y_diff:.4f}"
        print(f"  n_chunks={n_chunks} (ISL={S}): y max_diff={y_diff:.4f} ✓", flush=True)

    assert len(_KERNEL_CACHE_DYN) == 1, "Cache must stay at 1 entry across both ISLs"
    print(f"  _KERNEL_CACHE_DYN entries: {len(_KERNEL_CACHE_DYN)} (expected 1) ✓")
