# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused Mamba2 SSD chunked-scan kernel.

PCC benchmark: kernel output vs pure-PyTorch reference.
Perf benchmark: fused kernel vs per-chunk TTNN reference.

Run PCC test:
    python_env/bin/python -m pytest \
        models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_mamba2_ssd_scan.py \
        -k "pcc" -v -s --timeout=3600

Run perf test:
    python_env/bin/python -m pytest \
        models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_mamba2_ssd_scan.py \
        -k "perf" -v -s --timeout=3600
"""

import time

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_op import _CHUNK_SIZE as CHUNK_SIZE
from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_op import _HEAD_DIM as HEAD_DIM
from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_op import _N_GROUPS as N_GROUPS
from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_op import _NUM_HEADS as NUM_HEADS
from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_op import (
    _SSM_STATE_SIZE as SSM_STATE_SIZE,
)
from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_op import mamba2_ssd_scan
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

DEVICE_MESH = None


@pytest.fixture(scope="module")
def mesh_device():
    global DEVICE_MESH
    if DEVICE_MESH is None:
        DEVICE_MESH = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4),
            l1_small_size=32768,
            trace_region_size=50_000_000,
        )
    yield DEVICE_MESH


# ---------------------------------------------------------------------------
# Pure PyTorch SSD scan reference (h_prev = 0)
# ---------------------------------------------------------------------------


def _segment_sum_torch(x: torch.Tensor) -> torch.Tensor:
    """Causal segment sums: mirrors reference/functional.py::_segment_sum."""
    cs = x.size(-1)
    t = x[..., None].expand(*x.size(), cs)
    mask_lower = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool), diagonal=-1)
    t = t.masked_fill(~mask_lower, 0)
    seg = torch.cumsum(t, dim=-2)
    mask_diag = torch.tril(torch.ones(cs, cs, device=x.device, dtype=torch.bool), diagonal=0)
    return seg.masked_fill(~mask_diag, float("-inf"))


def _ssd_scan_ref(
    x_dt: torch.Tensor,  # [B, S, H, D]  x * dt (already discretized)
    B_in: torch.Tensor,  # [B, S, G, N]
    C_in: torch.Tensor,  # [B, S, G, N]
    x_raw: torch.Tensor,  # [B, S, H, D]  raw x (for D skip)
    log_decay: torch.Tensor,  # [B, S, H]     A * dt (negative)
    D_skip: torch.Tensor,  # [H]
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked SSD scan in float32, h_prev=0.

    Mirrors mamba2_layer() in reference/functional.py (SSD scan portion only).
    Returns (y [B, S_pad, H, D], h_next [B, H, D, N]).
    """
    B, S, H, D = x_dt.shape
    G, N = B_in.shape[2], B_in.shape[3]
    reps = H // G

    # Expand B, C from G groups to H heads
    B_f = B_in.repeat_interleave(reps, dim=2)  # [B, S, H, N]
    C_f = C_in.repeat_interleave(reps, dim=2)  # [B, S, H, N]

    pad_size = (chunk_size - S % chunk_size) % chunk_size

    # D skip connection (applied to padded x)
    x_pad = F.pad(x_raw, (0, 0, 0, 0, 0, pad_size))  # [B, S_pad, H, D]
    D_residual = D_skip[None, None, :, None] * x_pad  # [B, S_pad, H, D]

    def _chunk_rank4(t: torch.Tensor) -> torch.Tensor:
        t = F.pad(t, (0, 0, 0, 0, 0, pad_size))  # [B, S_pad, H, K]
        return t.reshape(B, -1, chunk_size, t.shape[2], t.shape[3])  # [B, nc, cs, H, K]

    def _chunk_rank3(t: torch.Tensor) -> torch.Tensor:
        t = F.pad(t, (0, 0, 0, pad_size))  # [B, S_pad, H]
        return t.reshape(B, -1, chunk_size, t.shape[2])  # [B, nc, cs, H]

    x_c = _chunk_rank4(x_dt)  # [B, nc, cs, H, D]
    B_c = _chunk_rank4(B_f)  # [B, nc, cs, H, N]
    C_c = _chunk_rank4(C_f)  # [B, nc, cs, H, N]
    A_c = _chunk_rank3(log_decay)  # [B, nc, cs, H]
    nc = x_c.shape[1]

    A_c_h = A_c.permute(0, 3, 1, 2)  # [B, H, nc, cs]
    A_cumsum = torch.cumsum(A_c_h, dim=-1)  # [B, H, nc, cs]

    # 1. Intra-chunk diagonal blocks
    L = torch.exp(_segment_sum_torch(A_c_h))  # [B, H, nc, cs, cs]

    G_mat = (C_c[:, :, :, None, :, :] * B_c[:, :, None, :, :, :]).sum(dim=-1)  # [B, nc, cs, cs, H]
    M = G_mat * L.permute(0, 2, 3, 4, 1)  # [B, nc, cs, cs, H]
    Y_diag = (M[..., None] * x_c[:, :, None]).sum(dim=3)  # [B, nc, cs, H, D]

    # 2. States for inter-chunk recurrence
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [B, H, nc, cs]
    B_decay = B_c * decay_states.permute(0, 2, 3, 1)[..., None]  # [B, nc, cs, H, N]
    states = (B_decay[..., None, :] * x_c[..., None]).sum(dim=2)  # [B, nc, H, D, N]

    # 3. Inter-chunk SSM recurrence (h_prev = 0)
    previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)  # [B, nc+1, H, D, N]
    A_cumsum_last = A_cumsum[:, :, :, -1]  # [B, H, nc]
    decay_chunk = torch.exp(_segment_sum_torch(F.pad(A_cumsum_last, (1, 0))))  # [B, H, nc+1, nc+1]
    decay_chunk = decay_chunk.transpose(1, 3)  # [B, nc+1, nc+1, H]
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)  # [B, nc+1, H, D, N]
    states, h_next = new_states[:, :-1], new_states[:, -1]  # [B, nc, H, D, N], [B, H, D, N]

    # 4. State -> output per chunk (off-diagonal blocks)
    state_decay_out = torch.exp(A_cumsum)  # [B, H, nc, cs]
    Y_off = (C_c[..., None, :] * states[:, :, None, ...]).sum(-1)  # [B, nc, cs, H, D]
    Y_off = Y_off * state_decay_out.permute(0, 2, 3, 1)[..., None]  # [B, nc, cs, H, D]

    y = (Y_diag + Y_off).reshape(B, -1, H, D) + D_residual  # [B, S_pad, H, D]

    return y, h_next


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------


def _make_inputs(n_chunks, device, mesh_mapper):
    """Return (raw_torch_dict, ttnn_dict) for n_chunks chunks."""
    B = 1
    S = n_chunks * CHUNK_SIZE
    H, D, N, G = NUM_HEADS, HEAD_DIM, SSM_STATE_SIZE, N_GROUPS

    torch.manual_seed(42)
    raw = {
        "x_dt": torch.randn(B, S, H, D) * 0.01,
        "B_in": torch.randn(B, S, G, N) * 0.1,
        "C_in": torch.randn(B, S, G, N) * 0.1,
        "x": torch.randn(B, S, H, D),
        "logd": torch.randn(B, S, H) * 0.01 - 0.5,  # negative log-decay
        "h_prev": torch.zeros(B, H, D, N),
        "D": torch.ones(1, H, 1, 1),
    }

    def _tt(t):
        return ttnn.from_torch(
            t.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt = {k: _tt(v) for k, v in raw.items()}
    return raw, tt


# ---------------------------------------------------------------------------
# PCC test — ISLs 128, 4K, 8K, 16K  (n_chunks = 2, 64, 128, 256)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_chunks", [2, 64, 128, 256])
def test_mamba2_ssd_scan_pcc(n_chunks, mesh_device):
    """Fused kernel output must match PyTorch reference (PCC ≥ 0.99)."""
    from ttnn import ReplicateTensorToMesh as _R

    mesh_mapper = _R(mesh_device)
    raw, tt = _make_inputs(n_chunks, mesh_device, mesh_mapper)

    # ---- PyTorch reference (float32, h_prev=0) ----
    D_skip = raw["D"].squeeze()  # [H]
    y_ref, h_ref = _ssd_scan_ref(
        x_dt=raw["x_dt"].float(),
        B_in=raw["B_in"].float(),
        C_in=raw["C_in"].float(),
        x_raw=raw["x"].float(),
        log_decay=raw["logd"].float(),
        D_skip=D_skip.float(),
    )
    # y_ref: [B, S_pad, H, D], h_ref: [B, H, D, N]

    # ---- Fused kernel ----
    y_fused, h_fused = mamba2_ssd_scan(
        mesh_device=mesh_device,
        x_dt_pad=tt["x_dt"],
        B_pad=tt["B_in"],
        C_pad=tt["C_in"],
        x_pad=tt["x"],
        log_decay_pad=tt["logd"],
        h_prev=tt["h_prev"],
        D_tt=tt["D"],
        n_chunks=n_chunks,
        mesh_mapper=mesh_mapper,
    )

    # ---- Compare ----
    y_fused_t = ttnn.to_torch(y_fused, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0].float()
    h_fused_t = ttnn.to_torch(h_fused, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0].float()

    y_ref_t = y_ref[0].float()  # remove batch dim
    h_ref_t = h_ref[0].float()

    y_pass, y_pcc = comp_pcc(y_ref_t, y_fused_t, pcc=0.99)
    h_pass, h_pcc = comp_pcc(h_ref_t, h_fused_t, pcc=0.99)

    isl = n_chunks * CHUNK_SIZE
    print(f"\nISL={isl} (n_chunks={n_chunks}): y_pcc={y_pcc:.4f}  h_pcc={h_pcc:.4f}")
    assert y_pass, f"y PCC {y_pcc:.4f} < 0.99  (ISL={isl})"
    assert h_pass, f"h PCC {h_pcc:.4f} < 0.99  (ISL={isl})"


# ---------------------------------------------------------------------------
# Perf test — ISLs ≈ 65K and 256K  (n_chunks = 1024, 4096)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_chunks", [1024, 4096])
def test_mamba2_ssd_scan_perf(n_chunks, mesh_device):
    """Perf: fused kernel vs Python chunk loop at large n_chunks (ISL ≈ 65K, 256K)."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_prefill import (
        _expand_groups,
        _mamba2_ssd_chunk,
    )
    from ttnn import ReplicateTensorToMesh as _R

    mesh_mapper = _R(mesh_device)
    _, tt = _make_inputs(n_chunks, mesh_device, mesh_mapper)

    x_dt = tt["x_dt"]
    B_in = tt["B_in"]
    C_in = tt["C_in"]
    x_in = tt["x"]
    logd = tt["logd"]
    h_prev = tt["h_prev"]
    D_tt = tt["D"]
    B = 1

    # ---- Reference timing (Python chunk loop, sampled) ----
    n_warm = min(4, n_chunks)
    n_ref_sample = min(16, n_chunks)
    for _ in range(n_warm):
        h_r = h_prev
        for c in range(n_ref_sample):
            t0, t1 = c * CHUNK_SIZE, (c + 1) * CHUNK_SIZE
            logd_c = ttnn.slice(logd, [0, t0, 0], [B, t1, NUM_HEADS], memory_config=ttnn.L1_MEMORY_CONFIG)
            xdt_c = ttnn.slice(x_dt, [0, t0, 0, 0], [B, t1, NUM_HEADS, HEAD_DIM])
            Bg_c = ttnn.slice(
                B_in, [0, t0, 0, 0], [B, t1, N_GROUPS, SSM_STATE_SIZE], memory_config=ttnn.L1_MEMORY_CONFIG
            )
            Bc = _expand_groups(Bg_c)
            Cg_c = ttnn.slice(
                C_in, [0, t0, 0, 0], [B, t1, N_GROUPS, SSM_STATE_SIZE], memory_config=ttnn.L1_MEMORY_CONFIG
            )
            Cc = _expand_groups(Cg_c)
            xc = ttnn.slice(x_in, [0, t0, 0, 0], [B, t1, NUM_HEADS, HEAD_DIM])
            _, h_r = _mamba2_ssd_chunk(logd_c, xdt_c, Bc, Cc, D_tt, xc, h_r, mesh_device)
        ttnn.synchronize_device(mesh_device)

    t_start = time.perf_counter()
    h_r = h_prev
    for c in range(n_ref_sample):
        t0c, t1c = c * CHUNK_SIZE, (c + 1) * CHUNK_SIZE
        logd_c = ttnn.slice(logd, [0, t0c, 0], [B, t1c, NUM_HEADS], memory_config=ttnn.L1_MEMORY_CONFIG)
        xdt_c = ttnn.slice(x_dt, [0, t0c, 0, 0], [B, t1c, NUM_HEADS, HEAD_DIM])
        Bg_c = ttnn.slice(B_in, [0, t0c, 0, 0], [B, t1c, N_GROUPS, SSM_STATE_SIZE], memory_config=ttnn.L1_MEMORY_CONFIG)
        Bc = _expand_groups(Bg_c)
        Cg_c = ttnn.slice(C_in, [0, t0c, 0, 0], [B, t1c, N_GROUPS, SSM_STATE_SIZE], memory_config=ttnn.L1_MEMORY_CONFIG)
        Cc = _expand_groups(Cg_c)
        xc = ttnn.slice(x_in, [0, t0c, 0, 0], [B, t1c, NUM_HEADS, HEAD_DIM])
        _, h_r = _mamba2_ssd_chunk(logd_c, xdt_c, Bc, Cc, D_tt, xc, h_r, mesh_device)
    ttnn.synchronize_device(mesh_device)
    t_ref_sample = time.perf_counter() - t_start
    t_ref_full_est = t_ref_sample * (n_chunks / n_ref_sample)

    # ---- Fused kernel timing ----
    for _ in range(3):
        y_f, h_f = mamba2_ssd_scan(
            mesh_device=mesh_device,
            x_dt_pad=x_dt,
            B_pad=B_in,
            C_pad=C_in,
            x_pad=x_in,
            log_decay_pad=logd,
            h_prev=h_prev,
            D_tt=D_tt,
            n_chunks=n_chunks,
            mesh_mapper=mesh_mapper,
        )
        ttnn.synchronize_device(mesh_device)

    t_start = time.perf_counter()
    y_f, h_f = mamba2_ssd_scan(
        mesh_device=mesh_device,
        x_dt_pad=x_dt,
        B_pad=B_in,
        C_pad=C_in,
        x_pad=x_in,
        log_decay_pad=logd,
        h_prev=h_prev,
        D_tt=D_tt,
        n_chunks=n_chunks,
        mesh_mapper=mesh_mapper,
    )
    ttnn.synchronize_device(mesh_device)
    t_fused = time.perf_counter() - t_start

    speedup = t_ref_full_est / t_fused if t_fused > 0 else float("inf")
    isl = n_chunks * CHUNK_SIZE
    print(f"\nISL={isl} (n_chunks={n_chunks}):")
    print(f"  ref (extrapolated): {t_ref_full_est*1000:.1f} ms")
    print(f"  fused kernel:       {t_fused*1000:.1f} ms")
    print(f"  speedup:            {speedup:.1f}×")

    if n_chunks >= 1024:
        assert speedup >= 10.0, (
            f"Fused kernel not fast enough: {speedup:.1f}× < 10× target "
            f"(ref={t_ref_full_est*1000:.1f}ms, fused={t_fused*1000:.1f}ms)"
        )
