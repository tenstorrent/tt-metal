# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test for _mamba2_ssd_all_chunks_ttlang — verifies on-device preprocessing path.

This test verifies that the rewritten _mamba2_ssd_all_chunks_ttlang (which
permutes/reshapes activations on device instead of downloading them to CPU)
produces the same output as the PyTorch reference.

Run:
    cd /home/ttuser/ssinghal/tt-metal && \
    NEMOTRON_USE_TTLANG_SSD=1 \
    TT_LANG_PYTHON_PATH=/home/ttuser/ssinghal/tt-lang/python \
    python_env/bin/python -m pytest \
        models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_mamba2_ssd_all_chunks_ttlang_pcc.py \
        -v -s
"""
import os
import sys

os.environ["NEMOTRON_USE_TTLANG_SSD"] = "1"

_root = os.environ.get("TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
for _p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_tt_lang_path = os.environ.get("TT_LANG_PYTHON_PATH", "/home/ttuser/ssinghal/tt-lang/build/python_packages")
if _tt_lang_path and _tt_lang_path not in sys.path:
    sys.path.insert(0, _tt_lang_path)

import pytest
import torch
import torch.nn.functional as F

import ttnn

NUM_HEADS = 64
HEAD_DIM = 64
N_GROUPS = 8
SSM_STATE_SIZE = 128
CHUNK_SIZE = 64

PCC_THRESHOLD = 0.99


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a -= a.mean()
    b -= b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-8)
    return float((a * b).sum() / denom)


def _ssd_scan_ref(x_dt, B_in, C_in, x_raw, log_decay, D_skip, chunk_size=CHUNK_SIZE):
    """Iterative PyTorch reference for Mamba2 SSD chunked scan."""
    B, S, H, D_dim = x_dt.shape
    G_dim, N_dim = B_in.shape[2], B_in.shape[3]
    reps = H // G_dim
    B_f = B_in.repeat_interleave(reps, dim=2)  # [B, S, H, N]
    C_f = C_in.repeat_interleave(reps, dim=2)  # [B, S, H, N]

    pad_size = (chunk_size - S % chunk_size) % chunk_size
    n_chunks_total = (S + pad_size) // chunk_size

    xdt_p = F.pad(x_dt, (0, 0, 0, 0, 0, pad_size))
    B_p = F.pad(B_f, (0, 0, 0, 0, 0, pad_size))
    C_p = F.pad(C_f, (0, 0, 0, 0, 0, pad_size))
    x_p = F.pad(x_raw, (0, 0, 0, 0, 0, pad_size))
    ld_p = F.pad(log_decay, (0, 0, 0, pad_size))

    causal = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool))
    y_out = torch.zeros(B, n_chunks_total * chunk_size, H, D_dim, dtype=x_dt.dtype)
    h = torch.zeros(B, H, D_dim, N_dim, dtype=x_dt.dtype)

    for ci in range(n_chunks_total):
        s, e = ci * chunk_size, (ci + 1) * chunk_size
        xdt_c = xdt_p[:, s:e]
        B_c = B_p[:, s:e]
        C_c = C_p[:, s:e]
        x_c = x_p[:, s:e]
        ld_c = ld_p[:, s:e]

        A_cum = torch.cumsum(ld_c, dim=1)  # [B, C, H]
        A_last = A_cum[:, -1, :]  # [B, H]

        diff = A_cum[:, :, None, :] - A_cum[:, None, :, :]  # [B, C, C, H]
        diff.masked_fill_(~causal[None, :, :, None], float("-inf"))
        L = torch.exp(diff)  # [B, C, C, H]
        CB = (C_c[:, :, None] * B_c[:, None]).sum(-1)  # [B, C, C, H]
        LCB = (L * CB).permute(0, 3, 1, 2)  # [B, H, C, C]
        y_intra = torch.einsum("bhis,bshd->bihd", LCB, xdt_c)  # [B, C, H, D]

        gamma = torch.exp(A_cum)  # [B, C, H]
        y_cross = torch.einsum("bchn,bhdn->bchd", C_c, h) * gamma[:, :, :, None]
        y_out[:, s:e] = y_intra + y_cross + D_skip[None, None, :, None] * x_c

        delta = torch.exp(A_last[:, None, :] - A_cum)  # [B, C, H]
        xdt_sc = xdt_c * delta[:, :, :, None]
        dh = torch.einsum("bchd,bchn->bhdn", xdt_sc, B_c)
        h = torch.exp(A_last)[:, :, None, None] * h + dh

    return y_out, h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_device():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    dev = open_device_tp4()
    yield dev
    close_device_tp4(dev)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_chunks", [2, 4, 128])
def test_mamba2_ssd_all_chunks_ttlang_on_device_pcc(mesh_device, n_chunks):
    """_mamba2_ssd_all_chunks_ttlang (on-device permute path) PCC vs PyTorch reference."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_prefill import _mamba2_ssd_all_chunks_ttlang
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import clear_device_weight_cache

    clear_device_weight_cache()

    H, D, N, G, C = NUM_HEADS, HEAD_DIM, SSM_STATE_SIZE, N_GROUPS, CHUNK_SIZE
    S = n_chunks * C
    B = 1
    torch.manual_seed(42)

    # Float32 inputs (match model scale)
    x_dt_raw = torch.randn(B, S, H, D) * 0.01
    B_in_raw = torch.randn(B, S, G, N) * 0.1
    C_in_raw = torch.randn(B, S, G, N) * 0.1
    x_raw = torch.randn(B, S, H, D)
    log_decay_r = torch.randn(B, S, H) * 0.01 - 0.5  # negative log-decay
    D_skip_raw = torch.ones(H)

    # PyTorch reference (float32)
    y_ref, h_ref = _ssd_scan_ref(
        x_dt_raw.float(),
        B_in_raw.float(),
        C_in_raw.float(),
        x_raw.float(),
        log_decay_r.float(),
        D_skip_raw.float(),
    )
    y_ref = y_ref[:, :S].float()  # [1, S, H, D]
    h_ref = h_ref.float()  # [1, H, D, N]

    def _to_dev(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x_dt_dev = _to_dev(x_dt_raw)  # [1, S, H, D]
    B_dev = _to_dev(B_in_raw)  # [1, S, G, N]
    C_dev = _to_dev(C_in_raw)  # [1, S, G, N]
    x_dev = _to_dev(x_raw)  # [1, S, H, D]
    logd_dev = _to_dev(log_decay_r)  # [1, S, H]
    D_tt = _to_dev(D_skip_raw.view(1, H, 1, 1))  # [1, H, 1, 1]

    y_mesh, h_mesh = _mamba2_ssd_all_chunks_ttlang(
        mesh_device,
        x_dt_dev,
        B_dev,
        C_dev,
        x_dev,
        logd_dev,
        h_prev=None,
        D_tt=D_tt,
        n_chunks=n_chunks,
    )

    # Extract first device replica
    y_tt = ttnn.to_torch(ttnn.get_device_tensors(y_mesh)[0]).float()  # [1, S, H, D]
    h_tt = ttnn.to_torch(ttnn.get_device_tensors(h_mesh)[0]).float()  # [1, H, D, N]

    y_pcc = pcc(y_tt[:, :S], y_ref)
    h_pcc = pcc(h_tt, h_ref)
    y_mdiff = (y_tt[:, :S] - y_ref).abs().max().item()
    h_mdiff = (h_tt - h_ref).abs().max().item()

    print(
        f"\n  n_chunks={n_chunks}: y_pcc={y_pcc:.4f}  h_pcc={h_pcc:.4f}"
        f"  y_max_diff={y_mdiff:.4f}  h_max_diff={h_mdiff:.4f}"
    )

    assert y_pcc >= PCC_THRESHOLD, f"y PCC {y_pcc:.4f} < {PCC_THRESHOLD}"
    assert h_pcc >= PCC_THRESHOLD, f"h PCC {h_pcc:.4f} < {PCC_THRESHOLD}"
