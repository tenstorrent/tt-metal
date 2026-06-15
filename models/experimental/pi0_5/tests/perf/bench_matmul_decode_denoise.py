# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone tracy benchmark: CLAUDE-optimized ttnn.matmul_decode at the pi0.5
DENOISE matmul shapes (M=32, the action-expert decode regime).

This is NOT a pytest under the pi05_chunked_mmdecode conftest (which HARD-gates on
the matmul_decode FORK). It opens device 0 directly and runs against whatever ttnn
is importable -- intended to be the CLONE (tt-metal-pi05-openpi) build that carries
the ported CLAUDE-optimized matmul_decode op.

Per denoise matmul shape (all M=32 = 1 tile row):
  qkv_fused    K=1024 N=2560
  mlp_gate_up  K=1024 N=4096
  o_proj       K=2048 N=1024
  mlp_down     K=4096 N=1024

Each shape is driven through MatmulDecodeLinear (the CLAUDE-optimized scheme-selection
wrapper, vendored at ../../tt/mmdecode/matmul_decode_linear.py — self-contained, no
tt_symbiote dependency). At M=32 the wrapper makes exactly ONE device matmul_decode
call per forward (no M-split). One tracy signpost region per shape.

Device KERNEL time is extracted SOLELY from the tracy ops_perf_results CSV
"DEVICE KERNEL DURATION [ns]" (col 20) over MatmulDecodeDeviceOperation rows -- use
extract_perf with METRIC=KERNEL, or read the CSV directly.

Also prints PCC vs torch for each shape (real comparison, threshold 0.99).

Run (one tracy job at a time, op-support-count=20000):
  TT_SYMBIOTE_SIGNPOST_MODE=1 PI05_TRACY_SIGNPOST=1 \
  TT_METAL_HOME=<clone> PYTHONPATH=<clone> \
  <clone>/python_env/bin/python -m tracy -p -r -v --op-support-count 20000 \
    -m '<this file> '
"""
from __future__ import annotations

import os

import torch
import ttnn

# Import MatmulDecodeLinear directly from its file to bypass the tt_symbiote.models
# package __init__ recipe-registration scan (which crashes on transformers version
# drift unrelated to this benchmark). Set MMD_LINEAR_PATH to override the location.
import importlib.util as _ilu

_MMD_PATH = os.environ.get(
    "MMD_LINEAR_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "tt",
        "mmdecode",
        "matmul_decode_linear.py",
    ),
)
_spec = _ilu.spec_from_file_location("matmul_decode_linear", _MMD_PATH)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MatmulDecodeLinear = _mod.MatmulDecodeLinear

N_ITERS = int(os.environ.get("N_ITERS", "5"))
SEED = 1234
BF16 = ttnn.bfloat16
BF8 = ttnn.bfloat8_b
M = 32

# (label, K, N) -- the four denoise expert matmuls.
SHAPES = [
    ("qkv_fused", 1024, 2560),
    ("mlp_gate_up", 1024, 4096),
    ("o_proj", 2048, 1024),
    ("mlp_down", 4096, 1024),
]


def _signpost(name):
    try:
        from tracy import signpost

        signpost(header=name)
    except Exception:
        pass


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (torch.mean((t1 - m1) * (t2 - m2)) / (s1 * s2)).item()


def _native_linear_baseline(device, label, K, N, a_torch, w_KN):
    """Native ttnn.linear baseline at this shape (default auto-config program selection;
    the explicit-pcfg production picker FATALs with not_on_dispatch_core on this p300c host,
    so the fair on-host native point is the default auto-config). bf8_b weight, bf16 out,
    matching the model's denoise weight dtype. Signposted + PCC-checked the SAME way."""
    a_dev = ttnn.from_torch(a_torch, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device)
    # weight uploaded as [K, N] (ttnn.linear convention)
    w_dev = ttnn.from_torch(w_KN, dtype=BF8, layout=ttnn.TILE_LAYOUT, device=device)

    y = ttnn.linear(a_dev, w_dev, dtype=BF16)
    y_torch = ttnn.to_torch(y).reshape(M, N).float()
    ref = a_torch.reshape(M, K).float() @ w_KN.float()
    pcc = _pcc(ref, y_torch)
    ttnn.deallocate(y)

    yw = ttnn.linear(a_dev, w_dev, dtype=BF16)
    ttnn.deallocate(yw)
    ttnn.synchronize_device(device)
    _signpost(f"MMD_DENOISE_NATIVE_{label}")
    for _ in range(N_ITERS):
        yt = ttnn.linear(a_dev, w_dev, dtype=BF16)
        ttnn.deallocate(yt)
    ttnn.synchronize_device(device)
    ttnn.deallocate(a_dev)
    ttnn.deallocate(w_dev)
    return pcc


def run(device):
    torch.manual_seed(SEED)
    print(f"\n=== matmul_decode DENOISE benchmark  M={M}  N_ITERS={N_ITERS} ===")
    print(f"ttnn from: {os.path.realpath(ttnn.__file__)}")
    pcc_results = {}
    native_pcc = {}
    # ---- native baseline arm (default auto-config ttnn.linear) ----
    for label, K, N in SHAPES:
        a_torch = torch.randn(1, M, K) * 0.1
        w_KN = torch.randn(K, N) * 0.02
        native_pcc[label] = _native_linear_baseline(device, label, K, N, a_torch, w_KN)
        print(f"[native] {label:<12} K={K:<5} N={N:<5}  PCC={native_pcc[label]:.5f}")
    # ---- matmul_decode arm ----
    for label, K, N in SHAPES:
        # A = [M, K] bf16 activation; W = [K, N] (matches MatmulDecodeLinear [K,N] convention).
        a_torch = torch.randn(1, M, K) * 0.1
        w_KN = torch.randn(K, N) * 0.02

        lin = MatmulDecodeLinear(
            device,
            w_KN.clone(),
            weight_dtype=BF8,
            out_dtype=BF16,
            role=f"denoise_{label}",
        )
        a_dev = ttnn.from_torch(a_torch, dtype=BF16, layout=ttnn.TILE_LAYOUT, device=device)

        # ---- PCC vs torch (real comparison) ----
        y = lin(a_dev)
        y_torch = ttnn.to_torch(y).reshape(M, N).float()
        ref = a_torch.reshape(M, K).float() @ w_KN.float()
        pcc = _pcc(ref, y_torch)
        pcc_results[label] = pcc
        print(
            f"[plan] {label:<12} K={K:<5} N={N:<5} "
            f"mode={lin.describe().get('mode')} "
            f"n_chunks={lin.describe().get('n_chunks')} "
            f"k_split_G={lin.describe().get('k_split_G')}  PCC={pcc:.5f}"
        )
        ttnn.deallocate(y)

        # ---- warm-up OUTSIDE signpost ----
        yw = lin(a_dev)
        ttnn.deallocate(yw)
        ttnn.synchronize_device(device)

        # ---- timed region ----
        _signpost(f"MMD_DENOISE_{label}")
        for _ in range(N_ITERS):
            yt = lin(a_dev)
            ttnn.deallocate(yt)
        ttnn.synchronize_device(device)

        ttnn.deallocate(a_dev)

    print("\n=== PCC summary (threshold 0.99) ===")
    ok = True
    for label, _, _ in SHAPES:
        p = pcc_results[label]
        flag = "PASS" if p >= 0.99 else "FAIL"
        if p < 0.99:
            ok = False
        print(f"  {label:<12} PCC={p:.5f}  {flag}")
    print(f"=== overall PCC {'PASS' if ok else 'FAIL'} ===")


def main():
    trace_region = int(os.environ.get("PI05_MMD_TRACE_REGION", 256 * 1024 * 1024))
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=trace_region)
    try:
        run(device)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
