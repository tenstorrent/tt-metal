#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SSD scan kernel sweep: vanilla TTNN chunk-loop vs tt-lang fused kernel, ISL 64→262144.

Times only the SSD scan itself — no layer norm, matmul, or conv.
Inputs are synthetic and pre-built/uploaded before each timer window so
only device execution is measured.

Flow per ISL:
  1. CPU: build synthetic SSD inputs (log_decay, x_dt, B, C, x, D, h_prev=0)
  2. CPU: pre-build tt-lang kernel inputs (cumsum, log_L, log_gamma, …)
  3. Device: upload all inputs  ← not timed
  ────────────────────────────────────────────
  4. [vanilla timer]  _mamba2_ssd_chunk × n_chunks  (all on device)
  5. [tt-lang timer]  single kernel dispatch
  ────────────────────────────────────────────
  6. Report vanilla_ms, ttlang_ms, speedup

Correctness (PCC vs CPU float32) is validated separately by
test_mamba2_ssd_scan_ttlang_hw.py which covers n_chunks 2→4096.

Usage:
    TT_LANG_PYTHON_PATH=/home/ttuser/ssinghal/tt-lang/python \\
    python_env/bin/python \\
        models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/bench_prefill_isl_sweep.py

Optional flags:
    --isls 64 128 512         # override ISL list
    --warmup-isl 128          # ISL used for kernel warm-up (default 128)
    --no-reset                # skip tt-smi device reset at startup
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# ── path setup ────────────────────────────────────────────────────────────────
os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for _p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_tt_lang_path = os.environ.get("TT_LANG_PYTHON_PATH", "")
if _tt_lang_path and _tt_lang_path not in sys.path:
    sys.path.insert(0, _tt_lang_path)

import torch

import ttnn

# ── model dims (must match mamba2_prefill.py) ─────────────────────────────────
H = 64  # NUM_HEADS
D = 64  # HEAD_DIM
G = 8  # N_GROUPS
N = 128  # SSM_STATE_SIZE
C = 64  # CHUNK_SIZE

ISLS_DEFAULT = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


# ── device tensor upload helpers ──────────────────────────────────────────────


def _to_dev(t: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ── single ISL benchmark ──────────────────────────────────────────────────────


def _bench_ssd_isl(mesh_device, n_chunks: int, *, timed: bool = True, ttlang_only: bool = False) -> dict:
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_prefill import (
        _build_ttlang_ssd_inputs,
        _expand_groups,
        _mamba2_ssd_chunk,
    )
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_ssd_scan_ttlang import (
        make_mamba2_ssd_scan_kernel,
    )

    S = n_chunks * C
    torch.manual_seed(0)

    # ── Synthetic SSD inputs (CPU float32) ────────────────────────────────
    # log_decay representative of real model: A_log~1.8, dt~0.001
    # → per-step log_decay ≈ -exp(1.8)*0.001 ≈ -0.006, well within BF16 range
    logd_cpu = (torch.randn(S, H) * 0.01 - 0.5).float()  # [S, H]
    x_dt_cpu = torch.randn(S, H, D).float() * 0.01  # [S, H, D]
    B_cpu = torch.randn(S, G, N).float() * 0.1  # [S, G, N]
    C_cpu = torch.randn(S, G, N).float() * 0.1  # [S, G, N]
    x_cpu = torch.randn(S, H, D).float()  # [S, H, D]
    D_cpu = torch.ones(H).float()  # [H]

    # ── tt-lang: pre-build kernel inputs on CPU (NOT timed) ───────────────
    cpu_in = _build_ttlang_ssd_inputs(x_dt_cpu, B_cpu, C_cpu, x_cpu, logd_cpu, D_cpu, None, n_chunks)
    # Upload all tt-lang inputs to device (NOT timed)
    dev_in = {k: _to_dev(v, mesh_device) for k, v in cpu_in.items()}

    kernel = make_mamba2_ssd_scan_kernel(n_chunks, num_heads=H, n_groups=G)

    # ── tt-lang: time kernel dispatch only ────────────────────────────────
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    kernel(
        dev_in["log_L"],
        dev_in["x_dt"],
        dev_in["B"],
        dev_in["C_mat"],
        dev_in["x"],
        dev_in["log_gamma"],
        dev_in["log_delta"],
        dev_in["log_gscalar"],
        dev_in["h_in"],
        dev_in["D_skip_t"],
        dev_in["y_out"],
        dev_in["h_out"],
    )
    ttnn.synchronize_device(mesh_device)
    ttlang_ms = (time.perf_counter() - t0) * 1e3

    # Free tt-lang device tensors
    for t in dev_in.values():
        t.deallocate(True)

    vanilla_ms = float("nan")
    if not ttlang_only:
        # ── vanilla: upload chunk-loop inputs to device (NOT timed) ──────────
        log_decay_tt = _to_dev(logd_cpu.bfloat16().unsqueeze(0), mesh_device)  # [1, S, H]
        x_dt_tt = _to_dev(x_dt_cpu.bfloat16().unsqueeze(0), mesh_device)  # [1, S, H, D]
        B_tt = _to_dev(B_cpu.bfloat16().unsqueeze(0), mesh_device)  # [1, S, G, N]
        C_tt = _to_dev(C_cpu.bfloat16().unsqueeze(0), mesh_device)  # [1, S, G, N]
        x_tt = _to_dev(x_cpu.bfloat16().unsqueeze(0), mesh_device)  # [1, S, H, D]
        D_tt = _to_dev(D_cpu.bfloat16().reshape(1, 1, H, 1), mesh_device)  # [1, 1, H, 1]

        # ── vanilla: time chunk loop (all device ops) ─────────────────────
        h_prev = None
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for ci in range(n_chunks):
            t0_c, t1_c = ci * C, (ci + 1) * C
            log_decay_c = ttnn.slice(log_decay_tt, [0, t0_c, 0], [1, t1_c, H], memory_config=ttnn.L1_MEMORY_CONFIG)
            x_dt_c = ttnn.slice(x_dt_tt, [0, t0_c, 0, 0], [1, t1_c, H, D])
            B_g_c = ttnn.slice(B_tt, [0, t0_c, 0, 0], [1, t1_c, G, N], memory_config=ttnn.L1_MEMORY_CONFIG)
            B_c = _expand_groups(B_g_c)
            C_g_c = ttnn.slice(C_tt, [0, t0_c, 0, 0], [1, t1_c, G, N], memory_config=ttnn.L1_MEMORY_CONFIG)
            C_c = _expand_groups(C_g_c)
            x_c = ttnn.slice(x_tt, [0, t0_c, 0, 0], [1, t1_c, H, D])
            _, h_prev = _mamba2_ssd_chunk(log_decay_c, x_dt_c, B_c, C_c, D_tt, x_c, h_prev, mesh_device)
        ttnn.synchronize_device(mesh_device)
        vanilla_ms = (time.perf_counter() - t0) * 1e3

        for t in [log_decay_tt, x_dt_tt, B_tt, C_tt, x_tt, D_tt]:
            t.deallocate(True)

    speedup = vanilla_ms / ttlang_ms if (timed and not ttlang_only) else float("nan")
    return {
        "S": S,
        "n_chunks": n_chunks,
        "vanilla_ms": vanilla_ms if timed else float("nan"),
        "ttlang_ms": ttlang_ms if timed else float("nan"),
        "v_tok_s": S / vanilla_ms * 1e3 if (timed and not ttlang_only) else float("nan"),
        "tl_tok_s": S / ttlang_ms * 1e3 if timed else float("nan"),
        "speedup": speedup,
    }


# ── V_bad permanent guard ─────────────────────────────────────────────────────

_V_BAD = 0x90A8CB80
_GUARD_MB = 16
_MAX_PROBES = 300


def _place_vbad_guard(mesh_device) -> "ttnn.Tensor":
    probe_bytes = _GUARD_MB * 1024 * 1024
    probe_elems = probe_bytes // 2
    _cpu = torch.zeros(1, probe_elems, dtype=torch.bfloat16)
    _mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    def _alloc():
        return ttnn.from_torch(
            _cpu,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _dev2_addr(t) -> int:
        try:
            devs = ttnn.get_device_tensors(t)
            return (devs[2] if len(devs) > 2 else devs[0]).buffer_address()
        except Exception:
            return 0

    probes = [_alloc()]
    for i in range(_MAX_PROBES):
        addr = _dev2_addr(probes[-1])
        if addr + probe_bytes > _V_BAD:
            guard = probes[-1]
            for p in probes[:-1]:
                p.deallocate(True)
            print(
                f"    probe {i}: addr=0x{addr:08x} covers "
                f"[0x{addr:08x}, 0x{addr+probe_bytes:08x}) — V_bad 0x{_V_BAD:08x} blocked",
                flush=True,
            )
            return guard
        probes.append(_alloc())

    print(f"    WARNING: V_bad not reached after {_MAX_PROBES} probes", flush=True)
    for p in probes[:-1]:
        p.deallocate(True)
    return probes[-1]


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isls", type=int, nargs="+", default=ISLS_DEFAULT)
    ap.add_argument("--warmup-isl", type=int, default=128)
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--ttlang-only", action="store_true", help="Skip vanilla chunk-loop timing; report tt-lang only")
    args = ap.parse_args()

    if not args.no_reset:
        print("Resetting device (tt-smi -r all) ...", flush=True)
        ret = subprocess.run(["tt-smi", "-r", "all"], capture_output=True, text=True)
        if ret.returncode != 0:
            print(f"  WARNING: tt-smi returned {ret.returncode}: {ret.stderr.strip()}")
        else:
            print("  Device reset OK.", flush=True)
        time.sleep(2)

    print("Opening TP=4 device ...", flush=True)
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    mesh_device = open_device_tp4()

    print("Placing permanent V_bad DRAM guard on device-2 ...", flush=True)
    _vbad_guard = _place_vbad_guard(mesh_device)
    print("  V_bad guard placed.\n", flush=True)

    # Pre-warm ALL ISLs (untimed) so JIT compilation does not pollute measurements.
    # Each unique n_chunks compiles a new ELF on first call; warmup amortizes that.
    print(f"Pre-warming all {len(args.isls)} ISLs (JIT compilation + kernel cache) ...", flush=True)
    for _S in args.isls:
        _nc = _S // C
        print(f"  warm-up ISL={_S} (n_chunks={_nc}) ...", flush=True)
        _bench_ssd_isl(mesh_device, _nc, timed=False, ttlang_only=True)
    print("  Pre-warm done.\n", flush=True)

    if args.ttlang_only:
        hdr = f"{'ISL':>7}  {'n_chunks':>8}  " f"{'ttlang_ms':>11}  {'tl_tok/s':>9}"
    else:
        hdr = (
            f"{'ISL':>7}  {'n_chunks':>8}  "
            f"{'vanilla_ms':>11}  {'v_tok/s':>9}  "
            f"{'ttlang_ms':>11}  {'tl_tok/s':>9}  "
            f"{'speedup':>8}"
        )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    errors = []
    results = []
    for S in args.isls:
        assert S % C == 0, f"ISL {S} not divisible by CHUNK_SIZE={C}"
        n_chunks = S // C
        try:
            r = _bench_ssd_isl(mesh_device, n_chunks, ttlang_only=args.ttlang_only)
        except Exception as exc:
            print(f"  ISL={S}: ERROR — {exc}", flush=True)
            errors.append((S, str(exc)))
            continue

        if args.ttlang_only:
            print(
                f"{r['S']:>7}  {r['n_chunks']:>8}  " f"{r['ttlang_ms']:>11.1f}  {r['tl_tok_s']:>9.0f}",
                flush=True,
            )
        else:
            print(
                f"{r['S']:>7}  {r['n_chunks']:>8}  "
                f"{r['vanilla_ms']:>11.1f}  {r['v_tok_s']:>9.0f}  "
                f"{r['ttlang_ms']:>11.1f}  {r['tl_tok_s']:>9.0f}  "
                f"{r['speedup']:>7.2f}x",
                flush=True,
            )
        results.append(r)

    print(sep)
    if errors:
        print(f"\nERRORS at {len(errors)} ISL(s):")
        for S, msg in errors:
            print(f"  ISL={S}: {msg}")
    else:
        print(f"\nAll {len(results)} ISLs completed.")

    close_device_tp4(mesh_device)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
