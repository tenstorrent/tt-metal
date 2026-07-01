#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
FFT benchmark: Tenstorrent Wormhole B0 (ttnn.experimental.fft) vs CPU baselines.

═══════════════════════════════════════════════════════════════════════════════
TIMING AND ENERGY METHODOLOGY
═══════════════════════════════════════════════════════════════════════════════
WH timing  — kernel-only execution time.
  Window:  right before ttnn.experimental.fft(...)
         → ttnn.synchronize_device(device)
  Excludes: host→device upload, device→host download, Python overhead.
  Rationale: matches Brown et al. ISC 2025 ("performance numbers for WH are
             execution time only").

CPU timing — wall-clock time for the entire FFT call (plan is cached after
  warmup).  numpy uses pocketfft (multi-core); torch uses its own FFT backend.

Energy     — E = avg_power_W × median_time_s  (same formula as Brown et al.)
  Both --wh-power and --cpu-power default to None (not set).
  • WH energy is reported only when --wh-power is explicitly supplied.
  • CPU energy and the CPU-over-WH energy ratio are reported only when
    --cpu-power is also explicitly supplied.
  • If either is absent the corresponding column prints "N/A".
  Obtain power values from actual measurement before publishing:
      WH  → TT-SMI:  tt-smi -s --json | jq '.boards[0].telemetry.input_current'
             or tt_power_sidecar.py --backend sysfs
      CPU → RAPL:    /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj

CPU baseline note:
  The prior FFT paper (Brown et al. ISC 2025) used a native OpenMP C++
  FFT on Xeon Platinum 8260 (24 cores, 353 W measured).  This script uses
  torch.fft.fft (PyTorch CPU backend) by default, which is NOT the same
  baseline.  Speedup and energy numbers from this script are therefore NOT
  directly comparable to those in Brown et al.  Label your baseline clearly
  in any paper or report.

bfloat16 note:
  WH B0 executes bf16 natively (no upcast).
  NumPy promotes bf16 input to float64 internally (no native bf16 path).
  PyTorch promotes bf16 input to float32 internally on CPU.
  The --dtype bfloat16 flag therefore benchmarks different compute precisions
  on each side.  The output labels this explicitly so you don't confuse them.

Mode:
  --mode real     Input is real-valued (B, N); WH returns (re, im) tensors.
                  CPU reference uses numpy.fft.rfft or fft depending on context.
  --mode complex  Input is complex-valued (B, N); WH receives real part only,
                  imaginary part only and runs fft on each independently.
                  (WH B0 does not have a native complex dtype — it stores
                  re/im as separate real tensors, which is what this mode
                  benchmarks explicitly.)
═══════════════════════════════════════════════════════════════════════════════

CLI quick reference:
  --backend  numpy | torch | ttnn | check | compare
  --mode     real | complex
  --dtype    float32 | bfloat16
  --n        FFT length
  --batch    batch size (rows)
  --warmup   warmup iterations (default 10)
  --iters    timed iterations  (default 20)
  --wh-power   measured WH avg power in W  (no default — must be supplied for energy)
  --cpu-power  measured CPU avg power in W (no default — energy ratio skipped if absent)
  --json-out   path to write JSON result
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _fft_flops(n: int, batch: int) -> float:
    """5 * batch * n * log2(n) — standard complex FFT FLOP count (cuFFT docs)."""
    return 5.0 * batch * n * math.log2(max(n, 2))


def _make_real_np(n: int, batch: int, seed: int, dtype: str) -> np.ndarray:
    """Return (batch, n) float32 array, quantised to bf16 if dtype='bfloat16'.

    Always returns float32 memory layout; bf16 quantisation has already been
    applied element-wise via PyTorch so the values are bf16-representable.
    NumPy has no native bf16 dtype, so we carry them as float32.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, n)).astype(np.float32)
    if dtype == "bfloat16":
        x = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    return x


def _percentile(vals: List[float], p: float) -> float:
    return float(np.percentile(vals, p))


@dataclass
class TimingResult:
    backend: str            # "numpy" | "torch" | "ttnn"
    n: int
    batch: int
    dtype: str
    effective_dtype: str    # what compute actually ran in (may differ for bf16 on CPU)
    mode: str               # "real" | "complex"
    warmup: int
    iters: int
    times_ms: List[float] = field(repr=False)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def min_ms(self) -> float:
        return min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.times_ms)

    @property
    def p25_ms(self) -> float:
        return _percentile(self.times_ms, 25)

    @property
    def p75_ms(self) -> float:
        return _percentile(self.times_ms, 75)

    def gflops_s(self) -> float:
        return _fft_flops(self.n, self.batch) / (self.median_ms * 1e-3) / 1e9

    def energy_j(self, avg_power_w: float) -> float:
        return avg_power_w * self.median_ms * 1e-3

    def joules_per_fft(self, avg_power_w: float) -> float:
        return self.energy_j(avg_power_w) / self.batch

    def ffts_per_joule(self, avg_power_w: float) -> float:
        e = self.energy_j(avg_power_w)
        return self.batch / e if e > 0 else float("inf")

    def to_dict(self, avg_power_w: Optional[float] = None) -> dict:
        d: dict = {
            "backend": self.backend,
            "n": self.n,
            "batch": self.batch,
            "dtype_requested": self.dtype,
            "effective_compute_dtype": self.effective_dtype,
            "mode": self.mode,
            "warmup": self.warmup,
            "iters": self.iters,
            "median_ms": round(self.median_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "p25_ms": round(self.p25_ms, 4),
            "p75_ms": round(self.p75_ms, 4),
            "gflops_s": round(self.gflops_s(), 4),
        }
        if avg_power_w is not None:
            d["avg_power_w_used"] = avg_power_w
            d["energy_j"] = round(self.energy_j(avg_power_w), 6)
            d["joules_per_fft"] = round(self.joules_per_fft(avg_power_w), 8)
            d["ffts_per_joule"] = round(self.ffts_per_joule(avg_power_w), 2)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# CPU backends
# ─────────────────────────────────────────────────────────────────────────────

def _cpu_effective_dtype(requested: str, backend: str) -> str:
    """Document what precision the CPU actually computes at."""
    if requested == "bfloat16":
        if backend == "numpy":
            return "float64 (numpy upcasts bf16→float64 internally)"
        else:
            return "float32 (torch upcasts bf16→float32 on CPU)"
    return requested


def benchmark_numpy(x_np: np.ndarray, warmup: int, iters: int,
                    dtype: str, mode: str) -> TimingResult:
    """Benchmark numpy.fft.fft (pocketfft, all available CPU cores).

    For 'complex' mode we concatenate real and imaginary arrays and time
    two independent FFTs (matching the WH approach of separate re/im).
    """
    if mode == "complex":
        # Generate matching imaginary part with a different seed offset
        rng = np.random.default_rng(seed=1)
        x_im = rng.standard_normal(x_np.shape).astype(x_np.dtype)
        inputs = [x_np, x_im]
    else:
        inputs = [x_np]

    # Warmup
    for _ in range(warmup):
        for xi in inputs:
            np.fft.fft(xi, axis=-1)

    times_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for xi in inputs:
            y = np.fft.fft(xi, axis=-1)
            _ = y[0, 0]
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return TimingResult(
        backend="numpy",
        n=x_np.shape[1],
        batch=x_np.shape[0],
        dtype=dtype,
        effective_dtype=_cpu_effective_dtype(dtype, "numpy"),
        mode=mode,
        warmup=warmup,
        iters=iters,
        times_ms=times_ms,
    )


def benchmark_torch_cpu(x_np: np.ndarray, warmup: int, iters: int,
                         dtype: str, mode: str) -> TimingResult:
    """Benchmark torch.fft.fft on CPU."""
    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    x_t = torch.from_numpy(x_np).to(torch_dtype)

    if mode == "complex":
        rng = np.random.default_rng(seed=1)
        x_im_np = rng.standard_normal(x_np.shape).astype(x_np.dtype)
        x_im = torch.from_numpy(x_im_np).to(torch_dtype)
        inputs = [x_t, x_im]
    else:
        inputs = [x_t]

    for _ in range(warmup):
        for xi in inputs:
            torch.fft.fft(xi)

    times_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for xi in inputs:
            y = torch.fft.fft(xi)
            _ = y[0, 0].real.item()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return TimingResult(
        backend="torch_cpu",
        n=x_np.shape[1],
        batch=x_np.shape[0],
        dtype=dtype,
        effective_dtype=_cpu_effective_dtype(dtype, "torch"),
        mode=mode,
        warmup=warmup,
        iters=iters,
        times_ms=times_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Wormhole / ttnn backend
# ─────────────────────────────────────────────────────────────────────────────

def _tt_dtype(dtype: str):
    import ttnn
    return ttnn.float32 if dtype == "float32" else ttnn.bfloat16


def _upload_real(x_np: np.ndarray, device, tt_dtype) -> "ttnn.Tensor":
    import ttnn
    return ttnn.from_torch(
        torch.from_numpy(x_np),
        dtype=tt_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def _run_fft(tt_in: "ttnn.Tensor") -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
    """ttnn.experimental.fft returns (re, im) — WH has no complex dtype."""
    import ttnn
    return ttnn.experimental.fft(tt_in)


def benchmark_ttnn(x_np: np.ndarray, warmup: int, iters: int,
                   dtype: str, mode: str, device_id: int) -> TimingResult:
    """Benchmark WH kernel execution time only (no D2H, no upload in timing).

    Timing window: dispatch → ttnn.synchronize_device().
    For 'complex' mode: two separate FFT calls (re path + im path) are timed
    together, matching the CPU 'complex' mode which also runs two FFTs.
    """
    import ttnn

    device = ttnn.open_device(device_id=device_id)
    device.enable_program_cache()
    try:
        tt_re = _upload_real(x_np, device, _tt_dtype(dtype))

        if mode == "complex":
            rng = np.random.default_rng(seed=1)
            x_im_np = rng.standard_normal(x_np.shape).astype(x_np.dtype)
            tt_im = _upload_real(x_im_np, device, _tt_dtype(dtype))
            inputs = [tt_re, tt_im]
        else:
            inputs = [tt_re]

        # Warmup — fills program cache so first timed call is not JIT
        for _ in range(warmup):
            for tt_in in inputs:
                out_re, out_im = _run_fft(tt_in)
                ttnn.to_torch(out_re)
                ttnn.to_torch(out_im)

        # Timed loop — kernel only
        times_ms: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            for tt_in in inputs:
                _run_fft(tt_in)                   # dispatch, non-blocking
            ttnn.synchronize_device(device)        # wait for all kernels
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        return TimingResult(
            backend="ttnn_wh",
            n=x_np.shape[1],
            batch=x_np.shape[0],
            dtype=dtype,
            effective_dtype=f"{dtype} (native on WH B0)",
            mode=mode,
            warmup=warmup,
            iters=iters,
            times_ms=times_ms,
        )
    finally:
        ttnn.close_device(device)


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy check
# ─────────────────────────────────────────────────────────────────────────────

def check_accuracy(x_np: np.ndarray, dtype: str, mode: str,
                   device_id: int) -> dict:
    """Compare WH output against numpy float64 reference across all batch rows.

    Reports max and mean relative L2 error: ||y_wh - y_ref||_2 / ||y_ref||_2
    per row, then summarises across the batch.
    """
    import ttnn

    if mode == "complex":
        rng = np.random.default_rng(seed=1)
        x_im_np = rng.standard_normal(x_np.shape).astype(x_np.dtype)
    else:
        x_im_np = np.zeros_like(x_np)

    # Double-precision reference
    ref_re = np.fft.fft(x_np.astype(np.float64), axis=-1)
    ref_im = np.fft.fft(x_im_np.astype(np.float64), axis=-1)
    ref = ref_re + 1j * ref_im  # shape (batch, n)

    device = ttnn.open_device(device_id=device_id)
    device.enable_program_cache()
    try:
        tt_re_in = _upload_real(x_np, device, _tt_dtype(dtype))
        tt_im_in = _upload_real(x_im_np, device, _tt_dtype(dtype))

        out_re_re, out_re_im = _run_fft(tt_re_in)  # FFT of real part
        out_im_re, out_im_im = _run_fft(tt_im_in)  # FFT of imag part

        wh_re_re = ttnn.to_torch(out_re_re).float().numpy()  # (batch, n)
        wh_re_im = ttnn.to_torch(out_re_im).float().numpy()
        wh_im_re = ttnn.to_torch(out_im_re).float().numpy()
        wh_im_im = ttnn.to_torch(out_im_im).float().numpy()
    finally:
        ttnn.close_device(device)

    # Reconstruct complex output: FFT(x_re + j*x_im) = FFT(x_re) + j*FFT(x_im)
    wh_out = (wh_re_re + 1j * wh_re_im) + 1j * (wh_im_re + 1j * wh_im_im)
    if mode == "real":
        wh_out = wh_re_re + 1j * wh_re_im  # only the real-input FFT

    # Per-row relative L2 error
    row_rel_errs = []
    for b in range(x_np.shape[0]):
        ref_row = ref[b] if mode == "complex" else (ref_re[b])
        wh_row  = wh_out[b]
        diff = wh_row.astype(np.complex128) - ref_row.astype(np.complex128)
        rel = float(np.linalg.norm(diff) / (np.linalg.norm(ref_row) + 1e-30))
        row_rel_errs.append(rel)

    tol = 5e-4 if dtype == "float32" else 5e-2
    return {
        "n": int(x_np.shape[1]),
        "batch": int(x_np.shape[0]),
        "dtype": dtype,
        "mode": mode,
        "tolerance": tol,
        "max_rel_l2_err": float(max(row_rel_errs)),
        "mean_rel_l2_err": float(sum(row_rel_errs) / len(row_rel_errs)),
        "all_rows_pass": bool(max(row_rel_errs) < tol),
        "per_row_rel_l2_err": [round(e, 8) for e in row_rel_errs],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_perf(x_np: np.ndarray, warmup: int, iters: int,
                 dtype: str, mode: str, device_id: int,
                 wh_power_w: Optional[float], cpu_power_w: Optional[float],
                 cpu_backend: str) -> dict:
    """Run both CPU (torch) and WH backends; compute speedup and optional energy ratio.

    Energy ratio is only computed when both wh_power_w and cpu_power_w are
    explicitly supplied.  If either is None the corresponding energy fields
    are omitted and the ratio is reported as 'N/A'.

    NOTE: the CPU baseline here is torch.fft.fft, NOT the native OpenMP C++
    baseline used in Brown et al. ISC 2025.  Results are not directly
    comparable to that paper.
    """
    if cpu_backend == "numpy":
        cpu_result = benchmark_numpy(x_np, warmup, iters, dtype, mode)
    else:
        cpu_result = benchmark_torch_cpu(x_np, warmup, iters, dtype, mode)

    wh_result = benchmark_ttnn(x_np, warmup, iters, dtype, mode, device_id)

    cpu_med = cpu_result.median_ms
    wh_med  = wh_result.median_ms
    speedup = cpu_med / wh_med

    # Energy and ratio are optional — only when both power values are provided.
    energy_ratio: Optional[float] = None
    if wh_power_w is not None and cpu_power_w is not None:
        cpu_energy_j = cpu_result.energy_j(cpu_power_w)
        wh_energy_j  = wh_result.energy_j(wh_power_w)
        energy_ratio = cpu_energy_j / wh_energy_j if wh_energy_j > 0 else float("inf")

    summary: dict = {
        "n": x_np.shape[1],
        "batch": x_np.shape[0],
        "dtype": dtype,
        "mode": mode,
        "cpu_baseline": cpu_result.backend,
        "wh_faster_than_cpu": bool(speedup > 1.0),
        "speedup_cpu_over_wh": round(speedup, 4),
        "baseline_note": (
            "CPU baseline is torch.fft.fft — NOT the native OpenMP C++ baseline "
            "used in Brown et al. ISC 2025.  Do not compare these numbers directly "
            "to that paper without noting the difference."
        ),
    }
    if energy_ratio is not None:
        summary["wh_more_energy_efficient"] = bool(energy_ratio > 1.0)
        summary["energy_ratio_cpu_over_wh"] = round(energy_ratio, 4)
        summary["power_note"] = (
            f"WH power: {wh_power_w} W, CPU power: {cpu_power_w} W  "
            "(supply --wh-power and --cpu-power from TT-SMI / RAPL measurement)"
        )
    else:
        summary["wh_more_energy_efficient"] = "N/A (--cpu-power not provided)"
        summary["energy_ratio_cpu_over_wh"] = "N/A"
        summary["power_note"] = (
            "Energy ratio not computed — provide --wh-power and --cpu-power "
            "from actual measurement (TT-SMI for WH, RAPL for CPU)."
        )

    return {
        "summary": summary,
        "cpu": cpu_result.to_dict(avg_power_w=cpu_power_w),
        "wh":  wh_result.to_dict(avg_power_w=wh_power_w),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_timing(r: TimingResult, avg_power_w: Optional[float] = None,
                  power_w: Optional[float] = None) -> None:
    # accept both keyword spellings for backwards compatibility
    power_w = avg_power_w if avg_power_w is not None else power_w
    print(f"  backend          : {r.backend}")
    print(f"  N                : {r.n:,}  batch={r.batch}  mode={r.mode}")
    print(f"  dtype requested  : {r.dtype}")
    print(f"  effective compute: {r.effective_dtype}")
    print(f"  warmup / iters   : {r.warmup} / {r.iters}")
    print(f"  median           : {r.median_ms:.3f} ms")
    print(f"  min / max        : {r.min_ms:.3f} / {r.max_ms:.3f} ms")
    print(f"  p25  / p75       : {r.p25_ms:.3f} / {r.p75_ms:.3f} ms")
    print(f"  GFLOPs/s         : {r.gflops_s():.3f}")
    if power_w is not None:
        print(f"  avg power used   : {power_w} W  ⚠ must be from real measurement")
        print(f"  energy           : {r.energy_j(power_w)*1000:.3f} mJ")
        print(f"  joules/fft       : {r.joules_per_fft(power_w):.6f} J")
        print(f"  ffts/joule       : {r.ffts_per_joule(power_w):.2f}")


def _fmt(val, fmt=".3f") -> str:
    """Format a value, printing 'N/A' if it is None or a string sentinel."""
    if val is None or isinstance(val, str):
        return "N/A"
    return format(val, fmt)


def _print_compare(result: dict) -> None:
    s   = result["summary"]
    cpu = result["cpu"]
    wh  = result["wh"]

    cpu_energy_mj = f"{cpu['energy_j']*1000:.3f}" if "energy_j" in cpu else "N/A"
    wh_energy_mj  = f"{wh['energy_j']*1000:.3f}"  if "energy_j" in wh  else "N/A"
    cpu_jpf = f"{cpu['joules_per_fft']:.6f}" if "joules_per_fft" in cpu else "N/A"
    wh_jpf  = f"{wh['joules_per_fft']:.6f}"  if "joules_per_fft" in wh  else "N/A"
    cpu_fpj = f"{cpu['ffts_per_joule']:.2f}" if "ffts_per_joule" in cpu else "N/A"
    wh_fpj  = f"{wh['ffts_per_joule']:.2f}"  if "ffts_per_joule" in wh  else "N/A"

    print(f"\n{'═'*66}")
    print(f"  N={s['n']:,}  batch={s['batch']}  dtype={s['dtype']}  mode={s['mode']}")
    print(f"{'─'*66}")
    print(f"  {'':24s}  {'CPU (torch)':>14}  {'WH B0 (ttnn)':>14}")
    print(f"  {'backend':24s}  {cpu['backend']:>14}  {wh['backend']:>14}")
    print(f"  {'eff. compute':24s}  {cpu['effective_compute_dtype']!s:>14}  {wh['effective_compute_dtype']!s:>14}")
    print(f"  {'median (ms)':24s}  {cpu['median_ms']:>14.3f}  {wh['median_ms']:>14.3f}")
    print(f"  {'min / max (ms)':24s}  {cpu['min_ms']:.2f}/{cpu['max_ms']:.2f}  {wh['min_ms']:.2f}/{wh['max_ms']:.2f}")
    print(f"  {'GFLOPs/s':24s}  {cpu['gflops_s']:>14.3f}  {wh['gflops_s']:>14.3f}")
    print(f"  {'energy (mJ)':24s}  {cpu_energy_mj:>14}  {wh_energy_mj:>14}")
    print(f"  {'joules/fft':24s}  {cpu_jpf:>14}  {wh_jpf:>14}")
    print(f"  {'ffts/joule':24s}  {cpu_fpj:>14}  {wh_fpj:>14}")
    print(f"{'─'*66}")

    faster  = "YES" if s["wh_faster_than_cpu"] else "NO"
    greener = s["wh_more_energy_efficient"]
    greener_str = "YES" if greener is True else ("NO" if greener is False else str(greener))
    energy_ratio_str = _fmt(s["energy_ratio_cpu_over_wh"])

    print(f"  WH faster than CPU?        {faster}   (speedup {s['speedup_cpu_over_wh']:.3f}×)")
    print(f"  WH more energy efficient?  {greener_str}")
    if energy_ratio_str != "N/A":
        print(f"  energy ratio (CPU/WH):     {energy_ratio_str}×")
    print(f"{'─'*66}")
    print(f"  ⚠ BASELINE NOTE: {s['baseline_note']}")
    print(f"  ⚠ POWER  NOTE:   {s['power_note']}")
    print(f"{'═'*66}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="ttnn.experimental.fft vs CPU benchmark — scientifically fair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fft_energy_compare.py --backend numpy --n 1048576\n"
            "  python fft_energy_compare.py --backend ttnn --n 1048576 --dtype bfloat16\n"
            "  python fft_energy_compare.py --backend check --n 64512 --batch 4\n"
            "  python fft_energy_compare.py --backend compare --n 1048576 \\\n"
            "      --wh-power 42 --cpu-power 353 --json-out result.json\n"
        ),
    )
    p.add_argument("--backend", required=True,
                   choices=["numpy", "torch", "ttnn", "check", "compare"],
                   help=(
                       "numpy  = CPU baseline via numpy.fft.fft (pocketfft, multi-core)\n"
                       "torch  = CPU baseline via torch.fft.fft\n"
                       "ttnn   = WH B0 kernel-only timing\n"
                       "check  = accuracy check WH vs numpy float64 reference\n"
                       "compare= both CPU and WH side-by-side with energy"
                   ))
    p.add_argument("--mode", default="real", choices=["real", "complex"],
                   help=(
                       "real    = real-valued input (B,N); WH returns (re,im)\n"
                       "complex = two separate FFTs (re-path + im-path) timed together"
                   ))
    p.add_argument("--n",      type=int,   default=16384, help="FFT length N")
    p.add_argument("--batch",  type=int,   default=1,     help="batch size (rows)")
    p.add_argument("--warmup", type=int,   default=10,
                   help="warmup iterations — fills ttnn program cache")
    p.add_argument("--iters",  type=int,   default=20,    help="timed iterations")
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--dtype",  default="float32",
                   choices=["float32", "bfloat16"],
                   help="float32 or bfloat16  (float64 not supported on WH B0)")
    p.add_argument("--device-id",  type=int,   default=0)
    p.add_argument("--cpu-backend", default="torch", choices=["numpy", "torch"],
                   help="which CPU library to use in 'compare' mode (default: torch)")
    p.add_argument("--wh-power",  type=float, default=None,
                   help=(
                       "MEASURED WH avg power in W (from TT-SMI / tt_power_sidecar.py). "
                       "No default — WH energy is reported as N/A if not supplied. "
                       "Brown et al. measured 42 W on n300; your value may differ."
                   ))
    p.add_argument("--cpu-power", type=float, default=None,
                   help=(
                       "MEASURED CPU avg power in W (from RAPL). "
                       "No default — CPU energy and energy ratio are N/A if not supplied. "
                       "Brown et al. measured 353 W on Xeon Platinum; your value may differ."
                   ))
    p.add_argument("--json-out", default=None,
                   help="Write full result dict to this JSON file")
    args = p.parse_args()

    x_np = _make_real_np(args.n, args.batch, args.seed, args.dtype)

    result: dict | TimingResult

    if args.backend == "numpy":
        result = benchmark_numpy(x_np, args.warmup, args.iters, args.dtype, args.mode)
        print(f"\n── numpy CPU benchmark ──")
        print(f"  ⚠ NOTE: numpy is not the same baseline as Brown et al. OpenMP C++ FFT.")
        _print_timing(result, avg_power_w=args.cpu_power)
        result = result.to_dict(avg_power_w=args.cpu_power)

    elif args.backend == "torch":
        result = benchmark_torch_cpu(x_np, args.warmup, args.iters, args.dtype, args.mode)
        print(f"\n── torch CPU benchmark (default comparison baseline) ──")
        print(f"  ⚠ NOTE: torch.fft.fft is not the same baseline as Brown et al. OpenMP C++ FFT.")
        _print_timing(result, avg_power_w=args.cpu_power)
        result = result.to_dict(avg_power_w=args.cpu_power)

    elif args.backend == "ttnn":
        result = benchmark_ttnn(x_np, args.warmup, args.iters,
                                args.dtype, args.mode, args.device_id)
        print(f"\n── ttnn WH B0 benchmark (kernel-only timing) ──")
        if args.wh_power is None:
            print(f"  ⚠ --wh-power not supplied: energy will be reported as N/A.")
        _print_timing(result, avg_power_w=args.wh_power)
        result = result.to_dict(avg_power_w=args.wh_power)

    elif args.backend == "check":
        result = check_accuracy(x_np, args.dtype, args.mode, args.device_id)
        print(f"\n── accuracy check ──")
        print(f"  N={result['n']:,}  batch={result['batch']}  "
              f"dtype={result['dtype']}  mode={result['mode']}")
        print(f"  max  rel-L2 error : {result['max_rel_l2_err']:.3e}")
        print(f"  mean rel-L2 error : {result['mean_rel_l2_err']:.3e}")
        print(f"  tolerance         : {result['tolerance']:.1e}")
        status = "PASS" if result["all_rows_pass"] else "FAIL"
        print(f"  all {result['batch']} rows pass: {status}")

    else:  # compare
        result = compare_perf(
            x_np, args.warmup, args.iters,
            args.dtype, args.mode, args.device_id,
            wh_power_w=args.wh_power, cpu_power_w=args.cpu_power,
            cpu_backend=args.cpu_backend,
        )
        _print_compare(result)

    # Remove verbose per-iteration list before printing/saving JSON
    if isinstance(result, dict):
        result.pop("times_ms", None)
        for sub in result.values():
            if isinstance(sub, dict):
                sub.pop("times_ms", None)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult written to {args.json_out}")


if __name__ == "__main__":
    main()
