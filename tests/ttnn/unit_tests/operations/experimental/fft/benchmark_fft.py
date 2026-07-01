#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
FFT benchmark for HPEC 2026 — paper-safe results only.

════════════════════════════════════════════════════════════════════════════════
TIMING METHODOLOGY
════════════════════════════════════════════════════════════════════════════════
WH timing  — kernel-only execution time.
  Window:  right before ttnn.experimental.fft(...)
         → ttnn.synchronize_device(device)
  Excludes: host→device upload, device→host download, Python overhead.
  Rationale: matches Brown et al. ISC 2025 ("performance numbers for WH are
             execution time only").

CPU timing — wall-clock for the full numpy.fft.fft() call (plan cached after
  warmup). Labelled "numpy_cpu" in all output. This is a practical Python
  software baseline, NOT the native OpenMP C++ baseline used by Brown et al.
  Do not compare CPU speedup numbers directly to that paper.

════════════════════════════════════════════════════════════════════════════════
ENERGY METHODOLOGY
════════════════════════════════════════════════════════════════════════════════
Energy is NEVER computed from borrowed constants or TDP estimates.
Two modes:

  Mode 0 — no sidecar dirs supplied (default):
    All energy fields = N/A.  energy_ratio_valid = False.
    Use this for timing-only runs.

  Mode 1 — --wh-energy-dir and/or --cpu-energy-dir supplied:
    Per-run sidecar JSON is loaded for each (N, dtype) point.
    File naming: N{N}_{dtype}.json inside the directory.
    Example:  wh_energy/N1048576_fp32.json
    Accepted schemas:
      {"energy_J": 1.220}                        ← preferred (direct measurement)
      {"energy_J": 1.220, "avg_power_W": 42.1}   ← energy_J used, power informational
      {"avg_power_W": 42.1, "duration_s": 0.029} ← fallback: power × duration
    If no file exists for a given (N, dtype), that point's energy = N/A.
    energy_ratio_valid = True only when both WH and CPU energy are present.

bfloat16 note:
  WH B0 executes bf16 natively — no upcast.
  NumPy pocketfft has no native bf16 path; it upcasts input to float64
  internally.  bf16 CPU timings therefore reflect float64 compute, not bf16.
  Do not interpret CPU bf16 numbers as a precision-matched native bf16 baseline.

To measure energy per run:
  WH  → tt_power_sidecar.py --backend sysfs --out wh_energy/N{N}_{dtype}.json
         or integrate TT-SMI power over the benchmark window.
  CPU → RAPL: read energy_uj before/after run → compute energy_J directly.

════════════════════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════════════════════
  # Runtime only (no energy) — default mode
  python benchmark_fft.py --dtype fp32 --warmup 10 --runs 50 --csv out.csv

  # Single N
  python benchmark_fft.py --n 1048576 --dtype fp32 --csv out.csv

  # Per-run sidecar energy (energy_J used directly when available)
  python benchmark_fft.py --wh-energy-dir wh_energy/ \\
                           --cpu-energy-dir cpu_energy/ --csv out.csv

  # Single tier
  python benchmark_fft.py --two-pass --dtype fp32 --csv out.csv

GFLOPs/s convention (matches cuFFT docs):
  complex FFT of length N:  5 × B × N × log2(N)  FLOPs
"""

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

import numpy as np
import torch
import ttnn


# ─────────────────────────────────────────────────────────────────────────────
# N lists per algorithm tier
# ─────────────────────────────────────────────────────────────────────────────

POW2_STOCKHAM  = [32, 64, 128, 256, 512, 1024]
POW2_TWO_PASS  = [2048, 4096, 8192, 65536, 1 << 17, 1 << 20]
POW2_THREE_PASS = [1 << 21, 1 << 23, 1 << 24]   # 2^26 OOMs on 1 GB WH B0
BLUESTEIN_N = [
    97, 127, 257, 509,
    1000, 3000, 9999,
    64512,           # 63 × 1024, M = 2^17
    525312,          # 513 × 1024  (XL, M = 3-pass inner)
    786432,          # 768 × 1024  (XL)
]

# Stockham: one transform per core → fill all 64 cores with B=64.
# All other tiers distribute work across cores internally → B=1.
STOCKHAM_BATCH = 64


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

_NA = "N/A"                 # sentinel for unmeasured energy fields in CSV
_BF16_NOTE_PRINTED = False  # print bf16 upcast warning only once per run


@dataclass
class BenchResult:
    # ── identity ──────────────────────────────────────────────────────────────
    N: int
    batch: int                  # B=64 for Stockham, B=1 for all others
    dtype: str                  # "fp32" | "bf16"
    algorithm: str              # "stockham" | "two_pass" | "three_pass" | "bluestein"
    cpu_baseline: str           # always "numpy_cpu"

    # ── timing ────────────────────────────────────────────────────────────────
    wh_median_ms: float
    wh_p25_ms: float
    wh_p75_ms: float
    cpu_median_ms: float        # numpy wall-clock
    gflops_s: float             # WH device GFLOPs/s
    speedup_vs_cpu: float       # cpu_median / wh_median  (>1 = WH faster)

    # ── accuracy ──────────────────────────────────────────────────────────────
    rel_err: float              # ||y_wh - y_ref||_2 / ||y_ref||_2

    # ── energy — None when not measured ───────────────────────────────────────
    wh_power_w: Optional[float]         # informational; None if not available
    wh_energy_j: Optional[float]        # from sidecar energy_J or power×time
    wh_joules_per_fft: Optional[float]  # wh_energy_j / batch
    wh_ffts_per_joule: Optional[float]  # batch / wh_energy_j
    wh_energy_source: Optional[str]     # "sidecar_energy_J" | "power_x_time" | None

    cpu_power_w: Optional[float]           # informational; None if not available
    cpu_energy_j: Optional[float]         # from sidecar energy_J or power×time
    cpu_joules_per_fft: Optional[float]   # cpu_energy_j / batch
    cpu_ffts_per_joule: Optional[float]   # batch / cpu_energy_j
    cpu_energy_source: Optional[str]      # "sidecar_energy_J" | "power_x_time" | None
    cpu_duration_s: Optional[float]       # duration_s from sidecar if present
    wh_duration_s: Optional[float]        # duration_s from sidecar if present
    energy_ratio: Optional[float]         # cpu_energy_j / wh_energy_j  (>1 = WH greener)
    energy_ratio_valid: bool              # True only when both energy values are real measurements

    def csv_row(self) -> Dict:
        """Flat dict for CSV — replaces None with 'N/A'."""
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = _NA if v is None else v
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _algorithm_label(N: int) -> str:
    if N & (N - 1) == 0:
        if N <= 1024:      return "stockham"
        if N <= (1 << 20): return "two_pass"
        return "three_pass"
    return "bluestein"


def _fft_flops(N: int, B: int) -> float:
    """5 × B × N × log2(N) — standard complex FFT FLOP count (cuFFT docs)."""
    return 5.0 * B * N * math.log2(max(N, 2))



def _load_sidecar_energy(energy_dir: Optional[str],
                         N: int,
                         dtype_str: str) -> tuple:
    """Look up per-run sidecar JSON for this (N, dtype) point.

    File name convention: {energy_dir}/N{N}_{dtype}.json
    e.g.  wh_energy/N1048576_fp32.json

    Returns: (energy_j, power_w_info, duration_s, source_label)
      energy_j      — direct energy in Joules if sidecar has energy_J,
                      else power × duration if both present, else None.
      power_w_info  — avg_power_W from sidecar (informational), or None.
      duration_s    — duration_s field from sidecar if present, else None.
      source_label  — "sidecar_energy_J" | "sidecar_power_x_duration" | None
    """
    if energy_dir is None:
        return None, None, None, None

    import os
    path = os.path.join(energy_dir, f"N{N}_{dtype_str}.json")
    if not os.path.exists(path):
        return None, None, None, None

    with open(path) as f:
        data = json.load(f)

    power_w  = data.get("avg_power_W") or data.get("avg_power_w")
    if power_w is not None:
        power_w = float(power_w)

    duration = data.get("duration_s") or data.get("duration")
    if duration is not None:
        duration = float(duration)

    # Prefer direct energy_J measurement
    if "energy_J" in data or "energy_j" in data:
        e = float(data.get("energy_J") or data.get("energy_j"))
        return e, power_w, duration, "sidecar_energy_J"

    # Fallback: avg_power_W × duration_s
    if power_w is not None and duration is not None:
        e = power_w * duration
        return e, power_w, duration, "sidecar_power_x_duration"

    return None, power_w, duration, None


def _resolve_energy(energy_dir: Optional[str],
                    N: int,
                    dtype_str: str,
                    batch: int) -> tuple:
    """Resolve energy for one benchmark point from per-run sidecar JSON only.

    No fallback to global power scalars — if no matching sidecar file exists
    for this (N, dtype), all energy fields are None (reported as N/A).

    Returns: (energy_j, power_w_info, duration_s, joules_per_fft,
              ffts_per_joule, source)
    """
    # Per-run sidecar only — no fallback to global power scalar.
    # If no matching sidecar file exists, energy is N/A for this point.
    e, pw_info, duration_s, source = _load_sidecar_energy(energy_dir, N, dtype_str)

    if e is None:
        return None, None, None, None, None, None

    jpf = e / batch if batch > 0 else None
    fpj = batch / e if e > 0 else None
    return e, pw_info, duration_s, jpf, fpj, source


def _compute_energy_ratio(wh_energy_j: Optional[float],
                          cpu_energy_j: Optional[float]) -> tuple:
    """Return (ratio, valid).  valid=True only when both energies are real."""
    if wh_energy_j is None or cpu_energy_j is None:
        return None, False
    ratio = cpu_energy_j / wh_energy_j if wh_energy_j > 0 else None
    return ratio, True


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_device():
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    return device


def _upload(x_np: np.ndarray, device, tt_dtype, B: int) -> ttnn.Tensor:
    """Tile 1-D x_np → (B, N) and upload to device."""
    t = torch.from_numpy(np.tile(x_np.reshape(1, -1), (B, 1)))
    return ttnn.from_torch(t, dtype=tt_dtype,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _run_wh_fft(tt_in: ttnn.Tensor):
    return ttnn.experimental.fft(tt_in)   # returns (re, im)


def _download_first_row(re: ttnn.Tensor, im: ttnn.Tensor,
                        N: int, B: int) -> np.ndarray:
    r = ttnn.to_torch(re).reshape(B, N)[0].to(torch.float32).numpy()
    i = ttnn.to_torch(im).reshape(B, N)[0].to(torch.float32).numpy()
    return r + 1j * i


def _wh_fft_timed(tt_in: ttnn.Tensor, n_runs: int, device) -> List[float]:
    """Kernel-only timing: dispatch → synchronize_device (no D2H)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        re, im = _run_wh_fft(tt_in)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
        re.deallocate()
        im.deallocate()
    return times


def _cpu_fft_timed(x_np: np.ndarray, n_runs: int) -> List[float]:
    """Wall-clock for numpy.fft.fft (single-threaded pocketfft, plan cached)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = np.fft.fft(x_np, axis=-1)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return times


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark function
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one(
    N: int,
    dtype_str: str,
    device,
    warmup: int,
    runs: int,
    wh_energy_dir: Optional[str],      # dir with per-run WH sidecar JSONs
    cpu_energy_dir: Optional[str],     # dir with per-run CPU sidecar JSONs
) -> Optional[BenchResult]:

    # DRAM safety guard (fp32/bf16 N=2^27 = 512 MB input → OOM on 1 GB WH B0)
    if N > (1 << 26):
        print(f"  SKIP N={N:>10,} {dtype_str} (DRAM OOM on WH B0)")
        return None

    tt_dtype = ttnn.float32 if dtype_str == "fp32" else ttnn.bfloat16
    algo = _algorithm_label(N)
    B    = STOCKHAM_BATCH if algo == "stockham" else 1

    rng = np.random.default_rng(N)
    x_np = rng.standard_normal(N).astype(np.float32)
    if dtype_str == "bf16":
        x_np = torch.tensor(x_np).to(torch.bfloat16).float().numpy()

    print(f"  bench N={N:>10,}  B={B:<2}  {dtype_str}  {algo:<12}", end="", flush=True)

    # ── upload ────────────────────────────────────────────────────────────────
    try:
        tt_in = _upload(x_np, device, tt_dtype, B=B)
    except RuntimeError as e:
        print(f"  → SKIP (alloc failed: {e})")
        return None

    # ── warmup (fills program cache) ──────────────────────────────────────────
    try:
        for _ in range(warmup):
            re, im = _run_wh_fft(tt_in)
            ttnn.to_torch(re)
            ttnn.to_torch(im)
            re.deallocate()
            im.deallocate()
    except RuntimeError as e:
        print(f"  → SKIP (warmup failed: {e})")
        return None

    # ── accuracy (first row vs numpy float64 reference) ───────────────────────
    ref = np.fft.fft(x_np.astype(np.float64))
    re, im = _run_wh_fft(tt_in)
    got = _download_first_row(re, im, N, B)
    rel_err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30))
    re.deallocate()
    im.deallocate()

    # ── WH timed runs (kernel only, no D2H) ───────────────────────────────────
    wh_times = sorted(_wh_fft_timed(tt_in, runs, device))
    wh_med = float(np.median(wh_times))
    wh_p25 = float(np.percentile(wh_times, 25))
    wh_p75 = float(np.percentile(wh_times, 75))

    # ── CPU baseline (numpy pocketfft, single-threaded) ───────────────────────
    x_batch = np.tile(x_np, (B, 1))   # shape (B, N) — same total work as WH
    cpu_times = sorted(_cpu_fft_timed(x_batch, max(runs, 10)))
    cpu_med = float(np.median(cpu_times))

    # ── bf16 note — printed once per process, not once per N ──────────────────
    global _BF16_NOTE_PRINTED
    if dtype_str == "bf16" and not _BF16_NOTE_PRINTED:
        print(f"\n  ⚠ bf16 note: CPU numpy baseline upcasts bf16→float64 internally. "
              f"CPU bf16 time is NOT a precision-matched native bf16 baseline.")
        _BF16_NOTE_PRINTED = True

    # ── derived metrics ───────────────────────────────────────────────────────
    flops    = _fft_flops(N, B)
    gflops_s = flops / (wh_med * 1e-3) / 1e9
    speedup  = cpu_med / wh_med

    # WH energy — per-run sidecar only; N/A if no matching file.
    wh_e, wh_pw_info, wh_dur_s, wh_jpf, wh_fpj, wh_src = _resolve_energy(
        wh_energy_dir, N, dtype_str, B)

    # CPU energy — per-run sidecar only; N/A if no matching file.
    # Use B (not 1): CPU baseline runs B FFTs via np.tile(x_np, (B, 1)).
    # For Stockham B=64; for all other tiers B=1.
    cpu_e, cpu_pw_info, cpu_dur_s, cpu_jpf, cpu_fpj, cpu_src = _resolve_energy(
        cpu_energy_dir, N, dtype_str, B)

    ratio, ratio_valid = _compute_energy_ratio(wh_e, cpu_e)

    # ── progress line ─────────────────────────────────────────────────────────
    if ratio_valid and ratio is not None:
        energy_str = f"energy={ratio:.1f}× (wh:{wh_src}, cpu:{cpu_src})"
    else:
        energy_str = "energy=N/A"
    print(f"  WH={wh_med:7.2f}ms  CPU(numpy)={cpu_med:7.2f}ms  "
          f"{gflops_s:5.2f}GFlops/s  speedup={speedup:.2f}×  "
          f"{energy_str}  err={rel_err:.1e}")

    return BenchResult(
        N=N, batch=B, dtype=dtype_str, algorithm=algo,
        cpu_baseline="numpy_cpu",
        wh_median_ms=wh_med, wh_p25_ms=wh_p25, wh_p75_ms=wh_p75,
        cpu_median_ms=cpu_med,
        gflops_s=gflops_s,
        speedup_vs_cpu=speedup,
        rel_err=rel_err,
        wh_power_w=wh_pw_info,
        wh_energy_j=wh_e,
        wh_joules_per_fft=wh_jpf,
        wh_ffts_per_joule=wh_fpj,
        wh_energy_source=wh_src,
        wh_duration_s=wh_dur_s,
        cpu_power_w=cpu_pw_info,
        cpu_energy_j=cpu_e,
        cpu_joules_per_fft=cpu_jpf,
        cpu_ffts_per_joule=cpu_fpj,
        cpu_energy_source=cpu_src,
        cpu_duration_s=cpu_dur_s,
        energy_ratio=ratio,
        energy_ratio_valid=ratio_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary printers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: Optional[float], fmt: str = ".2f", suffix: str = "") -> str:
    return _NA if v is None else f"{v:{fmt}}{suffix}"


def _print_summary_table(results: List[BenchResult]) -> None:
    has_energy = any(r.energy_ratio_valid for r in results)
    has_wh_e   = any(r.wh_energy_j is not None for r in results)

    hdr = (f"{'N':>10}  {'B':>2}  {'dtype':<5}  {'algo':<12}  "
           f"{'WH(ms)':>8}  {'CPU(ms)':>8}  {'GFlops/s':>9}  {'Speedup':>8}  "
           f"{'RelErr':>8}")
    if has_wh_e:
        hdr += f"  {'WH_mJ':>8}  {'J/FFT':>10}"
    if has_energy:
        hdr += f"  {'Energy×':>8}"

    width = len(hdr)
    print("\n" + "=" * width)
    print(hdr)
    print("-" * width)

    for r in results:
        line = (f"{r.N:>10,}  {r.batch:>2}  {r.dtype:<5}  {r.algorithm:<12}  "
                f"{r.wh_median_ms:>8.2f}  {r.cpu_median_ms:>8.2f}  "
                f"{r.gflops_s:>9.2f}  {r.speedup_vs_cpu:>8.2f}×  "
                f"{r.rel_err:>8.1e}")
        if has_wh_e:
            wh_mj  = _fmt(r.wh_energy_j,       ".1f") if r.wh_energy_j is None else f"{r.wh_energy_j*1000:>8.1f}"
            j_per_f = _fmt(r.wh_joules_per_fft, ".6f")
            line += f"  {wh_mj:>8}  {j_per_f:>10}"
        if has_energy:
            line += f"  {_fmt(r.energy_ratio, '.2f', '×'):>8}"
        print(line)

    print("=" * width)

    if not has_wh_e:
        print("  ⚠ Energy not reported — supply --wh-energy-dir to enable.")
    elif not has_energy:
        print("  ⚠ Energy ratio not reported — supply --cpu-energy-dir too.")
    print(f"  ⚠ CPU baseline = numpy_cpu (pocketfft). NOT Brown et al. native OpenMP C++ FFT.")
    print(f"  ⚠ bf16 CPU numbers reflect float64 compute (numpy internal upcast) — "
          f"not a precision-matched bf16 baseline.")


def _print_per_algo_summary(results: List[BenchResult]) -> None:
    """Print median GFLOPs/s per algorithm tier (fp32 only)."""
    by_algo: Dict[str, List[BenchResult]] = defaultdict(list)
    for r in results:
        if r.dtype == "fp32":
            by_algo[r.algorithm].append(r)

    if not by_algo:
        return

    print("\n── Per-algorithm summary (fp32, median GFLOPs/s) ──")
    for algo in ["stockham", "two_pass", "three_pass", "bluestein"]:
        rs = by_algo.get(algo)
        if not rs:
            continue
        med_gf = sorted(r.gflops_s for r in rs)[len(rs) // 2]
        med_sp = sorted(r.speedup_vs_cpu for r in rs)[len(rs) // 2]
        ratio_str = ""
        valid_ratios = [r.energy_ratio for r in rs if r.energy_ratio_valid and r.energy_ratio is not None]
        if valid_ratios:
            med_er = sorted(valid_ratios)[len(valid_ratios) // 2]
            ratio_str = f"  energy_ratio={med_er:.1f}× (measured)"
        print(f"  {algo:<12}: {med_gf:5.2f} GFLOPs/s  speedup={med_sp:.2f}×{ratio_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FFT benchmark for HPEC 2026 — paper-safe results only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dtype", choices=["fp32", "bf16", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=10,
                        help="warmup iterations per (N, dtype)")
    parser.add_argument("--runs",   type=int, default=50,
                        help="timed iterations per (N, dtype)")
    parser.add_argument("--csv",    default="fft_benchmark.csv",
                        help="output CSV path")

    parser.add_argument("--n", type=int, default=None,
                        help="Run only a single FFT size N (overrides tier flags). "
                             "Useful for generating one sidecar JSON per N: "
                             "python benchmark_fft.py --n 1048576 --dtype fp32")

    # Tier selection (ignored when --n is set)
    parser.add_argument("--stockham",    action="store_true")
    parser.add_argument("--two-pass",    action="store_true")
    parser.add_argument("--three-pass",  action="store_true")
    parser.add_argument("--bluestein",   action="store_true")

    # Energy inputs — per-run sidecar only, no global power scalars
    pwr = parser.add_argument_group(
        "energy (sidecar JSONs only — no global power fallback)")
    pwr.add_argument("--wh-energy-dir",  default=None,
                     help="Directory of per-run WH sidecar JSONs. "
                          "Files named N{N}_{dtype}.json (e.g. N1048576_fp32.json). "
                          "energy_J key used directly if present; "
                          "falls back to avg_power_W × duration_s if not.")
    pwr.add_argument("--cpu-energy-dir", default=None,
                     help="Directory of per-run CPU sidecar JSONs (same naming convention).")
    args = parser.parse_args()

    if args.wh_energy_dir:
        print(f"  WH energy sidecar dir : {args.wh_energy_dir}/N{{N}}_{{dtype}}.json")
    if args.cpu_energy_dir:
        print(f"  CPU energy sidecar dir: {args.cpu_energy_dir}/N{{N}}_{{dtype}}.json")
    if not args.wh_energy_dir and not args.cpu_energy_dir:
        print("  Energy: not measured (no --wh-energy-dir / --cpu-energy-dir supplied)."
              " All energy columns will be N/A.")

    dtypes = ["fp32", "bf16"] if args.dtype == "both" else [args.dtype]

    # Build N list
    any_sel = args.stockham or args.two_pass or args.three_pass or args.bluestein
    n_list: List[int] = []
    if not any_sel or args.stockham:   n_list += POW2_STOCKHAM
    if not any_sel or args.two_pass:   n_list += POW2_TWO_PASS
    if not any_sel or args.three_pass: n_list += POW2_THREE_PASS
    if not any_sel or args.bluestein:  n_list += BLUESTEIN_N
    n_list = sorted(set(n_list))

    # Single-N override — useful for generating one sidecar JSON per point
    if args.n is not None:
        n_list = [args.n]

    # Header
    print("=" * 80)
    print("FFT Benchmark — Tenstorrent Wormhole B0  (HPEC 2026)")
    print(f"  dtypes      : {dtypes}")
    print(f"  warmup/runs : {args.warmup} / {args.runs}")
    print(f"  WH sidecar  : {args.wh_energy_dir or 'not set (energy = N/A)'}")
    print(f"  CPU sidecar : {args.cpu_energy_dir or 'not set (energy = N/A)'}")
    print(f"  N sizes     : {len(n_list)} × {len(dtypes)} dtypes = "
          f"{len(n_list)*len(dtypes)} benchmarks")
    print(f"  CPU baseline: numpy_cpu (pocketfft, single-threaded)")
    print(f"  ⚠ CPU baseline ≠ Brown et al. native OpenMP C++ FFT")
    print("=" * 80)

    device = _open_device()
    results: List[BenchResult] = []

    try:
        for dtype_str in dtypes:
            print(f"\n── dtype = {dtype_str} ──")
            for N in n_list:
                r = benchmark_one(N, dtype_str, device,
                                  warmup=args.warmup, runs=args.runs,
                                  wh_energy_dir=args.wh_energy_dir,
                                  cpu_energy_dir=args.cpu_energy_dir)
                if r is not None:
                    results.append(r)
    finally:
        ttnn.close_device(device)

    if not results:
        print("No results collected.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    _print_summary_table(results)
    _print_per_algo_summary(results)

    # ── CSV ───────────────────────────────────────────────────────────────────
    if args.csv:
        fieldnames = [f.name for f in fields(BenchResult)]
        with open(args.csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.csv_row())
        print(f"\nCSV saved to: {args.csv}")
        print("  Columns with N/A = energy not measured for this run.")


if __name__ == "__main__":
    main()
