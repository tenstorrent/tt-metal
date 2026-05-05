#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark for DIT model weight loading from shared storage (NFS).

Measures sequential vs threaded I/O throughput of .tensorbin files, with
optional device transfer. Use this to quantify the impact of prefetch-based
loading before deploying on multi-host systems.

Examples:

  # 1) Host-only benchmark against real cached weights on NFS:
  python benchmark_loading.py --cache-dir /mnt/nfs/cache/mochi-1-preview/transformer/SP2_0_TP4_1_mesh2x4_bf16

  # 2) With device transfer (requires TT devices):
  python benchmark_loading.py --cache-dir /path/to/cache --mesh-shape 2 4

  # 3) Full comparison (sequential vs threaded, host vs device):
  python benchmark_loading.py --cache-dir /path/to/cache --mesh-shape 2 4 --compare

  # 4) Generate synthetic weights on NFS and benchmark:
  python benchmark_loading.py --synthetic --output-dir /mnt/nfs/synthetic --num-files 300 --total-size-gb 20

  # 5) Cold-cache measurement (drop page cache between runs, Linux + sudo):
  python benchmark_loading.py --cache-dir /path/to/cache --drop-caches --compare
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

import ttnn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_tensorbin_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob("*.tensorbin"))
    if not files:
        print(f"ERROR: No .tensorbin files found in {directory}", file=sys.stderr)
        sys.exit(1)
    return files


def total_bytes(files: list[Path]) -> int:
    return sum(f.stat().st_size for f in files)


def fmt(label: str, nbytes: int, elapsed: float) -> str:
    gb = nbytes / 1024**3
    gbps = gb / elapsed if elapsed > 0 else float("inf")
    return f"  {label:40s}  {elapsed:7.2f}s  {gb:6.2f} GB  {gbps:5.2f} GB/s"


def drop_page_cache() -> None:
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
            capture_output=True,
        )
        time.sleep(0.5)
    except Exception:
        print("  (could not drop page cache — needs sudo on Linux)")


def deallocate_tensors(tensors: list) -> None:
    for t in tensors:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    tensors.clear()
    gc.collect()


# ---------------------------------------------------------------------------
# Benchmark kernels
# ---------------------------------------------------------------------------


def bench_sequential_host(files: list[Path]) -> float:
    """Load all files to host memory sequentially."""
    start = time.perf_counter()
    tensors = [ttnn.load_tensor(str(f), device=None) for f in files]
    elapsed = time.perf_counter() - start
    del tensors
    gc.collect()
    return elapsed


def bench_threaded_host(files: list[Path], num_threads: int) -> float:
    """Load all files to host memory using a thread pool."""

    def load(path: str):
        return ttnn.load_tensor(path, device=None)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        tensors = list(pool.map(load, [str(f) for f in files]))
    elapsed = time.perf_counter() - start
    del tensors
    gc.collect()
    return elapsed


def bench_sequential_device(files: list[Path], device: ttnn.MeshDevice) -> float:
    """Load all files directly to device sequentially (current production path)."""
    tensors = []
    start = time.perf_counter()
    for f in files:
        tensors.append(ttnn.load_tensor(str(f), device=device))
    elapsed = time.perf_counter() - start
    deallocate_tensors(tensors)
    return elapsed


def bench_prefetch_device(files: list[Path], device: ttnn.MeshDevice, num_threads: int) -> float:
    """Prefetch files to host in parallel, then transfer to device sequentially."""

    def load_host(path: str):
        return ttnn.load_tensor(path, device=None)

    # Phase 1: threaded NFS reads to host
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        host_tensors = list(pool.map(load_host, [str(f) for f in files]))

    # Phase 2: sequential host-to-device transfer
    device_tensors = []
    for ht in host_tensors:
        device_tensors.append(ttnn.to_device(ht, device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    del host_tensors

    elapsed = time.perf_counter() - start
    deallocate_tensors(device_tensors)
    return elapsed


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_cache(output_dir: Path, num_files: int, total_size_gb: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    bytes_per_file = int(total_size_gb * 1024**3) // num_files
    elements = bytes_per_file // 2  # bf16 = 2 bytes
    dim = max(32, int(elements**0.5) // 32 * 32)

    print(f"Generating {num_files} synthetic .tensorbin files in {output_dir}")
    print(f"  Target: {total_size_gb:.1f} GB total, ~{bytes_per_file / 1024**2:.1f} MB/file, shape ({dim}, {dim})")

    for i in range(num_files):
        path = output_dir / f"param_{i:04d}.tensorbin"
        if path.exists():
            continue
        t = torch.randn(dim, dim, dtype=torch.bfloat16)
        tt = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(str(path), tt)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{num_files}")

    print("  Done.")


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------


def run_single(label: str, bench_fn, files, nbytes, drop: bool) -> float:
    if drop:
        drop_page_cache()
    elapsed = bench_fn(files)
    print(fmt(label, nbytes, elapsed))
    return elapsed


def run_benchmark(args: argparse.Namespace, files: list[Path]) -> None:
    nbytes = total_bytes(files)
    gb = nbytes / 1024**3
    thread_counts = [int(x) for x in args.threads.split(",")]

    print(f"\n{'=' * 80}")
    print(f"  {len(files)} files, {gb:.2f} GB total")
    print(f"{'=' * 80}")

    # --- File size distribution ---
    sizes = sorted(f.stat().st_size for f in files)
    print(f"\n  File sizes:  min={sizes[0]/1024**2:.2f} MB  "
          f"median={sizes[len(sizes)//2]/1024**2:.2f} MB  "
          f"max={sizes[-1]/1024**2:.2f} MB  "
          f"<1MB: {sum(1 for s in sizes if s < 1024**2)}/{len(sizes)}")

    device = None
    if args.mesh_shape:
        print(f"\n  Opening mesh device {args.mesh_shape} ...")
        device = ttnn.open_mesh_device(ttnn.MeshShape(*args.mesh_shape))
        print("  Device ready.")

    # ---- Warmup ----
    if not args.no_warmup:
        print("\n  Warmup (1 sequential host pass) ...")
        bench_sequential_host(files)

    results: dict[str, list[float]] = {}

    def record(label, bench_fn, rounds):
        times = []
        for r in range(rounds):
            if args.drop_caches:
                drop_page_cache()
            elapsed = bench_fn()
            times.append(elapsed)
            gbps = gb / elapsed if elapsed > 0 else 0
            print(f"    round {r + 1}: {elapsed:7.2f}s  ({gbps:.2f} GB/s)")
        results[label] = times

    rounds = args.rounds

    # ---- Host-only benchmarks ----
    print(f"\n--- Sequential host load ---  [{rounds} rounds]")
    record("sequential_host", lambda: bench_sequential_host(files), rounds)

    if args.compare:
        for nt in thread_counts:
            print(f"\n--- Threaded host load ({nt} threads) ---  [{rounds} rounds]")
            record(f"threaded_host_{nt}t", lambda _nt=nt: bench_threaded_host(files, _nt), rounds)

    # ---- Device benchmarks ----
    if device is not None:
        print(f"\n--- Sequential device load ---  [{rounds} rounds]")
        record("sequential_device", lambda: bench_sequential_device(files, device), rounds)

        if args.compare:
            for nt in thread_counts:
                print(f"\n--- Prefetch + device ({nt} threads) ---  [{rounds} rounds]")
                record(
                    f"prefetch_device_{nt}t",
                    lambda _nt=nt: bench_prefetch_device(files, device, _nt),
                    rounds,
                )

    # ---- Summary ----
    print(f"\n{'=' * 80}")
    print("  SUMMARY (median of rounds)")
    print(f"{'=' * 80}")
    for label, times in results.items():
        median = sorted(times)[len(times) // 2]
        gbps = gb / median if median > 0 else 0
        print(f"  {label:40s}  {median:7.2f}s  {gbps:5.2f} GB/s")

    if "sequential_host" in results:
        baseline = sorted(results["sequential_host"])[len(results["sequential_host"]) // 2]
        print(f"\n  Speedup vs sequential_host:")
        for label, times in results.items():
            if label == "sequential_host":
                continue
            median = sorted(times)[len(times) // 2]
            speedup = baseline / median if median > 0 else 0
            print(f"    {label:40s}  {speedup:.2f}x")

    if device is not None:
        ttnn.close_mesh_device(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark .tensorbin loading (sequential vs threaded, host vs device)"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache-dir", type=Path, help="Directory containing .tensorbin files")
    src.add_argument("--synthetic", action="store_true", help="Generate synthetic .tensorbin files first")

    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/synthetic_bench"))
    parser.add_argument("--num-files", type=int, default=300)
    parser.add_argument("--total-size-gb", type=float, default=20.0)

    parser.add_argument("--mesh-shape", type=int, nargs=2, metavar=("ROWS", "COLS"), help="Device mesh shape (e.g. 2 4)")
    parser.add_argument("--threads", default="2,4,8,16", help="Comma-separated thread counts (default: 2,4,8,16)")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per benchmark (default: 3)")
    parser.add_argument("--compare", action="store_true", help="Run all modes (sequential + threaded, host + device)")
    parser.add_argument("--drop-caches", action="store_true", help="Drop OS page cache between rounds (sudo, Linux)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup pass")

    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_cache(args.output_dir, args.num_files, args.total_size_gb)
        cache_dir = args.output_dir
    else:
        cache_dir = args.cache_dir

    files = find_tensorbin_files(cache_dir)
    run_benchmark(args, files)


if __name__ == "__main__":
    main()
