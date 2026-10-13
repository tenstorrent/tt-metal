# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Bound the speedup a disk-to-device (peer-to-peer) weight load could give over today's ttnn.load_tensor path.

For a set of .tensorbin files, measures three times:

  direct  Sequential O_DIRECT read of the files into host memory: what the disk can deliver, bypassing the page cache.
  cold    ttnn.load_tensor(..., device=mesh) with the files evicted from the page cache: today's path from disk.
  warm    The same load with the files resident in the page cache: today's path without the disk.

A peer-to-peer transfer still has to read the disk and cross the device's PCIe link, so its best case is about
max(direct, warm). The same bound is what a well-pipelined (chunked, overlapped) load through host memory would reach.

When the build has it, the experimental chunked O_DIRECT loader (load_tensor_flatbuffer_direct) is timed too, with
the files cold and cached. It bypasses the page cache, so both runs read from disk. CPU time is recorded per phase.

Usage:
  # Existing weight cache (e.g. a model's tensor cache directory):
  python benchmark_weight_load_path.py --files /path/to/cache --mesh-shape 1 8

  # Synthetic weight, written to a directory on the disk under test:
  python benchmark_weight_load_path.py --generate-gib 4 --workdir /nvme/scratch

Eviction uses posix_fadvise(DONTNEED), which needs no root but only drops clean pages that no process has mapped.
Pass --drop-caches (root) to drop the whole page cache instead. Residency is checked with mincore() and reported.
"""

import argparse
import ctypes
import json
import mmap
import os
import pathlib
import statistics
import sys
import time

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
PIN_WRITE_THRESHOLD_BYTES = 32 * 1024 * 1024
GIB = 1024**3

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--files", nargs="+", type=pathlib.Path, help=".tensorbin files or directories containing them")
    source.add_argument("--generate-gib", type=float, help="write one synthetic bfloat16 weight of this size")
    parser.add_argument("--workdir", type=pathlib.Path, help="directory for the synthetic weight (on the disk to test)")
    parser.add_argument("--mesh-shape", nargs=2, type=int, metavar=("ROWS", "COLS"), help="defaults to the full system")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--direct-chunk-mib", type=int, default=64)
    parser.add_argument("--drop-caches", action="store_true", help="drop the whole page cache (requires root)")
    parser.add_argument("--json", type=pathlib.Path, help="also write results to this file")
    args = parser.parse_args()
    if args.generate_gib is not None and args.workdir is None:
        parser.error("--generate-gib requires --workdir")
    return args


def collect_files(paths):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.tensorbin")))
        else:
            files.append(path)
    if not files:
        raise SystemExit("No .tensorbin files found")
    return files


def generate_weight(ttnn, mesh_device, workdir, size_gib):
    import torch

    num_devices = mesh_device.get_num_devices()
    width = 32 * 1024 * num_devices
    height = max(32, int(size_gib * GIB / (width * 2)) // 32 * 32)
    path = workdir / f"synthetic_{height}x{width}_bf16.tensorbin"
    if not path.exists():
        workdir.mkdir(parents=True, exist_ok=True)
        print(f"Writing {height * width * 2 / GIB:.2f} GiB synthetic weight to {path}")
        tensor = ttnn.from_torch(
            torch.randn(height, width, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        ttnn.dump_tensor(path, tensor)
        with open(path, "rb") as f:
            os.fsync(f.fileno())
    return [path]


def evict(files, drop_caches):
    if drop_caches:
        os.sync()
        pathlib.Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def resident_fraction(files):
    resident_pages = 0
    total_pages = 0
    for path in files:
        size = path.stat().st_size
        if size == 0:
            continue
        num_pages = (size + PAGE_SIZE - 1) // PAGE_SIZE
        vec = (ctypes.c_ubyte * num_pages)()
        with open(path, "rb") as f:
            mapping = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_COPY)
        try:
            anchor = ctypes.c_char.from_buffer(mapping)
            status = _libc.mincore(ctypes.addressof(anchor), size, vec)
            del anchor
        finally:
            mapping.close()
        if status != 0:
            raise OSError(ctypes.get_errno(), f"mincore failed for {path}")
        resident_pages += num_pages - bytes(vec).count(0)
        total_pages += num_pages
    return resident_pages / total_pages if total_pages else 0.0


def timed(fn):
    """Returns (wall seconds, CPU seconds of this process across all threads, fn's result)."""
    cpu_start, wall_start = os.times(), time.perf_counter()
    result = fn()
    cpu_end, wall_end = os.times(), time.perf_counter()
    cpu = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
    return wall_end - wall_start, cpu, result


def direct_read(files, chunk_bytes):
    buffer = mmap.mmap(-1, chunk_bytes)
    view = memoryview(buffer)
    total_bytes = 0
    for path in files:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            offset = 0
            while True:
                read = os.preadv(fd, [view], offset)
                total_bytes += read
                offset += read
                if read < chunk_bytes:
                    break
        finally:
            os.close(fd)
    view.release()
    buffer.close()
    return total_bytes


def load_all(ttnn, mesh_device, files, loader):
    for path in files:
        ttnn.deallocate(loader(path, mesh_device))
    ttnn.synchronize_device(mesh_device)


def main():
    args = parse_args()
    if sys.platform != "linux":
        raise SystemExit("This benchmark needs Linux (O_DIRECT, posix_fadvise, mincore)")

    import ttnn

    mesh_shape = ttnn.MeshShape(*args.mesh_shape) if args.mesh_shape else None
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    try:
        if args.generate_gib is not None:
            files = generate_weight(ttnn, mesh_device, args.workdir, args.generate_gib)
        else:
            files = collect_files(args.files)

        num_devices = mesh_device.get_num_devices()
        sizes = [path.stat().st_size for path in files]
        total_bytes = sum(sizes)
        # The pinned path is chosen by the total bytes a load writes from this host, not by each device's share.
        unpinned_bytes = sum(size for size in sizes if size <= PIN_WRITE_THRESHOLD_BYTES)
        print(f"{len(files)} files, {total_bytes / GIB:.2f} GiB, mesh of {num_devices} devices")

        loaders = {"mmap": lambda path, device: ttnn.load_tensor(path, device=device)}
        if hasattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer_direct"):
            loaders["chunked"] = lambda path, device: ttnn._ttnn.tensor.load_tensor_flatbuffer_direct(str(path), device)

        # Untimed load so one-time setup (allocator, pinning, dispatch) is not charged to the first cold run.
        for loader in loaders.values():
            load_all(ttnn, mesh_device, files[:1], loader)

        phases = ["direct", "cold", "warm"] + (["chunked_cold", "chunked_warm"] if "chunked" in loaders else [])
        runs = {phase: [] for phase in phases}
        cpu = {phase: [] for phase in phases}
        residency = {phase: [] for phase in phases if phase != "direct"}

        def run(phase, fn):
            if phase in residency:
                residency[phase].append(resident_fraction(files))
            wall, cpu_seconds, result = timed(fn)
            runs[phase].append(wall)
            cpu[phase].append(cpu_seconds)
            return result

        for repeat in range(args.repeats):
            read_bytes = run("direct", lambda: direct_read(files, args.direct_chunk_mib * 1024 * 1024))
            if read_bytes != total_bytes:
                raise SystemExit(f"Direct read returned {read_bytes} bytes, expected {total_bytes}")

            evict(files, args.drop_caches)
            run("cold", lambda: load_all(ttnn, mesh_device, files, loaders["mmap"]))
            run("warm", lambda: load_all(ttnn, mesh_device, files, loaders["mmap"]))
            if "chunked" in loaders:
                run("chunked_warm", lambda: load_all(ttnn, mesh_device, files, loaders["chunked"]))
                evict(files, args.drop_caches)
                run("chunked_cold", lambda: load_all(ttnn, mesh_device, files, loaders["chunked"]))
            print(f"  run {repeat + 1}: " + ", ".join(f"{phase} {runs[phase][-1]:.2f}s" for phase in phases))
    finally:
        ttnn.close_mesh_device(mesh_device)

    median = {phase: statistics.median(runs[phase]) for phase in phases}
    median_cpu = {phase: statistics.median(cpu[phase]) for phase in phases}
    direct, cold, warm = median["direct"], median["cold"], median["warm"]
    best_case = max(direct, warm)

    def rate(seconds):
        return total_bytes / seconds / 1e9

    results = {
        "files": len(files),
        "total_bytes": total_bytes,
        "num_devices": num_devices,
        "bytes_below_pin_threshold_fraction": unpinned_bytes / total_bytes,
        "median_seconds": median,
        "median_gb_per_s": {phase: rate(seconds) for phase, seconds in median.items()},
        "median_cpu_seconds": median_cpu,
        "cpu_seconds_per_gb": {phase: seconds / (total_bytes / 1e9) for phase, seconds in median_cpu.items()},
        "max_resident_fraction_before_cold": max(residency["cold"]),
        "min_resident_fraction_before_warm": min(residency["warm"]),
        "p2p_best_case_seconds": best_case,
        "cold_speedup_ceiling": cold / best_case,
        "warm_p2p_vs_today": warm / direct,
        "cold_vs_serial_model": cold / (direct + warm),
        "raw_seconds": runs,
        "raw_cpu_seconds": cpu,
    }
    if "chunked" in loaders:
        results["max_resident_fraction_before_chunked_cold"] = max(residency["chunked_cold"])

    labels = {
        "direct": "Disk, O_DIRECT read",
        "cold": "Today, cold cache",
        "warm": "Today, warm cache",
        "chunked_cold": "Chunked, cold cache",
        "chunked_warm": "Chunked, warm cache",
    }
    print()
    print(f"{'':22}{'wall':>9}  {'rate':>10}  {'CPU':>8}  {'CPU-s/GB':>8}")
    for phase in phases:
        print(
            f"{labels[phase]:22}{median[phase]:8.2f}s  {rate(median[phase]):5.2f} GB/s  "
            f"{median_cpu[phase]:7.2f}s  {results['cpu_seconds_per_gb'][phase]:8.3f}"
        )
    print()
    print(f"Best case without the RAM bounce: {best_case:.2f}s")
    print(f"  Ceiling on cold-load speedup:   {cold / best_case:.2f}x")
    print(f"  Warm-cache load, p2p vs today:  {warm / direct:.2f}x (below 1 means p2p would be slower)")
    print(
        f"Cold load vs direct + warm:       {cold / (direct + warm):.2f}x (well above 1: the page-fault read is slow)"
    )
    if "chunked" in loaders:
        print(f"Chunked vs today, cold cache:     {cold / median['chunked_cold']:.2f}x faster")
    cold_residency = max(
        results["max_resident_fraction_before_cold"], results.get("max_resident_fraction_before_chunked_cold", 0)
    )
    if cold_residency > 0.05:
        print(
            f"WARNING: up to {cold_residency:.0%} of the data was still cached before a cold run, so cold times are "
            "optimistic. Try --drop-caches."
        )
    if results["bytes_below_pin_threshold_fraction"] > 0:
        print(
            f"NOTE: {results['bytes_below_pin_threshold_fraction']:.0%} of the bytes are in files of at most 32 MiB; "
            "today's loader writes those through the command-queue copy path rather than the pinned path."
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
