# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage Kimi's prebuilt weights on local disk before running a CI pytest command.

Run this inside the worker's mpirun command so the cache and pytest share a host.
Serial tensor loads from /mnt/models can exhaust the job budget before prefill.
The source cache is read-only; each invocation owns and removes its local copy.
Provision /scratch or set TT_KIMI_PREFILL_CACHE_DIR to a worker-local volume.
Run --check-cache during setup to check the cache size plus 10 GiB of headroom.
Copies run in a separate process so cancellation stops active I/O before cleanup.
"""

import argparse
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CACHE_ENV = "TT_KIMI_PREFILL_TTNN_CACHE"
CACHE_SUBDIR = Path("kimi_k2_7_bh_32dev/8x4")
CACHE_DIR_ENV = "TT_KIMI_PREFILL_CACHE_DIR"
CACHE_HEADROOM = 10 * 2**30


def check_cache(source, cache_dir):
    files = [(path, path.stat().st_size) for path in source.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"No prebuilt weights found in {source}")
    total_bytes = sum(size for _, size in files)
    required = total_bytes + CACHE_HEADROOM
    hint = (
        f"Provision a worker-local volume with at least {required} bytes "
        f"({required / 2**30:.1f} GiB) free and select it with --cache-dir or {CACHE_DIR_ENV}. "
        "Blaze CI uses the KIMI_PREFILL_CACHE_DIR repository variable (default /scratch). "
        "No temporary-directory fallback is used."
    )
    if not cache_dir.is_dir() or not os.access(cache_dir, os.W_OK | os.X_OK):
        raise OSError(f"Cache setup: {cache_dir} is not a writable directory. {hint}")
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < required:
        raise OSError(f"Cache setup: only {free_bytes} bytes free at {cache_dir}. {hint}")
    print(
        f"Cache setup OK: {cache_dir}, {free_bytes / 2**30:.1f} GiB free, {required / 2**30:.1f} GiB required",
        flush=True,
    )
    return files


def stage_cache(source, destination, workers, files):
    total_bytes = sum(size for _, size in files)
    destination.mkdir(parents=True)
    print(f"Staging {len(files)} files ({total_bytes / 2**30:.1f} GiB) from {source} to {destination}", flush=True)

    def copy_file(entry):
        path, size = entry
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if target.stat().st_size != size:
            raise OSError(f"Incomplete cache copy: {path}")
        return size

    started = last_report = time.monotonic()
    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_file, entry) for entry in files]
        try:
            for count, future in enumerate(as_completed(futures), 1):
                copied_bytes += future.result()
                now = time.monotonic()
                if now - last_report >= 30 or count == len(files):
                    print(
                        f"Staged {count}/{len(files)} files, {copied_bytes / 2**30:.1f} GiB in {now - started:.1f}s",
                        flush=True,
                    )
                    last_report = now
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def run_staging(source, destination, workers, files):
    # Keep blocking filesystem I/O out of the parent: killing this process also
    # stops its copy threads before the parent removes the private directory.
    process = multiprocessing.get_context("spawn").Process(
        target=stage_cache, args=(source, destination, workers, files)
    )
    try:
        process.start()
        process.join()
        if process.exitcode != 0:
            raise OSError(f"Cache staging failed with exit code {process.exitcode}")
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 65), default=16)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get(CACHE_DIR_ENV, "/scratch")),
        help=f"Provisioned worker-local cache parent (default: {CACHE_DIR_ENV}, otherwise /scratch)",
    )
    parser.add_argument(
        "--check-cache", action="store_true", help="Validate cache capacity without copying or running pytest"
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    if not pytest_args and not args.check_cache:
        parser.error("pytest arguments are required after --")
    if CACHE_ENV not in os.environ:
        parser.error(f"{CACHE_ENV} must name the existing shared cache root")
    source = Path(os.environ[CACHE_ENV]) / CACHE_SUBDIR
    try:
        files = check_cache(source, args.cache_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.check_cache:
        return 0

    def terminate(signum, _frame):
        # A normal CI cancellation must also remove a cache on persistent scratch.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    with tempfile.TemporaryDirectory(prefix="kimi-prefill-", dir=args.cache_dir) as local_root:
        run_staging(source, Path(local_root) / CACHE_SUBDIR, args.workers, files)
        env = dict(os.environ, **{CACHE_ENV: local_root})
        with subprocess.Popen([sys.executable, "-m", "pytest", *pytest_args], env=env) as child:
            try:
                return child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()


if __name__ == "__main__":
    sys.exit(main())
