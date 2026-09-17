# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage Kimi's prebuilt weights on local disk before running a CI pytest command.

Run this inside the worker's mpirun command so the cache and pytest share a host.
Serial tensor loads from /mnt/models can exhaust the job budget before prefill.
The source cache is read-only; each invocation owns and removes its local copy.
"""

import argparse
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


def stage_cache(source, destination, workers):
    files = [(path, path.stat().st_size) for path in source.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"No prebuilt weights found in {source}")
    total_bytes = sum(size for _, size in files)
    destination.mkdir(parents=True)
    free_bytes = shutil.disk_usage(destination).free
    if free_bytes < total_bytes:
        raise OSError(f"Local cache needs {total_bytes} bytes; only {free_bytes} bytes free at {destination}")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 65), default=16)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Parent for the private cache (default: writable /scratch, otherwise the system temporary directory)",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    if not pytest_args:
        parser.error("pytest arguments are required after --")
    if CACHE_ENV not in os.environ:
        parser.error(f"{CACHE_ENV} must name the existing shared cache root")
    source = Path(os.environ[CACHE_ENV]) / CACHE_SUBDIR
    cache_dir = args.cache_dir
    if cache_dir is None:
        scratch = Path("/scratch")
        cache_dir = (
            scratch if scratch.is_dir() and os.access(scratch, os.W_OK | os.X_OK) else Path(tempfile.gettempdir())
        )

    # CI mounts the host's scratch disk separately from the container's writable
    # layer. Keep hundreds of GiB of staged tensors off that layer when possible.
    print(f"Using cache parent {cache_dir.resolve()}", flush=True)

    def terminate(signum, _frame):
        # A normal CI cancellation must also remove a cache on persistent scratch.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    with tempfile.TemporaryDirectory(prefix="kimi-prefill-", dir=cache_dir) as local_root:
        stage_cache(source, Path(local_root) / CACHE_SUBDIR, args.workers)
        env = dict(os.environ, **{CACHE_ENV: local_root})
        return subprocess.call([sys.executable, "-m", "pytest", *pytest_args], env=env)


if __name__ == "__main__":
    sys.exit(main())
