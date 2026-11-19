# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compact run provenance, independent of generated tensor data."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def loaded_native_binaries():
    """Identify mapped TT libraries without importing TTNN or opening a device."""
    result = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        address, _, _, device, inode, mapped_path = fields
        path = mapped_path.removesuffix(" (deleted)")
        if Path(path).name not in ("_ttnncpp.so", "_ttnn.so", "libtt_metal.so") or mapped_path in result:
            continue
        identity = {"path": path, "mapped_path": mapped_path, "inode": int(inode), "device": device}
        # Atomic replacement can leave this process mapped to an old inode.
        # Never label that old mapping with the replacement file's hash.
        for candidate in (Path(path), Path("/proc/self/map_files") / address):
            try:
                with candidate.open("rb") as stream:
                    stat = os.fstat(stream.fileno())
                    expected_device = tuple(int(part, 16) for part in device.split(":"))
                    if (stat.st_ino, os.major(stat.st_dev), os.minor(stat.st_dev)) != (
                        int(inode),
                        *expected_device,
                    ):
                        raise ValueError("file identity differs from the loaded mapping")
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        digest.update(chunk)
                    identity["sha256"] = digest.hexdigest()
                    break
            except (OSError, ValueError) as exc:
                error = str(exc)
        else:
            identity["sha256"] = None
            identity["hash_error"] = error
        result[mapped_path] = identity
    return result


def record(output):
    root = Path(__file__).parents[1]
    host_only = "--oracle-only" in sys.argv
    repo = root.parents[2]
    native_root = repo / "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device"
    native_sources = (
        native_root / "all_reduce_async_program_factory.cpp",
        native_root / "kernels/compute/reduction.cpp",
    )
    data = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("tt", "tests")
            for p in sorted((root / folder).glob("*.py"))
        },
        "native_source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in native_sources
        },
        "loaded_native_binaries": loaded_native_binaries(),
        "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "ttnn", "tt-perf-report")},
        "decoder": os.environ.get("QWEN_DECODER", "functional"),
        "execution": "host-only CPU" if host_only else "TTNN device",
        "device": None if host_only else "Blackhole P300c physical chip 0, MeshShape(1,1)",
        "regime": "serial",
        "env": {
            k: os.environ.get(k)
            for k in (
                "TT_VISIBLE_DEVICES",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT",
                "TT_METAL_PROFILER_NO_CACHE_OP_INFO",
                "TT_METAL_LOGS_PATH",
            )
        },
    }
    Path(str(output) + ".provenance.json").write_text(json.dumps(data, indent=2) + "\n")
