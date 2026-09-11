# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare PCH modes with fresh JIT caches and kernel ccache disabled."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

GIB = 1024**3
OUTPUT = Path("generated/pch-diagnosis")


def snapshot(root, phase):
    stats = {
        "phase": phase,
        "time": time.time(),
        "free_bytes": {p: shutil.disk_usage(p).free for p in ["/work", "/github/home", "/tmp"]},
        "free_inodes": {p: os.statvfs(p).f_favail for p in ["/work", "/github/home", "/tmp"]},
        "cache_bytes": 0,
        "pch_bytes": 0,
        "gch_count": 0,
        "gch_bytes": 0,
        "by_target": {},
    }
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            p = Path(directory) / filename
            try:
                size = p.stat().st_blocks * 512
            except FileNotFoundError:
                continue
            stats["cache_bytes"] += size
            if "pch" in p.parts:
                stats["pch_bytes"] += size
                if filename.endswith(".gch"):
                    stats["gch_count"] += 1
                    stats["gch_bytes"] += size
                    target = p.parts[p.parts.index("pch") + 1]
                    entry = stats["by_target"].setdefault(target, {"count": 0, "bytes": 0})
                    entry["count"] += 1
                    entry["bytes"] += size
    line = json.dumps(stats, sort_keys=True)
    print("PCH_DISK_SAMPLE " + line, flush=True)
    with (OUTPUT / "disk.jsonl").open("a") as stream:
        stream.write(line + "\n")
    return stats


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def save_cache_details(root, phase):
    files = []
    for directory, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(directory) / name
            try:
                info = p.stat()
            except FileNotFoundError:
                continue
            if "pch" in p.parts or name.endswith(".log"):
                files.append((info.st_size, str(p)))
    (OUTPUT / f"{phase}-cache-files.json").write_text(json.dumps(sorted(files, reverse=True), indent=2))
    logs = sorted(root.rglob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
    with (OUTPUT / f"{phase}-compiler-log-tails.txt").open("w") as stream:
        for path in logs:
            with path.open("rb") as source:
                source.seek(max(0, path.stat().st_size - 16384))
                tail = source.read().decode(errors="replace")
            stream.write(f"\n--- {path} ---\n{tail}\n")
            if any(word in tail.lower() for word in ("error:", "no space", "fatal")):
                print(f"COMPILER_DIAGNOSTIC {path}\n{tail}", flush=True)


def run_phase(phase, target, pch):
    parent = Path("/github/home/pch-diagnosis")
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{phase}-pch{pch}-", dir=parent))
    env = os.environ.copy()
    # JIT checks presence, so setting this to "0" would still enable ccache.
    env.pop("TT_METAL_CCACHE_KERNEL_SUPPORT", None)
    env["CCACHE_DISABLE"] = "1"
    env["TT_METAL_CACHE"] = str(root)
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OMP_WAIT_POLICY="passive")
    env.pop("TT_METAL_JIT_PCH_STRICT", None)
    if pch == "1":
        env["TT_METAL_JIT_PCH"] = "1"
    else:
        env.pop("TT_METAL_JIT_PCH", None)
    config = {
        "phase": phase,
        "pch": pch,
        "runner": env.get("RUNNER_NAME"),
        "test_commit": env.get("GITHUB_SHA"),
        "environment": {
            key: env.get(key)
            for key in ("TT_METAL_CACHE", "TT_METAL_JIT_PCH", "TT_METAL_CCACHE_KERNEL_SUPPORT", "CCACHE_DISABLE")
        },
    }
    print("PCH_AB_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    (OUTPUT / f"{phase}-config.json").write_text(json.dumps(config, indent=2))
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--timeout",
        "300",
        target,
        "-xv",
        "-m",
        "not disable_fast_runtime_mode",
        f"--junitxml={OUTPUT}/{phase}.xml",
    ]
    print(f"PCH_PHASE_START phase={phase} pch={pch} command={command}", flush=True)
    first = snapshot(root, phase)
    assert first["cache_bytes"] == 0, "Each A/B pass must start with an empty JIT cache"
    if min(first["free_bytes"].values()) < 3 * GIB:
        print("PCH_DISK_GUARD: insufficient free space before tests", flush=True)
        return 86
    started = time.monotonic()
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    rc = None
    try:
        while proc.poll() is None:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            current = snapshot(root, phase)
            if min(current["free_bytes"].values()) < 3 * GIB:
                print("PCH_DISK_GUARD: stopping tests with less than 3 GiB free to preserve diagnostics", flush=True)
                rc = 86
                stop(proc)
        if rc is None:
            rc = proc.returncode
    finally:
        stop(proc)
        final = snapshot(root, phase)
        save_cache_details(root, phase)
    result = {
        **config,
        "exit_code": rc,
        "elapsed_seconds": time.monotonic() - started,
        "cache": final,
    }
    (OUTPUT / f"{phase}-result.json").write_text(json.dumps(result, indent=2))
    print(f"PCH_PHASE_END phase={phase} exit_code={rc}", flush=True)
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pch", choices=["0", "1"], required=True)
    parser.add_argument("--phase", choices=["isolated", "suite"], required=True)
    parser.add_argument("--label", choices=["first", "second"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(["df", "-h", "/", "/work", "/github/home", "/tmp"], check=True)
    target = "tests/ttnn/unit_tests/operations/pool"
    if args.phase == "isolated":
        target += "/test_upsample.py::test_nearest_upsample_with_uneven_input_shards"
    return run_phase(args.label or args.phase, target, args.pch)


if __name__ == "__main__":
    sys.exit(main())
