#!/usr/bin/env python3
"""Parallel perf sweep across BH devices.

Reads (label, env_overrides_dict) pairs from a JSON list on stdin and runs
each on a separate TT device, capping concurrency. Aggregates the
"Per-call avg" line from each subprocess's perf-trace output.

Usage:
    python _bench_runs/parallel_perf_sweep.py <<'JSON'
    [
      {"label": "baseline",  "env": {"PI0_VLM_MINIMAL_CFG": "8,8,8,4,2"}},
      {"label": "M4_sub18",  "env": {"PI0_VLM_MINIMAL_CFG": "4,8,8,1,8"}}
    ]
    JSON

Env defaults (the full chunk=1024 minimal-path set) are applied to all
runs; per-run `env` keys override.
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/tt-admin/sdawle/pi0/tt-metal")
LOG_DIR = ROOT / "_bench_runs" / f"perf_sweep_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
DENOISE_STEPS = os.environ.get("DENOISE_STEPS", "5")
NUM_DEVICES = 32

# Common env across all runs (the chunk=1024 minimal-path baseline)
BASE_ENV = {
    "PI0_EXPERT_MM_LOFI": "1",
    "PI0_ROPE_TABLES_L1": "1",
    "PI0_MM_SWEEP_V2": "1",
    "PI0_DENOISE_MM_TUNE": "1",
    "PI0_PREFILL_MM_TUNE": "1",
    "PI0_UPSTREAM_MASKS": "1",
    "QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT": "1",
    "QWEN_NLP_CREATE_HEADS_HEAD_SPLIT": "1",
    "PI0_NUM_CAMERAS": "3",
    "PI0_VLM_CHUNK_SIZE": "1024",
    "PI0_VLM_MLP_BF8_OUT": "1",
    "PI0_VLM_MLP_MINIMAL": "1",
    "PI05_NUM_DENOISE_STEPS": DENOISE_STEPS,
    "PI05_CHECKPOINT_DIR": "/home/tt-admin/pi05_cache/pi05_libero_upstream",
    "TT_METAL_CACHE": str(ROOT / ".tt_metal_cache"),
    "TT_METAL_HOME": str(ROOT),
    "PYTHONPATH": str(ROOT),
    "VIRTUAL_ENV": str(ROOT / "python_env"),
    "PATH": f"{ROOT}/python_env/bin:" + os.environ.get("PATH", ""),
}


def run_one(idx: int, label: str, overrides: dict) -> dict:
    device = idx % NUM_DEVICES
    log_path = LOG_DIR / f"{idx:02d}_{label}.log"
    env = dict(os.environ)
    env.update(BASE_ENV)
    env["TT_VISIBLE_DEVICES"] = str(device)
    env.update(overrides)
    cmd = [
        str(ROOT / "python_env/bin/python"),
        "-m",
        "pytest",
        "-xvs",
        "models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py",
    ]
    t0 = time.time()
    with open(log_path, "wb") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode
    dt = time.time() - t0

    ms = None
    fail = None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Per-call avg:\s+([0-9.]+) ms", line)
            if m:
                ms = float(m.group(1))
                break
        if ms is None:
            f.seek(0)
            content = f.read()
            for needle in ("TT_FATAL", "TT_THROW", "RuntimeError", "Error"):
                if needle in content:
                    fail = needle
                    break
    return {"idx": idx, "label": label, "device": device, "ms": ms, "fail": fail, "wall": dt, "log": str(log_path)}


def main():
    configs = json.load(sys.stdin)
    print(f"Running {len(configs)} configs, concurrency={CONCURRENCY}, logs → {LOG_DIR}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as exe:
        futs = [exe.submit(run_one, i, c["label"], c.get("env", {})) for i, c in enumerate(configs)]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            if r["ms"] is not None:
                print(f"  {r['label']:<24} (dev={r['device']:2d})  →  {r['ms']:.2f} ms  ({r['wall']:.0f}s)")
            else:
                print(f"  {r['label']:<24} (dev={r['device']:2d})  →  ❌ {r['fail']}  — {r['log']}")

    print()
    print("=" * 60)
    print("  LEADERBOARD")
    print("=" * 60)
    ok = [r for r in results if r["ms"] is not None]
    fail = [r for r in results if r["ms"] is None]
    for r in sorted(ok, key=lambda r: r["ms"]):
        print(f"  {r['ms']:6.2f} ms   {r['label']}")
    for r in fail:
        print(f"  ❌          {r['label']}  ({r['fail']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
