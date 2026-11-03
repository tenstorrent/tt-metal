# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized stage-5 experiments with immutable commands and source snapshots."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_multichip_decoder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", type=Path)
    args = ap.parse_args()
    for case in json.loads(args.cases.read_text()):
        tag, layer, seq = case["tag"], case.get("layer", 0), case.get("seq", 32)
        key = f"{tag}_l{layer}_b{case.get('batch', 1)}_s{seq}"
        cmd = [
            sys.executable,
            "-m",
            "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.run_multichip_decoder",
            "--layer",
            str(layer),
            "--seq",
            str(seq),
            "--context",
            str(case.get("context", max(1024, seq * 2))),
            "--batch",
            str(case.get("batch", 1)),
            "--trace",
            "--iterations",
            "200",
            "--prefill-iterations",
            "20",
            "--artifact-dir",
            str(DOC),
            "--tag",
            tag,
            "--options",
            json.dumps(case.get("options", {})),
        ]
        if case.get("chunk_size"):
            cmd += ["--chunk-size", str(case["chunk_size"])]
        if case.get("trace_prefill"):
            cmd += ["--trace-prefill"]
        if case.get("baseline"):
            cmd += ["--baseline"]
        if case.get("input_variant"):
            cmd += ["--input-variant", str(case["input_variant"])]
        if case.get("source"):
            cmd += ["--source", str(ROOT / "tests" / case["source"])]
        log = DOC / "logs" / (key + ".log")
        env = {
            **os.environ,
            "TT_METAL_TRACE_ALLOC_TRACKING": str(case.get("tracking", 0)),
            "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
        }
        for name in ["TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_LOGS_PATH"]:
            env.pop(name, None)
        record = dict(
            command=cmd,
            environment={k: v for k, v in env.items() if k.startswith("TT_")},
            log=str(log),
            start=time.time(),
        )
        source_paths = [
            ROOT / "tt/multichip_decoder.py",
            ROOT / "tt/optimized_decoder.py",
            ROOT / "tests/run_multichip_decoder.py",
        ]
        if case.get("source"):
            source_paths.append(ROOT / "tests" / case["source"])
        record["sources_at_start"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        for idx, source_path in enumerate(source_paths):
            (DOC / "candidates" / f"{key}_start{idx}.py.txt").write_bytes(source_path.read_bytes())
        target = DOC / (key + "_execution.json")
        target.write_text(json.dumps(record, indent=2) + "\n")
        print("START", key, flush=True)
        with log.open("w") as output:
            proc = subprocess.Popen(cmd, env=env, stdout=output, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                print("TRIAGE_REQUIRED", proc.pid, log, flush=True)
                raise
        record.update(exit_code=code, elapsed_seconds=time.time() - record["start"])
        target.write_text(json.dumps(record, indent=2) + "\n")
        if code:
            print("FAILED", key, log, flush=True)
            sys.exit(code)
        result = json.loads((DOC / (key + ".json")).read_text())
        print("PASS", key, result["prefill_ms"], result["decode_ms"], result.get("pcc", "reference"), flush=True)


if __name__ == "__main__":
    main()
