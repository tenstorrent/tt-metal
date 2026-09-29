"""Serialized stage-owned correctness commands, stopping on the first failure."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"
MODULE = "models.autoports.ifm_k2_horizon_7b.tests."


def main():
    (DOC / "logs").mkdir(parents=True, exist_ok=True)
    cases = [
        (
            "boundaries_final",
            "run_fused",
            ["--lengths", "1", "31", "32", "33", "127", "128", "129", "255", "256", "257", "777", "17", "--audit"],
        ),
        ("unchunked", "run_fused", ["--lengths", "257", "--unchunked-control", "--audit"]),
        ("positions", "run_fused_positions", []),
        ("positions_vector", "run_fused_positions", ["--batch", "32"]),
        ("positions_accurate", "run_fused_positions", ["--pages", "1025"]),
        ("positions_accurate_vector", "run_fused_positions", ["--pages", "1025", "--batch", "4"]),
        ("batch32", "run_fused", ["--lengths", "33", "--batch", "32", "--decode-steps", "2", "--audit"]),
        ("stock_batches_final", "probe_stock_batches", []),
        ("stock_boundary", "run_fused", ["--lengths", "32766", "--decode-steps", "2", "--audit"]),
        ("accurate_boundary", "run_fused", ["--lengths", "32767", "--decode-steps", "2", "--audit"]),
        ("long_tails", "probe_fused_long_layer", ["--tails", "--continuation"]),
        ("long_context", "run_fused_long_context", []),
        (
            "synthetic",
            "pytest",
            [
                "-q",
                str(ROOT / "models/autoports/ifm_k2_horizon_7b/tests/test_fused_decoder.py"),
                "--basetemp=" + str(DOC / "pytest_tmp"),
            ],
        ),
        ("watcher", "run_fused", ["--lengths", "777", "--batch", "2", "--split", "31", "--remap", "--audit"]),
        ("watcher_b1", "run_fused", ["--lengths", "4096", "--decode-steps", "3", "--audit"]),
        ("watcher_vector", "run_fused_positions", ["--batch", "4", "--pages", "1025"]),
    ]
    runtime = ROOT / "models/autoports/ifm_k2_horizon_7b/tt/fused_decoder.py"
    result = {"completed": False, "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(), "commands": []}
    for name, module, args in cases:
        env = os.environ.copy()
        for k in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_DPRINT_CORES"):
            env.pop(k, None)
        if name.startswith("watcher"):
            env["TT_METAL_WATCHER"] = "10"
            env["TT_METAL_LOGS_PATH"] = str(DOC / "watcher" / name)
        if module != "pytest":
            args += ["--output", str(DOC / (name + ".json"))]
        cmd = [sys.executable, "-m", module if module == "pytest" else MODULE + module, *args]
        log = DOC / "logs" / (name + ".log")
        start = time.time()
        print("START", name, flush=True)
        with log.open("w") as f:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        result["commands"].append(
            dict(name=name, argv=cmd, exit_code=proc.returncode, wall_seconds=time.time() - start, log=str(log))
        )
        (DOC / "acceptance_commands.json").write_text(json.dumps(result, indent=2) + "\n")
        print("END", name, proc.returncode, flush=True)
        if proc.returncode:
            print(log.read_text()[-3500:], flush=True)
            raise SystemExit(proc.returncode)
    result["completed"] = True
    (DOC / "acceptance_commands.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
