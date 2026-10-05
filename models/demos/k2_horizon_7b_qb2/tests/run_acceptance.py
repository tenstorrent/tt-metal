"""Serialized post-repair functional acceptance commands (no profiling)."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parents[1] / "doc/functional_decoder"
MODULE = "models.demos.k2_horizon_7b_qb2.tests."


def main():
    cases = [
        (
            "boundaries_final",
            "run_functional",
            [
                "--lengths",
                "1",
                "31",
                "32",
                "33",
                "96",
                "97",
                "127",
                "128",
                "129",
                "255",
                "256",
                "257",
                "777",
                "17",
                "--audit",
            ],
        ),
        ("batch32_final", "run_functional", ["--lengths", "33", "--batch", "32", "--audit"]),
        ("unchunked_final", "run_functional", ["--lengths", "257", "--unchunked-control", "--audit"]),
        ("positions_final", "run_positions", []),
        ("accurate_uniform_final", "probe_long_attention", ["--accurate", "--uniform"]),
        ("long_tails_final", "probe_long_layer", ["--tails"]),
        (
            "synthetic_final",
            "pytest",
            [
                "-q",
                str(ROOT / "models/demos/k2_horizon_7b_qb2/tests/test_functional_decoder.py"),
                "--basetemp=" + str(DOC / "pytest_tmp"),
            ],
        ),
        (
            "watcher_final",
            "run_functional",
            ["--lengths", "777", "--batch", "2", "--split", "31", "--remap", "--audit"],
        ),
    ]
    summary = {"completed": False, "commands": []}
    for name, module, args in cases:
        env = os.environ.copy()
        for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_DPRINT_CORES"):
            env.pop(key, None)
        if name == "watcher_final":
            env["TT_METAL_WATCHER"] = "10"
            env["TT_METAL_LOGS_PATH"] = str(DOC / "watcher_final")
        if module in ("run_functional", "run_positions"):
            args += ["--output", str(DOC / (name + ".json"))]
        elif module == "probe_long_layer":
            args += ["--output", name + ".json"]
        command = [sys.executable, "-m", module if module == "pytest" else MODULE + module, *args]
        log = DOC / "logs" / (name + ".log")
        print("START", name, flush=True)
        start = time.time()
        with log.open("w") as output:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        summary["commands"].append(
            {
                "name": name,
                "argv": command,
                "exit_code": result.returncode,
                "wall_seconds": time.time() - start,
                "log": str(log),
                "watcher_seconds": 10 if name == "watcher_final" else None,
            }
        )
        (DOC / "acceptance_commands.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("END", name, result.returncode, flush=True)
        if result.returncode:
            print(log.read_text()[-3000:], flush=True)
            raise SystemExit(result.returncode)
    for evidence in (DOC / "pytest_tmp").rglob("dense_*.json"):
        (DOC / ("synthetic_" + evidence.name)).write_text(evidence.read_text())
    summary["completed"] = True
    (DOC / "acceptance_commands.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
