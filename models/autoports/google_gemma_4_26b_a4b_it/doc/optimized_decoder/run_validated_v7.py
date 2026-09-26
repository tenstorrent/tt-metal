# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate adopted full-prefill fidelity plus shared final-default regressions."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DOC = Path(__file__).resolve().parent
RUNTIME = DOC.parent.parent / "tt/optimized_decoder.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-sha256", required=True)
    args = parser.parse_args()
    assert digest(RUNTIME) == args.runtime_sha256
    journal = DOC / "validated_v7_contract_commands.json"
    assert not journal.exists()
    flag_names = json.loads((DOC / "validated_v5_environment_snapshot.json").read_text())["environment"]
    assert not any(os.environ.get(key) for key in flag_names)
    write(
        DOC / "validated_v7_environment_snapshot.json",
        dict(
            environment={key: os.environ.get(key) for key in flag_names},
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            driver_sha256=digest(Path(__file__)),
            runtime_sha256=args.runtime_sha256,
            note="Watcher children add interval10; request reuse adds full allocation tracking and tracebacks. No exclusions.",
        ),
    )
    previous = json.loads((DOC / "validated_v5_contract_commands.json").read_text())
    records = []
    for entry in previous:
        old = entry["command"]
        if "--layer" in old:
            layer = int(old[old.index("--layer") + 1])
            if layer == 0 and not any("headline_layer" in v or "watcher_layer" in v for v in old):
                continue
        command = [value.replace("validated_v5", "validated_v7") for value in old]
        command[0] = sys.executable
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            name = output.stem
            assert not output.exists()
        elif "pytest" in command:
            name = "pytest_validated_v7"
        else:
            name = "stress_validated_v7_defaults"
        environment = os.environ | {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        if entry.get("watcher"):
            environment["TT_METAL_WATCHER"] = "10"
        if "request_reuse" in command:
            environment.update(TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_TRACEBACKS="1")
        record = dict(
            command=command,
            watcher=bool(entry.get("watcher")),
            runtime_sha256=args.runtime_sha256,
            driver_sha256=digest(Path(__file__)),
            environment_overrides={
                key: value
                for key, value in environment.items()
                if key.startswith("TT_METAL_") and key not in os.environ
            },
        )
        records.append(record)
        write(journal, records)
        print("Starting", name, flush=True)
        with (DOC / (name + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=2400)
        record["returncode"] = result.returncode
        if entry.get("watcher"):
            shutil.copyfile("generated/watcher/watcher.log", DOC / (name + ".device.log"))
        if "--output" in command and output.exists():
            record["output_sha256"] = digest(output)
        write(journal, records)
        assert digest(RUNTIME) == args.runtime_sha256
        print("Completed", name, result.returncode, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    assert len(records) == 12


if __name__ == "__main__":
    main()
