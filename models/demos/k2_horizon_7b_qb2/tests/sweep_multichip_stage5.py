"""Run an explicit stage05 candidate manifest serially; stop on the first failure."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in json.loads(args.manifest.read_text()):
        name = case["name"]
        output = args.output_dir / (name + ".json")
        command = [
            sys.executable,
            "-m",
            "models.demos.k2_horizon_7b_qb2.tests.run_multichip",
            "--seq",
            "4096",
            "--steps",
            "128",
            "--output",
            str(output),
            *case.get("args", []),
        ]
        print("START", name, flush=True)
        started = time.monotonic()
        with (args.output_dir / (name + ".log")).open("w") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        record = {
            "name": name,
            "command": command,
            "exit_code": process.returncode,
            "elapsed_s": time.monotonic() - started,
        }
        if output.exists():
            value = json.loads(output.read_text())
            record.update(
                prefill_pcc=value["prefill_pcc"],
                decode_pcc=min(value["decode_pcc"]),
                prefill_host_us=value["measurements"]["multichip"]["prefill_host_us"],
                decode_host_us=value["measurements"]["multichip"]["decode_trace_host_mean_us"],
            )
        records.append(record)
        (args.output_dir / (args.manifest.stem + "_results.json")).write_text(json.dumps(records, indent=2) + "\n")
        print("FINISH", json.dumps(record), flush=True)
        if process.returncode:
            raise RuntimeError(f"Stop after {name}; inspect failure and device health before retrying")


if __name__ == "__main__":
    main()
