"""Serialize retained-check watcher processes and preserve full exit evidence."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CASES = {
    "mesh": (["--mode", "mesh"], {}, "1"),
    "headline": ([], {}, "10"),
    "continuation127": (
        ["--seq", "4097", "--steps", "127", "--batch", "2", "--split", "31", "--remap", "--heterogeneous"],
        {},
        "1",
    ),
    "stack31": (["--stack", "2", "--steps", "31"], {}, "1"),
    "fabric_single": ([], {"TT_METAL_DISABLE_FABRIC_TWO_ERISC": "1"}, "1"),
    "global_single": ([], {"TT_METAL_DISABLE_MULTI_AERISC": "1"}, "1"),
    "batch32": (
        ["--seq", "257", "--steps", "33", "--batch", "32", "--split", "31", "--remap", "--heterogeneous"],
        {},
        "1",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--watcher-interval", choices=["1", "10"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.cases:
        extra, mode, interval = CASES[name]
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("TT_METAL_WATCHER")
            and k
            not in (
                "TT_METAL_DISABLE_FABRIC_TWO_ERISC",
                "TT_METAL_DISABLE_MULTI_AERISC",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_PROFILER_DIR",
            )
        }
        overrides = {"TT_METAL_WATCHER": args.watcher_interval or interval, "TT_METAL_WATCHER_NOINLINE": "1", **mode}
        env.update(overrides)
        output = args.output_dir / (name + ".json")
        command = [
            sys.executable,
            "-m",
            "models.demos.k2_horizon_7b_qb2.tests.probe_multichip_watcher",
            "--output",
            str(output),
            *extra,
        ]
        log = args.output_dir / (name + ".log")
        print("START", name, flush=True)
        with log.open("w") as stream:
            process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        evidence = args.output_dir / (name + "_evidence")
        evidence.mkdir(exist_ok=True)
        for filename in ("watcher.log", "kernel_names.txt", "kernel_elf_paths.txt"):
            shutil.copyfile(Path("generated/watcher") / filename, evidence / filename)
        contents = log.read_text()
        failure = re.findall(
            r"(?im)^.*(?:TT_THROW|tripped assert|potentially unsafe|may be corrupted|Timeout|Skipping RISC reset|Traceback).*$",
            contents,
        )
        passed = process.returncode == 0 and not failure and "POST_CLOSE_WATCHER_POLLS_COMPLETE" in contents
        record = {
            "case": name,
            "command": command,
            "environment": overrides,
            "exit_code": process.returncode,
            "failure_lines": failure,
            "passed": passed,
        }
        records.append(record)
        output.with_suffix(".exit.json").write_text(json.dumps(record, indent=2) + "\n")
        (args.output_dir / "matrix.json").write_text(json.dumps(records, indent=2) + "\n")
        print("FINISH", name, "passed", passed, "exit", process.returncode, flush=True)
        if not passed:
            raise RuntimeError(f"Stopped after {name}; preserve live evidence and recover before another hardware run")


if __name__ == "__main__":
    main()
