"""Try the untouched IFEval task after preserving serving evidence, on one clock."""

import json
import sys
import time
from pathlib import Path

from benchmark_stage.evidence import validate_server
from benchmark_stage.run import command

ROOT = Path(__file__).resolve().parents[1] / "doc/benchmark"
RUN = ROOT / "run"


def main():
    config = json.loads((RUN / "run_config.json").read_text())
    invocation = json.loads((ROOT / "invocation.json").read_text())
    deadline = invocation["started_monotonic"] + min(3600, config["budget_seconds"])
    # Leave time on the original clock for cancellation, final checks/report.
    inference_deadline = deadline - 130
    resume = RUN / "accuracy-resume"
    resume.mkdir(exist_ok=False)
    identity = resume / "server.json"
    record = {
        "original_invocation": invocation,
        "started_monotonic": time.monotonic(),
        "original_deadline": deadline,
        "inference_deadline": inference_deadline,
        "task": "ifeval",
        "reason": "Untouched task retried independently after both performance profiles; no clock reset",
    }
    try:
        command(
            [
                *config["performance_server_command"],
                "--max-num-seqs",
                "32",
                "--base-url",
                config["base_url"],
                "--output",
                str(identity),
            ],
            resume / "server-selection.log",
            inference_deadline,
        )
        validate_server(
            json.loads(identity.read_text()),
            32,
            config["model"],
            config["base_url"],
            baseline=json.loads((RUN / "perf-b32-server.json").read_text()),
            output=resume,
        )
        argv = [
            sys.executable,
            "-m",
            "benchmark_stage",
            "evaluate",
            "--model",
            config["model"],
            "--base-url",
            config["base_url"],
            "--manifest",
            str(RUN / "manifest.json"),
            "--task",
            "ifeval",
            "--output",
            str(resume),
            "--generation",
            json.dumps(config["generation"]["ifeval"]),
            "--shared",
        ]
        record["evaluation_command"] = argv
        command(argv, resume / "accuracy.log", inference_deadline)
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
    finally:
        record["finished_monotonic"] = time.monotonic()
        record["original_elapsed_seconds"] = time.monotonic() - invocation["started_monotonic"]
        (resume / "attempt.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
