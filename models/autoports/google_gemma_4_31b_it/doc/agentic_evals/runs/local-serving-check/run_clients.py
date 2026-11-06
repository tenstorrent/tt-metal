# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Coordinator-launched HTTP measurements against an existing idle B8 server."""
import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from models.autoports.google_gemma_4_31b_it.tests.check_served_state import log_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--engine-pid", type=int, required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    model = Path("models/autoports/google_gemma_4_31b_it")
    out = Path(__file__).resolve().parent
    baseline = args.server_log.read_bytes()
    registrations = [e for e in log_events(baseline) if e["event"] == "autoport_registration"]
    if not registrations:
        raise RuntimeError("No autoport registration event")
    registration = registrations[-1]
    assert registration["class"] == "models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM"
    assert registration["layers"] == list(range(60))
    assert Path(registration["module_file"]).resolve() == (model / "tt/generator_vllm.py").resolve()
    publication = json.loads((model / "doc/agentic_evals/setup/source_publication.json").read_text())
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in publication["files"]}
    assert all(
        source_hashes[name] == item["sha256"] for name, item in publication["files"].items()
    ), "Published source hash mismatch"
    report = dict(
        status="running",
        registration=registration,
        planned_clients=2,
        batch=8,
        overlap_planned_clients=8,
        regime="serial local p300x2; no other device jobs",
        source_sha256=source_hashes,
        publication_hashes_match=True,
        steps=[],
    )
    target = out / "client_manifest.json"

    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")

    save()
    commands = [
        (
            "growing",
            [
                "python_env/bin/python",
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests.check_agentic_serving_traffic",
                "--concurrency",
                "2",
                "--contexts",
                "4096,8192,16384,24576,32768",
                "--server-log",
                str(args.server_log),
                "--url",
                args.url,
                "--output",
                str(out / "growing.json"),
            ],
        ),
        (
            "load_costs",
            [
                "python_env/bin/python",
                str(model / "tests/check_serving_load_costs.py"),
                "--batch",
                "8",
                "--stages",
                "prefill,overlap",
                "--engine-pid",
                str(args.engine_pid),
                "--server-log",
                str(args.server_log),
                "--url",
                args.url,
                "--output",
                str(out / "load_costs.json"),
            ],
        ),
    ]
    for name, command in commands:
        row = dict(name=name, command=["timeout", "900", *command], start_unix=time.time())
        report["steps"].append(row)
        save()
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(row["command"], stdout=log, stderr=subprocess.STDOUT)
        row.update(end_unix=time.time(), returncode=result.returncode)
        save()
        if result.returncode:
            report["status"] = "failed"
            save()
            return result.returncode
    events = log_events(args.server_log.read_bytes()[len(baseline) :])
    report["sampling_steps"] = dict(
        Counter(f"{e.get('phase')}:{e.get('mode')}" for e in events if e["event"] == "sampling_mode")
    )
    report["status"] = "pass"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
