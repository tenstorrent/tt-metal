#!/usr/bin/env python3
"""Adapter around the preserved AE leg. No changes to the frozen8+3 protocol."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import slurm_pair_checks as checks
import producer_profile as profiles
import inner_host_quiet
from owned_stock import StockProcesses


def admit_completed_leg(original, output, manifest, code, producer=False):
    assert json.loads((output / "phase-0.json").read_text())["exit_code"] == 0
    phase = json.loads((output / "phase-1.json").read_text())["exit_code"]
    original.completed_assertions(phase, output, manifest["assertion_payload_proof"])
    assert code == phase and code in (0, 1)
    assert not (output / "cache-generation-prohibited").exists()
    assert (output / ("cache-cap.jsonl" if producer else "cache-guard.jsonl")).is_file()
    assert not json.loads((output / "server-cleanup.json").read_text())["escalated"]
    assert json.loads((output / "post-cleanup.json").read_text())["quiet"]


def run_leg(name):
    base = Path("/pair")
    producer = name == "producer"
    stock_name = "baseline" if producer else name
    output = base / "evidence" / stock_name
    output.mkdir(parents=True, exist_ok=True)
    assignment = json.loads((base / "evidence/slurm-assignment.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "preserved_formatter_control", base / "control/.github/scripts/utils/formatter_serial_control.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    def quiet(path):
        return inner_host_quiet.request(path, assignment, name)

    processes = StockProcesses(original, output)
    original.subprocess = processes
    original.process_identity = processes.identity
    original.close_server = processes.close_server
    preserved_run = processes.run_owned

    def admitted_run(command, log, env, cwd, seconds):
        phase_env = env.copy()
        if log.name.startswith("phase-"):
            # Client-only processes do not import TTNN or change server instrumentation.
            for key in ("FORMATTER_CACHE_GUARD", "FORMATTER_PROFILE_RECORD", "FORMATTER_CACHE_PRODUCER"):
                phase_env.pop(key, None)
        if not producer and log.name == "phase-0.log":
            # Both scored arms use the frozen observer's existing replay branch.
            phase_env["FORMATTER_PAIR_LEG"] = "candidate"
        code = preserved_run(command, log, phase_env, cwd, seconds)
        if log.name == "install.log" and code == 0:
            # UMD topology discovery default leaves6u retraining disabled; it does not
            # start devices or reset boards. Use the SAME sealed EE native executable.
            quiet(output / "pre-topology-quiet.json")
            binary = Path("/work/build/tools/umd/topology")
            assert binary.is_file()
            member = json.loads((base / "evidence/native-admission.json").read_text())["critical_members"][
                "build/tools/umd/topology"
            ]
            assert binary.stat().st_size == member["bytes"] and checks.sha(binary) == member["sha256"]
            path = output / "topology.yaml"
            assert preserved_run([str(binary), "--path", str(path)], output / "topology.log", env, cwd, 120) == 0
            assert not original.HANG.search((output / "topology.log").read_text(errors="replace"))
            import yaml

            receipt = checks.topology(yaml.safe_load(path.read_text()), assignment["driver_indices"])
            receipt.update(
                {
                    "binary_sha256": checks.sha(binary),
                    "retrain_requested": False,
                    "reset_requested": False,
                    "diagnostic_only": True,
                }
            )
            original.write(output / "topology.json", receipt)
            quiet(output / "post-topology-quiet.json")
            env["PYTHONPATH"] += ":/pair/adapter:/pair/adapter/runtime_guard"
            env["FORMATTER_PROFILE_RECORD"] = str(output / "model-profile.json")
            env["FORMATTER_PROFILE_SOURCE"] = (
                original.BASELINE if producer or name == "baseline" else original.CANDIDATE
            )
            env["FORMATTER_PROFILE_PHASE"] = name
            if producer:
                env["FORMATTER_CACHE_PRODUCER"] = "1"
                env["FORMATTER_CACHE_CAP_RECEIPTS"] = str(output / "cache-cap.jsonl")
                env.pop("FORMATTER_CACHE_GUARD", None)
            else:
                env["FORMATTER_CACHE_GUARD"] = "1"
            env["FORMATTER_CACHE_GUARD_SENTINEL"] = str(output / "cache-generation-prohibited")
            env["FORMATTER_CACHE_GUARD_RECEIPTS"] = str(output / "cache-guard.jsonl")
        return code

    original.quiet = quiet
    original.run = admitted_run
    code = original.leg("chunked", stock_name)
    manifest = json.loads((base / "control/.github/scripts/utils/formatter_pair_manifest.json").read_text())
    admit_completed_leg(original, output, manifest, code, producer)
    profile_receipt = json.loads((output / "model-profile.json").read_text())
    profiles.validate_profile(profile_receipt["profile"])
    assert profile_receipt["phase"] == name and profile_receipt["source"] == env_source(stock_name)
    original.write(
        output / "leg-status.json",
        {
            "complete": True,
            "exit_code": code,
            "source": env_source(stock_name),
            "phase": name,
            "profile": profile_receipt["profile"],
            "cache_production_declared": producer,
            "control_reference_sha256": checks.sha(base / "control/.github/scripts/utils/formatter_serial_control.py"),
            "native_members": manifest["native_members"],
            "all_three_assertions_completed": True,
            "payloads_sha256": manifest["assertion_payload_proof"]["payloads_sha256"],
            "diagnostic_only": True,
            "timing_qualified": False,
        },
    )
    return code


def env_source(name):
    return {"baseline": profiles.BASELINE, "candidate": profiles.CANDIDATE}[name]


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise RuntimeError("Owned leg cancelled; ordinary server close, preserve and stop")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    assert len(sys.argv) == 2 and sys.argv[1] in ("producer", "baseline", "candidate")
    try:
        raise SystemExit(run_leg(sys.argv[1]))
    except Exception as error:
        inner_host_quiet.report_error(Path("/pair"), sys.argv[1], error)
        raise SystemExit(2)
