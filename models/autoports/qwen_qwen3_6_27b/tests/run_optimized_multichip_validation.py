# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial stage-5 default-path validation planner/runner; imports no TTNN/torch.

Planning is the default. --run starts bounded serialized device subprocesses.
All writable artifacts stay in optimized_multichip_decoder. No policy override
or candidate installer is used: finalize the decoder defaults before --run.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[2]
DOC = MODEL / "doc/optimized_multichip_decoder"
MODULE = "models.autoports.qwen_qwen3_6_27b.tests."
PHASES = ("correctness", "contract", "stack", "resources", "context", "stress", "watcher")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_snapshot():
    files = list((MODEL / "tt").glob("*.py"))
    files += [
        MODEL / "tests" / name
        for name in (
            "run_optimized_multichip_validation.py",
            "run_multichip_contract.py",
            "run_multichip_decoder.py",
            "run_multichip_stack.py",
            "run_multichip_context.py",
            "run_multichip_boundaries.py",
            "check_multichip_stack_resources.py",
            "check_optimized_multichip_evidence.py",
            "multichip_assertions.py",
            "run_decoder.py",
        )
    ]
    files += [DOC / "run_job.sh", MODEL / "doc/hf_config.json", REPO / "models/demos/gpt_oss/tt/ccl.py"]
    from models.autoports.qwen_qwen3_6_27b.tests.check_optimized_multichip_evidence import native_build_snapshot

    return {
        **{str(p.relative_to(REPO)): digest(p) for p in sorted(set(files))},
        **native_build_snapshot(REPO)["native_source_sha256"],
    }


def build_jobs(args):
    jobs = []
    prefix = args.prefix

    def add(phase, stem, module, flags, *, baseline=False, reference=None, watcher=False, timeout=1200):
        name = f"{prefix}_{stem}"
        output = DOC / f"{name}.json"
        command = ["python", "-m", MODULE + module, *map(str, flags), "--output", str(output)]
        if reference:
            command += ["--reference", str(reference)]
        if baseline:
            command += ["--baseline"]
        jobs.append(
            {
                "name": name,
                "phase": phase,
                "module": module,
                "baseline": baseline,
                "command": command,
                "output": str(output),
                "reference": str(reference) if reference else None,
                "timeout_seconds": timeout,
                "environment": {
                    "TT_METAL_TRACE_ALLOC_TRACKING": "1",
                    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0",
                    "TT_METAL_WATCHER": "10" if watcher else None,
                    "TT_METAL_DEVICE_PROFILER": None,
                    "MULTICHIP_JOB_TIMEOUT": str(timeout),
                },
            }
        )

    for phase in args.phases:
        if phase in ("correctness", "watcher"):
            shapes = (
                ((1, 31), (2, 33), (3, 513), (32, 33), (1, 1025))
                if phase == "correctness"
                else (
                    (2, 256),
                    (2, 257),
                    (32, 256),
                    (32, 257),
                    (3, 513),
                    (17, 31),
                )
            )
            for layer in (0, 3):
                for batch, seq in shapes:
                    stem = f"{phase}_l{layer}_b{batch}_s{seq}"
                    ref = DOC / f"{prefix}_{stem}_oracle.pt"
                    flags = ["--layer", layer, "--batch", batch, "--length", seq, "--trace", "--repeats", 5]
                    add(phase, stem + "_baseline", "run_multichip_decoder", flags, baseline=True, reference=ref)
                    add(
                        phase,
                        stem + "_tp4",
                        "run_multichip_boundaries" if phase == "watcher" else "run_multichip_decoder",
                        flags,
                        reference=ref,
                        watcher=phase == "watcher",
                    )
        elif phase == "contract":
            for layer in (0, 3):
                add(
                    phase,
                    f"contract_l{layer}_tp4",
                    "run_multichip_contract",
                    ["--layer", layer],
                    watcher=True,
                    timeout=1800,
                )
        elif phase == "stack":
            activation = MODEL / "doc/optimized_decoder/text_activations/text_l0_b32_s254.pt"
            for batch, seq in ((2, 65), (32, 33)):
                stem = f"stack_b{batch}_s{seq}"
                ref = DOC / f"{prefix}_{stem}_oracle.pt"
                flags = ["--batch", batch, "--length", seq, "--layers", "0,1,2,3,4", "--activation-file", activation]
                add(phase, stem + "_baseline", "run_multichip_stack", flags, baseline=True, reference=ref)
                add(phase, stem + "_tp4", "run_multichip_stack", flags, reference=ref)
        elif phase == "resources":
            for batch in (1, 32):
                add(
                    phase,
                    f"resources_b{batch}_tp4",
                    "check_multichip_stack_resources",
                    [
                        "--batch",
                        batch,
                        "--prefill-batch",
                        batch,
                        "--prefill-length",
                        65,
                        "--trace-mib",
                        32,
                        "--replays",
                        3,
                    ],
                    watcher=True,
                    timeout=1800,
                )
        elif phase == "context":
            for layer in (0, 3):
                for seq in args.context_lengths:
                    stem = f"context_l{layer}_s{seq}"
                    ref = DOC / f"{prefix}_{stem}_oracle.pt"
                    flags = ["--layer", layer, "--length", seq]
                    add(
                        phase,
                        stem + "_baseline",
                        "run_multichip_context",
                        flags,
                        baseline=True,
                        reference=ref,
                        timeout=1800,
                    )
                    add(
                        phase,
                        stem + "_tp4",
                        "run_multichip_context",
                        flags + ["--reserve-gib", args.reserve_gib],
                        reference=ref,
                        timeout=1800,
                    )
        elif phase == "stress":
            for layer in (0, 3):
                for batch, seq, steps in ((1, 128, 128), (32, 33, 512)):
                    stem = f"stress_l{layer}_b{batch}_s{seq}"
                    ref = DOC / f"{prefix}_{stem}_oracle.pt"
                    flags = [
                        "--layer",
                        layer,
                        "--batch",
                        batch,
                        "--length",
                        seq,
                        "--stress",
                        steps,
                        "--trace",
                        "--repeats",
                        5,
                    ]
                    add(
                        phase,
                        stem + "_baseline",
                        "run_multichip_decoder",
                        flags,
                        baseline=True,
                        reference=ref,
                        timeout=1800,
                    )
                    for repeat in range(1, args.stress_runs + 1):
                        add(
                            phase, stem + f"_tp4_r{repeat}", "run_multichip_decoder", flags, reference=ref, timeout=1800
                        )
    return jobs


def require(value, message):
    if not value:
        raise AssertionError(message)


def validate(job):
    output = Path(job["output"])
    report = json.loads(output.read_text())
    require(report.get("pass", True) and report.get("status", "pass") == "pass", f"Failed artifact: {output}")
    if job["reference"]:
        require(Path(job["reference"]).is_file(), "Missing oracle fixture")
    module = job["module"]
    if module in ("run_multichip_decoder", "run_multichip_boundaries"):
        require(report["finite"] and report["trace"]["bit_equal_eager"], "Finite/eager-trace check missing")
        require(
            report["trace"]["allocation_tracking"] and report["trace"]["positive_control_rejected"],
            "Tracker gate missing",
        )
        require(report["trace"]["acknowledged_buffers"] == [], "Trace allocation acknowledgments present")
        if job["phase"] == "stress":
            stress = report["stress"]
            require(
                stress["runs"] == 2 and stress["bit_equal"] and stress["changed_inputs"],
                "Internal repeated stress missing",
            )
            require(stress["steps"] == (128 if report["batch"] == 1 else 512), "Wrong stress duration")
            if not job["baseline"]:
                require(min(stress["output_pcc"], *stress["state_pcc"]) >= 0.995, "Stress PCC failed")
        if not job["baseline"]:
            require(report["pass"] and report["provenance"]["mesh"] == [1, 4], "Final TP4 path not verified")
            require(
                report["provenance"]["decode_input_shape_per_rank"] == [1, report["batch"], 1280],
                "Decode is not native compact",
            )
    elif module == "run_multichip_stack":
        from models.autoports.qwen_qwen3_6_27b.tests.check_optimized_multichip_evidence import validate_boundary_report

        validate_boundary_report(report, tp4=not job["baseline"])
        require(report["no_boundary_conversion"] and report["trace_bit_equal"], "Direct stack gate missing")
        if not job["baseline"]:
            require(
                report["boundary_shape"] == [1, report["batch"], 1280], "Stack boundary is not compact hidden-sharded"
            )
            require(min(report["pcc"].values()) >= 0.995, "Stack PCC failed")
    elif module == "check_multichip_stack_resources":
        from .check_optimized_multichip_evidence import validate_fused_decode_norm_resources

        validate_fused_decode_norm_resources(report)
        require(report["resource_counts"]["layers"] == 64 and report["mesh"] == [1, 4], "64 TP4 owners not exercised")
        require(
            report["tracker_positive_control"]["pass"] and report["allocation_acknowledgments"] == 0,
            "Resource tracker gate missing",
        )
        require(len(report["replays"]) >= 3, "Resource replay coverage incomplete")
    elif module == "run_multichip_context":
        require(
            report["full_decoder_all_tokens"] and report["traced_decode_bit_equal"], "Context numerical path missing"
        )
        require(len(report["trace_cases"]) == 4, "Context changed-input replay coverage missing")
        if not job["baseline"]:
            require(report["pass"], "Context PCC failed")
    elif module == "run_multichip_contract":
        from models.autoports.qwen_qwen3_6_27b.tests.check_optimized_multichip_evidence import (
            validate_hf_contract_report,
        )

        validate_hf_contract_report(report)
    if job["environment"]["TT_METAL_WATCHER"]:
        log = DOC / (job["name"] + ".watcher.log")
        text = log.read_text()
        require(not any(x in text for x in ("Watcher detected", "ASSERT", "TT_FATAL")), "Watcher error")
        require(all(f"detach device {rank}" in text for rank in range(4)), "Watcher does not cover all four devices")
    return report


def artifact_metadata(job):
    files = [Path(job["output"]), DOC / (job["name"] + ".command.json"), DOC / (job["name"] + ".log")]
    if job["reference"]:
        files.append(Path(job["reference"]))
    if job["environment"]["TT_METAL_WATCHER"]:
        files.append(DOC / (job["name"] + ".watcher.log"))
    return {
        str(path): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": digest(path)}
        for path in files
    }


def observe_native_child(parent_pid, job):
    """Observe only this job's descendant module process; never import TTNN."""
    from models.autoports.qwen_qwen3_6_27b.tests.check_optimized_multichip_evidence import loaded_native_snapshot

    parents = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            parents[int(path.name)] = int((path / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants = {parent_pid}
    while True:
        expanded = descendants | {pid for pid, ppid in parents.items() if ppid in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    for pid in sorted(descendants):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            if MODULE + job["module"] in argv and job["output"] in argv:
                snapshot = loaded_native_snapshot(REPO, pid)
                snapshot["observed_command"] = argv[:-1] if argv[-1] == "" else argv
                return snapshot
        except (OSError, ValueError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="final_validation")
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--context-lengths", nargs="+", type=int, choices=(32768, 65536), default=[32768])
    parser.add_argument("--reserve-gib", type=int, default=20)
    parser.add_argument("--stress-runs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--expected-default-sha256")
    parser.add_argument("--run", action="store_true", help="Execute device jobs; otherwise only write a plan")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix), "Prefix must be a simple filename component")
    require(len(args.phases) == len(set(args.phases)), "Duplicate phases")
    require(args.reserve_gib >= 0, "Negative reservation")
    DOC.mkdir(parents=True, exist_ok=True)
    sources = source_snapshot()
    from models.autoports.qwen_qwen3_6_27b.tests.check_optimized_multichip_evidence import native_build_snapshot

    native_build = native_build_snapshot(REPO)
    default_hash = digest(MODEL / "tt/multichip_decoder.py")
    if args.expected_default_sha256:
        require(default_hash == args.expected_default_sha256, "Selected default source changed")
    jobs = build_jobs(args)
    definition = {
        "jobs": jobs,
        "source_sha256": sources,
        "selected_default_sha256": default_hash,
        "native_build": native_build,
        "configuration": {
            "prefix": args.prefix,
            "phases": args.phases,
            "context_lengths": args.context_lengths,
            "reserve_gib": args.reserve_gib,
            "stress_runs": args.stress_runs,
        },
    }
    signature = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    plan = DOC / (args.prefix + "_manifest.json")
    if plan.exists():
        existing = json.loads(plan.read_text())
        require(existing["signature"] == signature, "Existing manifest differs; preserve it and choose a new prefix")
    else:
        plan.write_text(json.dumps({"created_utc": utc(), "signature": signature, **definition}, indent=2) + "\n")
    print(json.dumps({"plan": str(plan), "jobs": len(jobs), "phases": args.phases, "execute": args.run}), flush=True)
    if not args.run:
        return
    # This serializes this orchestrator's invocations. The parent must also
    # finish other device drivers before starting it; no reset/kill is done.
    with (DOC / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in jobs:
            require(not (DOC / "STOP").exists(), "Parent requested stop between device jobs")
            require(source_snapshot() == sources, "Source changed during final validation; stop and use a new manifest")
            require(native_build_snapshot(REPO) == native_build, "Native source/build changed during final validation")
            receipt_path = DOC / (job["name"] + ".validation.json")
            if args.resume and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("pass"):
                    require(receipt["signature"] == signature, "Stale resume receipt")
                    validate(job)
                    require(receipt["artifact_metadata"] == artifact_metadata(job), "Resume evidence changed")
                    print("RESUME " + job["name"], flush=True)
                    continue
            occupied = [
                DOC / (job["name"] + suffix) for suffix in (".json", ".log", ".command.json", ".validation.json")
            ]
            require(
                not any(p.exists() for p in occupied),
                "Existing unverified/failed artifacts: choose a fresh prefix; do not overwrite",
            )
            if job["baseline"]:
                require(not Path(job["reference"]).exists(), "Oracle fixture already exists without a passing receipt")
            env = dict(os.environ)
            for name in (
                "QWEN_DECODER",
                "QWEN_OPT_POLICY",
                "QWEN_OPT_CANDIDATE",
                "QWEN_FUSION_CANDIDATE",
                "TT_VISIBLE_DEVICES",
            ):
                env.pop(name, None)
            for name, value in job["environment"].items():
                if value is None:
                    env.pop(name, None)
                else:
                    env[name] = value
            started = time.monotonic()
            receipt = {
                "pass": False,
                "signature": signature,
                "job": job,
                "started_utc": utc(),
                "effective_environment": {
                    k: v for k, v in env.items() if k.startswith("TT_") or k == "MULTICHIP_JOB_TIMEOUT"
                },
            }
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            try:
                process = subprocess.Popen(
                    ["bash", str(DOC / "run_job.sh"), job["name"], *job["command"]],
                    cwd=REPO,
                    env=env,
                )
                deadline = time.monotonic() + job["timeout_seconds"] + 120
                try:
                    while process.poll() is None:
                        if "native_loaded" not in receipt:
                            observed = observe_native_child(process.pid, job)
                            if observed is not None:
                                receipt["native_loaded"] = observed
                        if time.monotonic() > deadline:
                            raise subprocess.TimeoutExpired(process.args, job["timeout_seconds"] + 120)
                        try:
                            process.wait(timeout=0.25)
                        except subprocess.TimeoutExpired:
                            pass
                except BaseException:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    raise
                receipt["exit_code"] = process.returncode
                require(process.returncode == 0, f"Job failed: {job['name']}")
                require("native_loaded" in receipt, "Actual target native-library mappings were not observed")
                from models.autoports.qwen_qwen3_6_27b.tests.check_optimized_multichip_evidence import (
                    validate_loaded_native,
                )

                validate_loaded_native(receipt["native_loaded"], native_build)
                command = json.loads((DOC / (job["name"] + ".command.json")).read_text())
                require(
                    command["native_build"] == native_build, "Command native build differs from prepared validation"
                )
                require(
                    all(
                        command["environment"].get(key) == value
                        for key, value in job["environment"].items()
                        if key.startswith("TT_METAL_")
                    ),
                    "Actual instrumentation environment differs from planned validation",
                )
                require(source_snapshot() == sources, "Source changed while device job was active")
                require(
                    native_build_snapshot(REPO) == native_build,
                    "Native source/build changed while device job was active",
                )
                receipt["native_build_verified_before_after"] = True
                report = validate(job)
                if job["module"] == "run_multichip_contract":
                    receipt["reference_provenance"] = {
                        key: report[key]
                        for key in ("contract_schema_version", "pcc_threshold", "hf_revision", "reference_types")
                    }
                receipt["artifact_metadata"] = artifact_metadata(job)
                receipt["pass"] = True
            except BaseException as exc:
                receipt["error"] = repr(exc)
                raise
            finally:
                receipt["finished_utc"] = utc()
                receipt["duration_seconds"] = time.monotonic() - started
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                with (DOC / (args.prefix + "_commands.jsonl")).open("a") as stream:
                    stream.write(json.dumps(receipt) + "\n")
        summary = {
            "pass": True,
            "signature": signature,
            "manifest": str(plan),
            "jobs": len(jobs),
            "phases": args.phases,
            "receipts": [str(DOC / (j["name"] + ".validation.json")) for j in jobs],
            "finished_utc": utc(),
        }
        (DOC / (args.prefix + "_summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
