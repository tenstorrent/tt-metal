"""Run the packaged stage and include final evidence checks in its wall clock."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
ROOT = MODEL_DIR / "doc/benchmark"
RUN = ROOT / "run"
PLUGIN = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])


def fail(message, started):
    """Keep package and outer-check failures truthful on the same clock."""
    elapsed = time.monotonic() - started
    path = RUN / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary.setdefault("packaged_runner_elapsed_seconds", summary.get("elapsed_seconds"))
    summary.update(status="failed", error=message, elapsed_seconds=elapsed)
    summary["clock_scope"] = "before benchmark_stage run invocation through failed outer closure"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    report = RUN / "REPORT.md"
    body = report.read_text() if report.exists() else "# Benchmark: IFM/K2-Horizon-7B\n"
    body = body.replace("Status: completed.", "Status: failed.", 1)
    report.write_text(
        body + f"\nFinal status: failed. {message}\n\nOuter client-stage elapsed: {elapsed:.3f} seconds.\n"
    )
    raise SystemExit(1)


def main():
    if RUN.exists():
        raise FileExistsError("Preserve or archive the existing run before rerunning")
    started = time.monotonic()
    from models.demos.k2_horizon_7b_qb2.tests.benchmark_server import STATE, alive

    run_config = json.loads((ROOT / "run_config.json").read_text())
    order_plan = None
    if run_config.get("request_order_plan"):
        order_plan = (ROOT / run_config["request_order_plan"]).resolve()
        if not order_plan.is_file():
            raise FileNotFoundError(order_plan)
        os.environ["K2_BENCHMARK_REQUEST_ORDER"] = str(order_plan)
        hook = MODEL_DIR / "tests/benchmark_order_hook"
        os.environ["PYTHONPATH"] = str(hook) + os.pathsep + os.environ.get("PYTHONPATH", "")

    prior_server = json.loads(STATE.read_text()) if STATE.exists() else {}
    starting_server = {
        "running": bool(prior_server and alive(prior_server)),
        "recorded_process": prior_server,
        "inspected_unix": time.time(),
    }
    invocation = {
        "started_monotonic": started,
        "started_unix": time.time(),
        "starting_server": starting_server,
        "request_order_plan": str(order_plan) if order_plan else None,
        "request_order_plan_sha256": hashlib.sha256(order_plan.read_bytes()).hexdigest() if order_plan else None,
        "command": [
            sys.executable,
            "-m",
            "benchmark_stage",
            "run",
            "--config",
            str(ROOT / "run_config.json"),
            "--output",
            str(RUN),
        ],
    }
    (ROOT / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    with (ROOT / "runner.log").open("w") as log:
        code = subprocess.call(invocation["command"], stdout=log, stderr=subprocess.STDOUT)
    if code:
        fail(f"Packaged benchmark runner exited {code}; see retained runner log and evidence", started)
    # Preserve exact upstream inputs and the measured performance prompts
    # inside this same clock. These helpers do not generate model outputs.
    for name in ("preserve_benchmark_inputs", "preserve_performance_prompts", "verify_benchmark_chat_tokens"):
        interpreter = sys.executable
        environment = dict(os.environ)
        if name == "preserve_performance_prompts":
            run_config = json.loads((RUN / "run_config.json").read_text())
            interpreter = str(Path(run_config["vllm_cli"]).with_name("python"))
            environment.pop("PYTHONPATH", None)
            environment["VLLM_PLUGINS"] = ""
            environment["VLLM_TARGET_DEVICE"] = "empty"
        elif name == "verify_benchmark_chat_tokens":
            interpreter = str(MODEL_DIR.parents[2] / "python_env/bin/python")
        with (RUN / f"{name}.log").open("w") as log:
            result = subprocess.run(
                [interpreter, str(MODEL_DIR / "tests" / f"{name}.py")],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            fail(f"{name} failed with exit {result.returncode}; see its retained log", started)
    if order_plan is not None:
        with (RUN / "request-order-check.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(MODEL_DIR / "tests/benchmark_request_order.py"),
                    "--verify-run",
                    str(RUN),
                    "--plan",
                    str(order_plan),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            fail(f"Request-order proof failed with exit {result.returncode}; see its retained log", started)
    # These offline checks and the report supplement are included in the same
    # continuously running clock, beyond the package's own timing/watchdog.
    commands = [
        [
            sys.executable,
            "-m",
            "benchmark_stage.check",
            "--model-dir",
            str(MODEL_DIR),
            "--hf-model",
            "IFM/K2-Horizon-7B",
        ],
        [
            sys.executable,
            str(PLUGIN / "scripts/check_context_contract.py"),
            "--model-dir",
            str(MODEL_DIR),
            "--hf-model",
            "IFM/K2-Horizon-7B",
            "--stage",
            "benchmark",
            "--require-contract",
        ],
    ]
    for label, command in zip(("evidence-check", "context-check"), commands):
        with (RUN / f"{label}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            fail(f"{label} failed with exit {result.returncode}; see its retained log", started)
    original = (ROOT / "setup/context-contract-original.sha256").read_text().strip()
    if hashlib.sha256((MODEL_DIR / "doc/context_contract.json").read_bytes()).hexdigest() != original:
        fail("Context contract bytes changed", started)
    config = json.loads((RUN / "run_config.json").read_text())
    report = RUN / "REPORT.md"
    with report.open("a") as out:
        out.write("\n## Protocol and implementation evidence\n\n")
        out.write(config["protocol_notes"] + "\n\n" + config["reference_notes"] + "\n\n")
        if order_plan is not None:
            out.write(
                "Request admission used a frozen runtime-based order, with prior unfinished and longer responses first. "
                "All frozen questions were generated again; no responses were reused. The client restored upstream "
                "result order before scoring. [Frozen scheduling plan](request-order-plan.json), "
                "[actual scheduling and response proof](request-order-verification.json).\n\n"
            )
        out.write("| Task | Finish reasons |\n|---|---|\n")
        for task in config["tasks"]:
            results = json.loads((RUN / task / "results.json").read_text())
            reasons = results["benchmark_stage"]["finish_reasons"]
            out.write(f"| {task} | `{json.dumps(reasons, sort_keys=True)}` |\n")
        out.write("\n")
        if config.get("setup_record"):
            setup_path = ROOT / config["setup_record"]
            setup = json.loads(setup_path.read_text())
            out.write(
                f"Pre-invocation setup completed at {setup['recorded_utc']}; "
                f"[setup record](../{config['setup_record']}). "
                "Earlier incomplete attempts are retained separately and are not combined into this run's scores or timing.\n\n"
            )
            for attempt in setup.get("prior_attempts", []):
                out.write(
                    f"- [{attempt['path']}](../{attempt['path']}/run/REPORT.md): "
                    f"{attempt['elapsed_seconds']:.3f} seconds; incomplete.\n"
                )
            out.write("\n")
        out.write(
            "Full 36-layer generated autoport, revision `036114ce8d46c32b24c15423211069abb9c5d25e`, "
            "precision `down4_l24to34_head8_lofi`, four Blackhole chips (TP4/DP1), and context 524288. "
            "[Observed identity](../identity.json), [setup and limitations](../RUN_NOTES.md), "
            "[frozen documents](manifest.json), [evidence check](evidence-check.log), "
            "[context check](context-check.log), [actual prompt/token parsing check](chat-token-verification.json).\n\n"
        )
        if starting_server["running"]:
            supplied = starting_server["recorded_process"]
            out.write(
                f"The supplied server had {supplied['max_num_seqs']} slots; "
                f"its prior setup startup took {supplied['startup_seconds']:.3f} seconds. "
            )
        else:
            out.write(
                "No live server was supplied to this attempt. Its initial 32-slot launch " "is included in the clock. "
            )
        out.write(
            "All runner server selections/reloads, compilation, warmups, inference, collection, "
            "scoring, input preservation, reporting and final offline checks are included below.\n"
        )
    summary_path = RUN / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["packaged_runner_elapsed_seconds"] = summary["elapsed_seconds"]
    elapsed = time.monotonic() - started
    summary["elapsed_seconds"] = elapsed
    summary["clock_scope"] = "before benchmark_stage run invocation through report and evidence/context checks"
    summary["evidence_check"] = "passed"
    summary["context_check"] = "passed; original contract bytes unchanged"
    if order_plan is not None:
        summary["request_order_check"] = "passed; all frozen requests and upstream result mapping preserved"
    if elapsed >= min(3600, config["budget_seconds"]):
        summary.update(status="failed", error="Complete client stage including final checks exceeded one hour")
        report.write_text(report.read_text().replace("Status: completed.", "Status: failed.", 1))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with report.open("a") as out:
        out.write(f"\nComplete client stage including final checks: {elapsed:.3f} seconds.\n")
        if summary["status"] != "completed":
            out.write("\nFinal status: failed. " + summary["error"] + "\n")
    print(json.dumps({"status": summary["status"], "elapsed_seconds": elapsed}))
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
