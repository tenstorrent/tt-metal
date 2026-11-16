#!/usr/bin/env python3
"""One-case, root-admitted cap1024 extension. No launch without explicit --execute."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

CANDIDATE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[7]
H9 = (
    REPO_ROOT
    / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup/swe85-conditional-hf/revision-adaptive-expanded-call91-8192-reviewed"
)
DEADLINE = dt.datetime(2026, 11, 16, 1, 15, 25, tzinfo=dt.timezone.utc).timestamp()
TIMEOUT_SECONDS = 1200
REQUIRED_REMAINING_SECONDS = TIMEOUT_SECONDS + 60
PINNED_FILES = (
    "proposal.json",
    "command.json",
    "cases.json",
    "forecast.json",
    "source-pins.json",
    "runtime-pins.json",
    "admission-template.json",
    "run_cap1024.py",
    "test_run_cap1024.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fail(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolved(repo: Path, rel: str) -> Path:
    p = (repo / rel).resolve()
    fail(p.is_relative_to(repo.resolve()), f"Path escapes repo root: {rel}")
    return p


def verify_h9_predecessor(repo: Path, gate: dict[str, Any]) -> dict[str, Any]:
    """Require terminal HF9 progress, all eight completion/root-review pairs, and miss acceptances."""
    pp = resolved(repo, gate["progress"]["path"])
    fail(digest(pp) == gate["progress"]["sha256"], "HF9 progress hash differs from root admission")
    progress = read_json(pp)
    fail(progress.get("status") == gate["terminal_status"], "HF9 progress status differs from root admission")
    groups = progress.get("groups", [])
    fail(len(groups) == 8, f"HF9 must have 8 terminal groups, found {len(groups)}")
    fail(all(g.get("status") == "completed" for g in groups), "HF9 has a nonterminal group")
    evidence = {int(x["serial_order"]): x for x in gate.get("groups", [])}
    fail(set(evidence) == set(range(1, 9)), "Admission must pin all eight HF9 group completions and root reviews")
    verified = []
    for row in groups:
        n = int(row["serial_order"])
        ev = evidence[n]
        cp = resolved(repo, ev["completion"]["path"])
        rp = resolved(repo, ev["root_review"]["path"])
        fail(digest(cp) == ev["completion"]["sha256"], f"HF9 group{n} completion hash differs")
        comp = read_json(cp)
        fail(comp.get("exit_code") == 0, f"HF9 group{n} completion is not exit0")
        fail(
            row.get("completion_sha256") == ev["completion"]["sha256"],
            f"HF9 progress completion hash mismatch in group{n}",
        )
        fail(digest(rp) == ev["root_review"]["sha256"], f"HF9 group{n} root review hash differs")
        review = read_json(rp)
        fail(
            review.get("reviewer") == "root" and review.get("accepted") is True,
            f"HF9 group{n} lacks accepted root review",
        )
        if "completion_sha256" in review:
            fail(
                review["completion_sha256"] == ev["completion"]["sha256"],
                f"HF9 group{n} review points at different completion",
            )
        verified.append(
            {
                "serial_order": n,
                "completion_sha256": ev["completion"]["sha256"],
                "root_review_sha256": ev["root_review"]["sha256"],
            }
        )

    acceptances = {int(x["serial_order"]): x for x in gate.get("forecast_miss_acceptances", [])}
    for row in groups:
        if row.get("forecast_miss"):
            n = int(row["serial_order"])
            fail(n in acceptances, f"HF9 group{n} forecast miss lacks pinned root acceptance")
            ev = acceptances[n]
            ap = resolved(repo, ev["path"])
            fail(digest(ap) == ev["sha256"], f"HF9 group{n} forecast acceptance hash differs")
            rec = read_json(ap)
            fail(
                rec.get("reviewer") == "root"
                and rec.get("accepted") is True
                and rec.get("status") == "root_accepted_forecast_miss",
                f"HF9 group{n} forecast miss not root accepted",
            )
            comp_hash = evidence[n]["completion"]["sha256"]
            fail(
                rec.get("completion_sha256") == comp_hash,
                f"HF9 group{n} forecast acceptance references a different completion",
            )
            if row.get("forecast_miss_acceptance_sha256") is not None:
                fail(
                    row["forecast_miss_acceptance_sha256"] == ev["sha256"],
                    f"HF9 group{n} progress acceptance hash mismatch",
                )
    return {"progress_sha256": gate["progress"]["sha256"], "groups": verified}


def verify_candidate(candidate: Path, repo: Path) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Validate root admission and every frozen candidate/source/runtime pin without importing TT."""
    admission_path = candidate / "admission.json"
    fail(admission_path.is_file(), "Root admission.json is absent")
    admission = read_json(admission_path)
    fail(
        admission.get("status") == "admitted"
        and admission.get("reviewer") == "root"
        and admission.get("accepted") is True,
        "Root admission is not explicitly accepted",
    )
    fail(
        admission.get("deadline_utc") == "2026-11-16T01:15:25Z" and admission.get("timeout_seconds") == TIMEOUT_SECONDS,
        "Admission deadline/timeout mismatch",
    )
    pins = admission.get("candidate_pins", {})
    fail(set(PINNED_FILES) == set(pins), "Admission candidate_pins must bind every required candidate file exactly")
    for rel in PINNED_FILES:
        path = candidate / rel
        fail(path.is_file() and digest(path) == pins[rel], f"Candidate pin mismatch: {rel}")

    proposal = read_json(candidate / "proposal.json")
    cases = read_json(candidate / "cases.json")
    command_doc = read_json(candidate / "command.json")
    forecast = read_json(candidate / "forecast.json")
    sources = read_json(candidate / "source-pins.json")["files"]
    runtime = read_json(candidate / "runtime-pins.json")
    required_sources = {
        "fixture",
        "prior_full_512",
        "prior_full_load",
        "trace_prep_proxy_log",
        "batch_runner",
        "probe_runner",
    }
    fail(
        set(sources) == required_sources,
        "Source pin inventory must bind all required fixture/baseline/load/runner files exactly",
    )
    for key, val in sources.items():
        sp = resolved(repo, val["path"])
        fail(sp.is_file() and digest(sp) == val["sha256"], f"Source pin mismatch: {key}")

    fail(len(cases) == 1, "Expected exactly one Matplotlib extension case")
    job = cases[0]
    fail(
        job.get("branches") == ["chosen"] and job.get("free_run_tokens") == 1024 and job.get("free_run_trials") == 1,
        "Case manifest does not match chosen64/seed43/cap1024 contract",
    )
    fail(job.get("case") == sources["fixture"]["path"], "Case manifest does not point at the pinned fixture")
    fixture = resolved(repo, job["case"])
    fixture_data = read_json(fixture)
    fail(len(fixture_data.get("prompt_token_ids", [])) == 4096, "Fixture is not exact suffix4096")
    fail(len(fixture_data.get("continuations", {}).get("chosen", [])) == 64, "Fixture chosen branch is not 64 tokens")
    fail(fixture_data.get("metadata", {}).get("seed") == 43, "Fixture seed is not 43")
    expected_runtime = runtime["expected_runtime_precision"]
    prior_result = read_json(resolved(repo, sources["prior_full_512"]["path"]))
    fail(prior_result.get("case") == job["case"], "Prior512 result is not from the exact extension fixture")
    fail(
        prior_result.get("runtime_precision") == expected_runtime,
        "Prior512 runtime policy differs from pinned full policy",
    )
    prior_load = read_json(resolved(repo, sources["prior_full_load"]["path"]))
    fail(
        prior_load.get("build_kwargs") == read_json(resolved(repo, runtime["build_kwargs_path"])),
        "Prior load used different full build kwargs",
    )
    expected_runtime_hash = hashlib.sha256(
        json.dumps(expected_runtime, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fail(expected_runtime_hash == runtime["expected_runtime_precision_sha256"], "Expected runtime policy hash mismatch")
    build_kwargs = read_json(resolved(repo, runtime["build_kwargs_path"]))
    fail(
        digest(resolved(repo, runtime["build_kwargs_path"])) == runtime["build_kwargs_sha256"],
        "Build kwargs hash mismatch",
    )
    cfg_path = resolved(repo, build_kwargs["precision_config"])
    fail(
        str(cfg_path.relative_to(repo.resolve())) == runtime["precision_config_path"],
        "Build kwargs do not select pinned full precision config",
    )
    fail(digest(cfg_path) == runtime["precision_config_sha256"], "Precision config hash mismatch")
    fail(
        expected_runtime.get("config_id") == runtime["expected_config_id"] == "stage12_bf16_hifi4_diagnostic",
        "Unexpected full runtime policy",
    )

    argv = command_doc["argv"]
    fail(
        argv[0] == "env",
        "Command must start with env; timeout is enforced inside the driver so completion survives timeout",
    )
    fail("HF_HUB_OFFLINE=1" in argv and "HF_DATASETS_OFFLINE=1" in argv, "HF offline environment is required")
    fail("READINESS_TRACE_REGION_SIZE=134217728" in argv, "Readiness trace region differs from controls")
    fail(any(x.endswith("tools.precision_repro_batch") for x in argv), "Unexpected batch runner")
    fail(
        argv[argv.index("--cases") + 1] == str((candidate / "cases.json").resolve()),
        "Command does not use pinned cases.json",
    )
    fail(
        argv[argv.index("--build-kwargs-file") + 1] == str(resolved(repo, runtime["build_kwargs_path"])),
        "Command does not use pinned full build kwargs",
    )
    outdir = Path(argv[argv.index("--output-dir") + 1]).resolve()
    fail(outdir == (candidate / "runs/full").resolve(), "Command output path differs from candidate output")
    fail(not outdir.exists(), f"Refusing output overwrite; output directory exists: {outdir}")
    expected_output = Path(command_doc.get("expected_output", ""))
    if not expected_output.is_absolute():
        expected_output = repo / expected_output
    expected_output = expected_output.resolve()
    fail(expected_output == (outdir / (job["name"] + ".json")).resolve(), "Expected output path mismatch")

    review = admission.get("candidate_forecast_review", {})
    review_path = resolved(repo, review.get("path", ""))
    fail(
        review_path.is_file() and digest(review_path) == review.get("sha256"),
        "Root candidate forecast review pin missing/mismatch",
    )
    review_data = read_json(review_path)
    forecast_sha = digest(candidate / "forecast.json")
    frange = forecast["calculation"]["forecast_range_wall_s"]
    fail(
        review_data.get("reviewer") == "root" and review_data.get("accepted") is True,
        "Candidate forecast lacks root acceptance",
    )
    fail(
        review_data.get("forecast_sha256") == forecast_sha and review_data.get("forecast_range_wall_s") == frange,
        "Root forecast review does not bind exact candidate forecast",
    )

    h9 = verify_h9_predecessor(repo, admission.get("h9_predecessor", {}))
    return (
        admission,
        argv,
        {
            "proposal": proposal,
            "forecast": forecast,
            "runtime": runtime,
            "job": job,
            "fixture": fixture_data,
            "prior": read_json(resolved(repo, sources["prior_full_512"]["path"])),
            "h9": h9,
            "forecast_review_sha256": review["sha256"],
        },
    )


def validate_outputs(candidate: Path, repo: Path, ctx: dict[str, Any]) -> dict[str, Any]:
    job = ctx["job"]
    outdir = candidate / "runs/full"
    result_path = outdir / (job["name"] + ".json")
    load_path = outdir / "load.json"
    fail(result_path.is_file(), "Batch child did not produce expected result JSON")
    fail(load_path.is_file(), "Batch child did not produce load.json")
    result = read_json(result_path)
    load = read_json(load_path)
    runtime = ctx["runtime"]
    fail(result.get("case") == job["case"], "Result case identity differs from exact fixture")
    fail(
        result.get("runtime_precision") == runtime["expected_runtime_precision"],
        "Result runtime_precision differs from entire pinned full policy",
    )
    chosen = result.get("continuations", {}).get("chosen", {})
    fail(
        chosen.get("tokens") == 64 and len(chosen.get("positions", [])) == 64,
        "Result chosen continuation is not exactly64 positions",
    )
    fail(
        chosen.get("conditioning_token_ids") == ctx["fixture"]["prompt_token_ids"],
        "Chosen continuation conditioning IDs differ from pinned fixture",
    )
    trials = result.get("free_running", [])
    fail(len(trials) == 1, "Expected one free-running trial")
    trial = trials[0]
    ids = trial.get("token_ids", [])
    prior_rows = [x for x in ctx["prior"].get("free_running", []) if x.get("sampling", {}).get("seed") == 43]
    fail(len(prior_rows) == 1, "Pinned prior result lacks unique seed43 row")
    prior_ids = prior_rows[0].get("token_ids", [])
    fail(len(prior_ids) == 512, "Pinned prior result is not exactly512 generated IDs")
    fail(
        len(ids) >= 512 and ids[:512] == prior_ids,
        "Extension criterion failed: require len>=512 and exact equality of the entire prior512-token prefix",
    )
    fail(len(ids) <= 1024, "Result exceeded requested cap1024")
    expected_sampling = ctx["fixture"]["metadata"]["sampling"]
    fail(
        trial.get("sampling") == dict(expected_sampling, seed=43),
        "Free-run sampling settings differ from seed43 fixture",
    )
    fail(
        load.get("build_kwargs") == read_json(resolved(repo, runtime["build_kwargs_path"])),
        "Load record used different build kwargs",
    )
    return {
        "result_path": str(result_path),
        "result_sha256": digest(result_path),
        "result_tokens": len(ids),
        "load_path": str(load_path),
        "load_sha256": digest(load_path),
        "runtime_precision": result["runtime_precision"],
        "prefix_identity_tokens": 512,
    }


def is_hf_reference_process(args: str) -> bool:
    """Recognize both supported CPU-HF runners that must serialize against TT work."""
    return "precision_repro_hf.py" in args or "actual_branch_logits.py" in args


def active_hf_processes() -> list[dict[str, Any]]:
    found = []
    proc = Path("/proc")
    if not proc.exists():
        return [{"error": "/proc unavailable; cannot establish no-HF-process gate"}]
    for ent in proc.iterdir():
        if not ent.name.isdigit() or int(ent.name) == os.getpid():
            continue
        try:
            args = (ent / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if is_hf_reference_process(args):
            found.append({"pid": int(ent.name), "cmdline": args[:400]})
    return found


def device_owners() -> list[Any]:
    """Import the owner-query helper only after explicit admission and --execute."""
    sys.path.insert(0, str(REPO_ROOT / "models/autoports/google_gemma_4_31b_it/tools"))
    import benchmark_server  # noqa: PLC0415

    return benchmark_server.device_owners()


def write_json_exclusive(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def execute_candidate(
    candidate: Path = CANDIDATE,
    repo: Path = REPO_ROOT,
    child_runner: Callable[..., Any] | None = None,
    hf_checker: Callable[[], list[Any]] = active_hf_processes,
    owner_checker: Callable[[], list[Any]] = device_owners,
    now_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    admission, argv, ctx = verify_candidate(candidate, repo)
    fail(not (candidate / "STOP_BEFORE_LAUNCH").exists(), "STOP_BEFORE_LAUNCH is present")
    now = now_fn()
    fail(
        now + REQUIRED_REMAINING_SECONDS <= DEADLINE,
        "Less than timeout+60 seconds remain before the declared stage deadline",
    )
    hf = hf_checker()
    fail(not hf, f"Active HF reference process(es) found: {hf}")
    owners = owner_checker()
    fail(not owners, f"TT device owner(s) found: {owners}")

    launch_path = candidate / "launch.json"
    completion_path = candidate / "completion.json"
    log_path = candidate / "run.log"
    lock_path = candidate / "launch.lock"
    for path in (launch_path, completion_path, log_path, lock_path):
        fail(not path.exists(), f"Refusing duplicate/overwrite: {path}")
    # Create an exclusive launch lock before creating any run outputs.
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(f"pid={os.getpid()}\n")
        f.flush()
        os.fsync(f.fileno())

    outdir = candidate / "runs/full"
    outdir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    launch = {
        "status": "running",
        "start_unix": start,
        "timeout_seconds": TIMEOUT_SECONDS,
        "deadline_unix": DEADLINE,
        "candidate_admission_sha256": digest(candidate / "admission.json"),
        "candidate_pins": admission["candidate_pins"],
        "hf9_predecessor": ctx["h9"],
        "forecast_review_sha256": ctx["forecast_review_sha256"],
        "hf_processes": hf,
        "device_owners": owners,
        "command": argv,
    }
    write_json_exclusive(launch_path, launch)
    proc_exit: int | None = None
    child_error: str | None = None
    output_error: str | None = None
    outputs: dict[str, Any] | None = None
    log_hash: str | None = None
    try:
        run = child_runner or subprocess.run
        with log_path.open("x") as log:
            try:
                proc = run(argv, cwd=repo, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS, check=False)
                proc_exit = int(proc.returncode)
            except subprocess.TimeoutExpired as exc:
                child_error = f"TimeoutExpired after {TIMEOUT_SECONDS}s: {exc}"
            except Exception as exc:  # Preserve terminal evidence for spawn/child errors.
                child_error = f"{type(exc).__name__}: {exc}"
        log_hash = digest(log_path) if log_path.exists() else None
        try:
            outputs = validate_outputs(candidate, repo, ctx)
        except Exception as exc:
            output_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - start
        bounds = ctx["forecast"]["calculation"]["forecast_range_wall_s"]
        if proc_exit != 0 or child_error:
            status = "failed"
        elif output_error:
            status = "invalid_outputs"
        elif bounds[0] <= elapsed <= bounds[1]:
            status = "completed_forecast_hit"
        else:
            status = "completed_forecast_miss_requires_root_review"
        completion = {
            **launch,
            "status": status,
            "end_unix": time.time(),
            "elapsed_seconds": elapsed,
            "process_exit_code": proc_exit,
            "child_error": child_error,
            "output_validation_error": output_error,
            "outputs": outputs,
            "log_sha256": log_hash,
            "forecast_gate": {
                "status": "range_hit" if bounds[0] <= elapsed <= bounds[1] else "miss_requires_root_review",
                "measured_seconds": elapsed,
                "forecast_range_wall_s": bounds,
            },
        }
        write_json_exclusive(completion_path, completion)
        return completion
    except BaseException as exc:
        # Once launch.json exists, keep a completion record even if validation/writing fails.
        if not completion_path.exists():
            emergency = {
                **launch,
                "status": "driver_error",
                "end_unix": time.time(),
                "elapsed_seconds": time.time() - start,
                "process_exit_code": proc_exit,
                "child_error": child_error,
                "output_validation_error": output_error,
                "driver_error": f"{type(exc).__name__}: {exc}",
                "outputs": outputs,
                "log_sha256": log_hash,
            }
            try:
                write_json_exclusive(completion_path, emergency)
            except Exception:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Required to start the admitted single device command")
    args = parser.parse_args()
    if not args.execute:
        print("Inert candidate. No command was started; pass --execute only after root admission and review.")
        return 0
    try:
        result = execute_candidate()
    except Exception as exc:
        print(f"Preflight/driver refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("status", "process_exit_code", "output_validation_error", "elapsed_seconds", "forecast_gate")
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed_forecast_hit" else 3


if __name__ == "__main__":
    raise SystemExit(main())
