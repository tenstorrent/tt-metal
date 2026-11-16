#!/usr/bin/env python3
"""Root-admitted, one-mode-at-a-time executor for actual-branch-logit captures.

Default behavior is inert. This wrapper never advances automatically from one
model arm to another. Root admission, terminal HF9 evidence, current owners,
the stage deadline, and prior forecast decisions are checked per invocation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "bringup/run_contract.json").is_file()), None)
if ROOT is None:
    raise RuntimeError("repository root resolution failed")
D = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
TOOL = ROOT / "models/autoports/google_gemma_4_31b_it/tools/actual_branch_logits.py"
PINS = HERE / "source-pins.json"
PROPOSAL = HERE / "proposal.json"
COMMANDS = HERE / "command.json"
FORECAST = HERE / "forecast.json"
BRANCHES = HERE / "branches.json"
BRANCH_PROVENANCE = HERE / "branch-provenance.json"
REREVIEW = HERE / "review-rereview.json"
ROOT_STATIC_REVIEW = HERE / "root-static-review.json"
ADMISSION = HERE / "root-admission.json"  # created only by root after review
STOP = HERE / "STOP_BEFORE_LAUNCH"
OUTPUTS = HERE / "outputs"
RUNS = HERE / "serial-runs"
H9 = D / "setup/swe85-conditional-hf/revision-adaptive-expanded-call91-8192-reviewed"
H9_ADMISSION = H9 / "admission.json"
H9_PROGRESS = H9 / "progress.json"
H9_ROOT_TERMINAL_REVIEW = H9 / "root-terminal-review.json"
STAGE_DEADLINE = "2026-11-16T01:15:25Z"
TIMEOUT_GRACE_S = 60
MIN_MEM_BYTES = 100 * 1024**3
MODES = ("hf", "tt-selected", "tt-full", "compare")
MODEL_ORDER = ("hf", "tt-selected", "tt-full")
PREDECESSORS = {"hf": (), "tt-selected": ("hf",), "tt-full": ("hf", "tt-selected"), "compare": MODEL_ORDER}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def candidate_pins():
    return {
        "tool_sha256": sha(TOOL),
        "source_pins_sha256": sha(PINS),
        "proposal_sha256": sha(PROPOSAL),
        "commands_sha256": sha(COMMANDS),
        "forecast_sha256": sha(FORECAST),
        "branches_sha256": sha(BRANCHES),
        "branch_provenance_sha256": sha(BRANCH_PROVENANCE),
        "review_rereview_sha256": sha(REREVIEW),
        "root_static_review_sha256": sha(ROOT_STATIC_REVIEW),
        "wrapper_sha256": sha(__file__),
    }


def candidate_static():
    proposal, commands, forecast, pins = map(read, (PROPOSAL, COMMANDS, FORECAST, PINS))
    require(proposal.get("status") == "candidate_only_not_admitted_not_run", "probe proposal is not candidate-only")
    require(
        commands.get("status") == "prepared_candidate_commands_unadmitted_unlaunched",
        "saved commands are not inert/unadmitted",
    )
    require(forecast.get("status") == "planning_forecast_candidate_only_not_measured", "forecast status mismatch")
    require(pins.get("status") == "source_pins_prepared_candidate_only", "source pin status mismatch")
    for rel, digest in pins.get("files", {}).items():
        path = ROOT / rel
        require(path.is_file() and sha(path) == digest, f"source pin mismatch: {rel}")
    for rel, digest in pins.get("candidate_files", {}).items():
        path = HERE / rel
        require(path.is_file() and sha(path) == digest, f"candidate input pin mismatch: {rel}")
    require(
        proposal.get("tool_sha256") == sha(TOOL)
        and proposal.get("source_pins_sha256") == sha(PINS)
        and proposal.get("command_sha256") == sha(COMMANDS)
        and proposal.get("forecast_sha256") == sha(FORECAST),
        "proposal frozen pin mismatch",
    )
    require(proposal.get("branches", {}).get("sha256") == sha(BRANCHES), "proposal branch pin mismatch")
    require(
        proposal.get("branch_provenance", {}).get("sha256") == sha(BRANCH_PROVENANCE),
        "proposal branch-provenance pin mismatch",
    )
    require(
        commands.get("validated_provenance", {}).get("branch_provenance_sha256") == sha(BRANCH_PROVENANCE),
        "command branch-provenance pin mismatch",
    )
    require(
        commands.get("serial_order")
        == ["hf", "tt-selected", "tt-full", "compare-hf-selected", "compare-hf-full", "compare-selected-full"],
        "saved serialized order changed",
    )
    command_rows = commands.get("commands", [])
    require(
        [row.get("name") for row in command_rows]
        == ["hf", "tt-selected", "tt-full", "compare-hf-selected", "compare-hf-full", "compare-selected-full"],
        "saved command names/order changed",
    )
    for row in command_rows:
        argv = row.get("argv")
        require(
            isinstance(argv, list) and argv[:2] == ["timeout", str(row.get("timeout_seconds"))],
            f"saved command timeout mismatch: {row.get('name')}",
        )
        require(
            "--execute" in argv and "--validate-only" not in argv,
            f"saved command is not an execution command: {row.get('name')}",
        )
        if row["name"] in MODEL_ORDER:
            require(
                Path(argv[argv.index("--case") + 1]).resolve()
                == (HERE / "fixture/anomaly.psf__requests-5414.call11.onset.suffix2048.private.json").resolve(),
                f"saved fixture path changed: {row.get('name')}",
            )
            require(
                Path(argv[argv.index("--branches") + 1]).resolve() == BRANCHES.resolve(),
                f"saved branch path changed: {row.get('name')}",
            )
            if row["name"].startswith("tt-"):
                require(
                    "READINESS_TRACE_REGION_SIZE=134217728" in argv,
                    f"TT command lacks the reviewed readiness trace region: {row['name']}",
                )
        else:
            require(
                Path(argv[argv.index("--left-report") + 1]).resolve().is_relative_to(OUTPUTS.resolve())
                and Path(argv[argv.index("--right-report") + 1]).resolve().is_relative_to(OUTPUTS.resolve()),
                f"comparison source path escapes output root: {row.get('name')}",
            )
        tool_index = next((i for i, arg in enumerate(argv) if Path(arg).name == "actual_branch_logits.py"), None)
        require(
            tool_index is not None and Path(argv[tool_index]).resolve() == TOOL.resolve(),
            f"saved tool path changed: {row.get('name')}",
        )
    by_mode = {name: next(row for row in command_rows if row["name"] == name) for name in MODEL_ORDER}
    compare_rows = [row for row in command_rows if row["name"].startswith("compare-")]
    require(len(compare_rows) == 3, "expected three saved comparison commands")
    for mode, row in by_mode.items():
        require(
            row.get("timeout_seconds") == forecast["per_command_seconds"][mode]["timeout"],
            f"forecast/command timeout mismatch: {mode}",
        )
        prefix = Path(row["argv"][row["argv"].index("--output-prefix") + 1]).resolve()
        require(prefix.is_relative_to(OUTPUTS.resolve()), f"model output escapes candidate outputs: {mode}")
    for row in compare_rows:
        dest = Path(row["argv"][row["argv"].index("--comparison-output") + 1]).resolve()
        require(dest.is_relative_to(OUTPUTS.resolve()), "comparison output escapes candidate outputs")
    rereview = read(REREVIEW)
    require(
        str(rereview.get("status", "")).startswith("rereview_candidate_only_not_admitted_not_run")
        and str(rereview.get("execution_status", "")).startswith("No admission or launch"),
        "independent rereview missing or stale",
    )
    for key, value in (
        ("tool_sha256", sha(TOOL)),
        ("proposal_sha256", sha(PROPOSAL)),
        ("branches_sha256", sha(BRANCHES)),
        ("branch_provenance_sha256", sha(BRANCH_PROVENANCE)),
        ("source_pins_sha256", sha(PINS)),
        ("command_sha256", sha(COMMANDS)),
        ("forecast_sha256", sha(FORECAST)),
    ):
        require(rereview.get("frozen_hashes", {}).get(key) == value, f"independent rereview pin mismatch: {key}")
    root_review = read(ROOT_STATIC_REVIEW)
    require(
        root_review.get("status") == "root_static_review_accepted_candidate_only"
        and root_review.get("reviewer") == "root"
        and root_review.get("accepted") is True,
        "root static review missing or not accepted",
    )
    require(
        root_review.get("proposal_sha256") == sha(PROPOSAL)
        and root_review.get("tool_sha256") == sha(TOOL)
        and root_review.get("independent_rereview_sha256") == sha(REREVIEW),
        "root static review pins are stale",
    )
    return {
        "proposal": proposal,
        "commands": command_rows,
        "forecast": forecast,
        "pins": pins,
        "candidate_pins": candidate_pins(),
    }


def verify_h9_terminal(admission, root=ROOT, h9=H9):
    evidence = admission.get("h9_terminal", {})
    require(evidence.get("admission_sha256") == sha(h9 / "admission.json"), "H9 admission hash mismatch")
    progress_path = h9 / "progress.json"
    require(evidence.get("progress_sha256") == sha(progress_path), "H9 progress hash mismatch")
    progress = read(progress_path)
    require(
        progress.get("status") not in {None, "group_running", "running", "pending", "failed"}
        and progress.get("active_group") is None,
        "H9 progress is not terminal",
    )
    groups = progress.get("groups", [])
    require(
        len(groups) == 8 and [g.get("serial_order") for g in groups] == list(range(1, 9)),
        "H9 progress does not contain all eight ordered groups",
    )
    review_pins = evidence.get("root_reviews", {})
    observed_reviews = {}
    for order, group in enumerate(groups, start=1):
        require(group.get("status") == "completed", f"H9 group {order} is not completed")
        completion_path = h9 / f"completion-group-{order:02}.json"
        launch_path = h9 / f"launch-group-{order:02}.json"
        require(completion_path.is_file() and launch_path.is_file(), f"H9 group {order} terminal files missing")
        require(
            group.get("completion_sha256") == sha(completion_path) and group.get("launch_sha256") == sha(launch_path),
            f"H9 group {order} completion/launch hash mismatch",
        )
        completion = read(completion_path)
        launch = read(launch_path)
        require(
            completion.get("exit_code") == 0
            and launch.get("candidate_admission_sha256") == progress.get("candidate_admission_sha256"),
            f"H9 group {order} did not complete successfully",
        )
        review_path = h9 / f"root-group-{order:02}-review.json"
        review_hash = sha(review_path) if review_path.is_file() else None
        require(
            review_hash is not None and review_pins.get(f"{order:02}") == review_hash,
            f"H9 root review {order} missing or hash mismatch",
        )
        review = read(review_path)
        require(
            review.get("reviewer") == "root"
            and review.get("accepted") is True
            and review.get("serial_order") == order
            and review.get("completion_sha256") == group.get("completion_sha256")
            and review.get("launch_sha256") == group.get("launch_sha256"),
            f"H9 root review {order} does not accept matching completion",
        )
        observed_reviews[f"{order:02}"] = review_hash
    require(observed_reviews == review_pins, "H9 root review set differs from root admission")
    terminal_evidence = evidence.get("root_terminal_review", {})
    terminal_review_path = h9 / "root-terminal-review.json"
    require(
        terminal_evidence.get("path") == str(terminal_review_path)
        and terminal_evidence.get("sha256") == sha(terminal_review_path),
        "H9 terminal root-review hash is missing or stale in root admission",
    )
    terminal = read(terminal_review_path)
    admission_sha = sha(h9 / "admission.json")
    require(
        terminal.get("reviewer") == "root" and terminal.get("accepted") is True,
        "H9 terminal root review is not accepted by root",
    )
    require(
        terminal.get("progress_sha256") == sha(progress_path)
        and terminal.get("admission_sha256") == admission_sha
        and terminal.get("candidate_admission_sha256") == admission_sha,
        "H9 terminal root review does not bind progress/admission identity",
    )
    expected_launches = {
        str((h9 / f"launch-group-{order:02}.json").relative_to(root)): group["launch_sha256"]
        for order, group in enumerate(groups, start=1)
    }
    expected_completions = {
        str((h9 / f"completion-group-{order:02}.json").relative_to(root)): group["completion_sha256"]
        for order, group in enumerate(groups, start=1)
    }
    expected_outputs = {path: digest for group in groups for path, digest in group.get("results", {}).items()}
    require(
        terminal.get("root_group_reviews") == observed_reviews
        and terminal.get("launch_sha256") == expected_launches
        and terminal.get("completion_sha256") == expected_completions
        and terminal.get("output_sha256") == expected_outputs,
        "H9 terminal root review's group/output hash inventories differ",
    )
    require(
        evidence.get("root_terminal_review_sha256") == sha(terminal_review_path),
        "H9 terminal root-review hash alias is stale in root admission",
    )
    return {
        "progress_sha256": sha(progress_path),
        "admission_sha256": sha(h9 / "admission.json"),
        "root_reviews": observed_reviews,
        "root_terminal_review_sha256": sha(terminal_review_path),
        "group_count": 8,
    }


def verify_root_admission(mode, now=None):
    require(ADMISSION.is_file(), "root-admission.json is required; this candidate is not yet admitted")
    admission = read(ADMISSION)
    require(
        admission.get("status") == "root_admitted_serial_probe"
        and admission.get("reviewer") == "root"
        and admission.get("accepted") is True,
        "root admission status/reviewer mismatch",
    )
    require(mode in admission.get("authorized_modes", []), f"root admission does not authorize mode {mode}")
    spec = candidate_static()
    for key, digest in spec["candidate_pins"].items():
        require(admission.get("candidate_pins", {}).get(key) == digest, f"root admission pin mismatch: {key}")
    h9 = verify_h9_terminal(admission)
    deadline_text = admission.get("stage_deadline_utc")
    require(deadline_text == STAGE_DEADLINE, "root admission stage deadline mismatch")
    deadline = dt.datetime.fromisoformat(deadline_text.replace("Z", "+00:00")).timestamp()
    now = time.time() if now is None else now
    commands = spec["commands"]
    selected = (
        [row for row in commands if row["name"].startswith("compare-")]
        if mode == "compare"
        else [next(row for row in commands if row["name"] == mode)]
    )
    required_seconds = sum(row["timeout_seconds"] + TIMEOUT_GRACE_S for row in selected)
    check_deadline_budget(now, deadline, required_seconds)
    return {
        "admission_sha256": sha(ADMISSION),
        "h9_terminal": h9,
        "stage_deadline_utc": deadline_text,
        "remaining_seconds": deadline - now,
        "required_seconds": required_seconds,
        "spec": spec,
    }


def check_deadline_budget(now, deadline, required_seconds):
    require(
        required_seconds > 0 and now + required_seconds <= deadline,
        "hard timeout plus 60-second grace exceeds stage deadline",
    )
    return deadline - now


def process_snapshot(proc_root=Path("/proc")):
    hf_processes, owners = [], []
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            args = ""
        if (
            "precision_repro_hf.py" in args
            or "actual_branch_logits.py" in args
            or "run_serial_probe.py --execute" in args
        ):
            hf_processes.append({"pid": int(proc.name), "cmdline": args[:300]})
        for fd in (proc / "fd").glob("*"):
            try:
                if os.readlink(fd).startswith("/dev/tenstorrent/"):
                    owners.append({"pid": int(proc.name), "cmdline": args[:300]})
                    break
            except OSError:
                continue
    return hf_processes, owners


def fuser_probe(runner=subprocess.run):
    nodes = sorted(path for path in Path("/dev/tenstorrent").glob("*") if path.exists())
    require(nodes, "no /dev/tenstorrent nodes available for required fuser preflight")
    result = runner(["fuser", "-v", *map(str, nodes)], capture_output=True, text=True, check=False)
    output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    if result.returncode == 0:
        return {"active": True, "output": output}
    require(result.returncode == 1, f"fuser preflight failed ({result.returncode}): {output}")
    return {"active": False, "output": output}


def available_memory(meminfo=Path("/proc/meminfo")):
    for line in Path(meminfo).read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def resource_preflight(mode, hf_processes, owners, memory, fuser_result=None):
    require(not hf_processes, f"active HF/logit probe process blocks run: {hf_processes}")
    require(not owners, f"active TT device owner blocks run: {owners}")
    if mode.startswith("tt-"):
        require(
            fuser_result is not None and fuser_result.get("active") is False,
            "TT arm requires a clean fuser preflight before launch",
        )
    if mode == "hf":
        require(
            memory is not None and memory >= MIN_MEM_BYTES, f"HF arm requires MemAvailable >=100 GiB; observed {memory}"
        )
    return {
        "active_hf_or_probe_processes": hf_processes,
        "proc_device_owners": owners,
        "fuser": fuser_result,
        "mem_available_bytes": memory,
        "required_mem_available_bytes_for_hf": MIN_MEM_BYTES if mode == "hf" else None,
    }


def check_prior_forecasts(mode, record_dir=RUNS):
    records = {}
    for previous in PREDECESSORS[mode]:
        completion_path = record_dir / f"{previous}.completion.json"
        require(completion_path.is_file(), f"prior model arm has no completion: {previous}")
        completion = read(completion_path)
        require(completion.get("status") == "completed", f"prior model arm failed: {previous}")
        gate = completion.get("forecast_gate", {})
        if gate.get("status") == "range_hit":
            records[previous] = {"completion_sha256": sha(completion_path), "accepted_miss": None}
            continue
        require(
            gate.get("status") == "root_acceptance_required_before_next_model_arm",
            f"prior model arm has invalid forecast gate: {previous}",
        )
        accepted_path = record_dir / f"forecast-miss-accepted-{previous}.json"
        require(accepted_path.is_file(), f"root forecast-miss acceptance is required before next arm: {previous}")
        accepted = read(accepted_path)
        require(
            accepted.get("reviewer") == "root"
            and accepted.get("accepted") is True
            and accepted.get("status") == "root_accepted_forecast_miss"
            and accepted.get("completion_sha256") == sha(completion_path),
            f"root forecast-miss acceptance does not bind {previous} completion",
        )
        records[previous] = {"completion_sha256": sha(completion_path), "accepted_miss_sha256": sha(accepted_path)}
    return records


def ensure_outputs_fresh(mode, command_rows):
    for row in command_rows:
        argv = row["argv"]
        if mode == "compare":
            if not row["name"].startswith("compare-"):
                continue
        elif row["name"] != mode:
            continue
        if row["name"] in MODEL_ORDER:
            prefix = Path(argv[argv.index("--output-prefix") + 1])
            for suffix in (".logits.pt", ".report.json"):
                require(
                    not prefix.with_suffix(suffix).exists(), f"refusing output overwrite: {prefix.with_suffix(suffix)}"
                )
        else:
            dest = Path(argv[argv.index("--comparison-output") + 1])
            require(not dest.exists(), f"refusing comparison overwrite: {dest}")


def _check_row_identity(report, branches):
    require(report.get("branches") == branches, "report branches differ from exact pinned branch file")
    rows_by_branch = report.get("rows", {})
    require(set(rows_by_branch) == set(branches), "report branch row names mismatch")
    prompt_length = report.get("source_prompt_tokens")
    require(prompt_length == 2048, "report source prompt length mismatch")
    fixture = read(HERE / "fixture/anomaly.psf__requests-5414.call11.onset.suffix2048.private.json")
    prompt_last = fixture["prompt_token_ids"][-1]
    for name, branch in branches.items():
        rows = rows_by_branch[name]
        require(len(rows) == 7, f"report must contain seven rows for {name}")
        expected_inputs = [prompt_last, *branch]
        for index, row in enumerate(rows):
            require(
                row.get("row_index") == index and row.get("absolute_input_position") == 2047 + index,
                f"row index/position mismatch for {name}:{index}",
            )
            require(row.get("input_token_id") == expected_inputs[index], f"input token mismatch for {name}:{index}")
            require(
                row.get("history_after_input") == [prompt_last, *branch[:index]],
                f"forced history mismatch for {name}:{index}",
            )
            expected_next = branch[index] if index < 6 else None
            require(
                row.get("expected_next_token_id") == expected_next, f"expected next token mismatch for {name}:{index}"
            )
            summary = row.get("summary", {})
            require(
                isinstance(summary.get("vocab_size"), int) and summary["vocab_size"] > max(branch),
                f"logit summary vocabulary mismatch for {name}:{index}",
            )
            require(len(summary.get("top32", [])) == 32, f"top32 missing for {name}:{index}")


def validate_runtime_precision(report, mode, build_kwargs):
    config_path = ROOT / read(build_kwargs)["precision_config"]
    expected = read(config_path)
    require(report.get("precision") == expected, f"{mode} runtime precision policy differs from the exact saved config")
    require(
        report.get("precision_config_path") == str(config_path.resolve())
        and report.get("precision_config_sha256") == sha(config_path),
        f"{mode} precision config path/hash mismatch",
    )


def validate_capture(mode, argv, torch_load=None):
    fixture_path = HERE / "fixture/anomaly.psf__requests-5414.call11.onset.suffix2048.private.json"
    fixture = read(fixture_path)
    branches = read(BRANCHES)
    prefix = Path(argv[argv.index("--output-prefix") + 1]).resolve()
    report_path, tensor_path = prefix.with_suffix(".report.json"), prefix.with_suffix(".logits.pt")
    require(report_path.is_file() and tensor_path.is_file(), "capture report or tensor file is missing")
    report = read(report_path)
    require(
        report.get("status") == "completed_logits_capture" and report.get("mode") == mode,
        "capture report status/mode mismatch",
    )
    require(
        report.get("case_sha256") == sha(fixture_path)
        and report.get("source_prompt_tokens") == 2048
        and report.get("prompt_token_ids_sha256")
        == hashlib.sha256(json.dumps(fixture["prompt_token_ids"], separators=(",", ":")).encode()).hexdigest(),
        "capture report fixture/prompt identity mismatch",
    )
    canonical_branches_hash = hashlib.sha256(
        json.dumps(branches, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        report.get("hf_revision") == "842da3794eaa0b77d5f08bae87a17459d91ff475"
        and report.get("branches_sha256") == canonical_branches_hash,
        "capture report revision/branch hash mismatch",
    )
    require(
        Path(report.get("logits_pt", "")).resolve() == tensor_path.resolve()
        and report.get("logits_pt_sha256") == sha(tensor_path),
        "capture report tensor path/hash mismatch",
    )
    require(
        (report_path.stat().st_mode & 0o077) == 0 and (tensor_path.stat().st_mode & 0o077) == 0,
        "capture report/logits must remain private mode 0600",
    )
    if mode == "hf":
        require(
            report.get("precision", {}).get("reference") == "pinned Hugging Face CPU bfloat16",
            "HF report reference/device/dtype mismatch",
        )
    else:
        build_kwargs = Path(argv[argv.index("--build-kwargs-file") + 1])
        require(report.get("build_kwargs_sha256") == sha(build_kwargs), "TT report build-kwargs hash mismatch")
        validate_runtime_precision(report, mode, build_kwargs)
    _check_row_identity(report, branches)
    if torch_load is None:
        import torch

        torch_load = torch.load
    blob = torch_load(tensor_path, map_location="cpu", weights_only=True)
    require(
        blob.get("format") == "actual-branch-logits-v1"
        and blob.get("mode") == mode
        and blob.get("case_sha256") == sha(fixture_path)
        and blob.get("revision") == "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "saved logits tensor identity mismatch",
    )
    require(set(blob.get("rows", {})) == set(branches), "saved tensor branch names mismatch")
    for name in branches:
        tensor = blob["rows"][name]
        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else None
        expected_vocab = report["rows"][name][0]["summary"]["vocab_size"]
        require(shape == (7, expected_vocab), f"saved tensor shape mismatch for {name}: {shape}")
    return {
        "report_path": str(report_path),
        "report_sha256": sha(report_path),
        "tensor_path": str(tensor_path),
        "tensor_sha256": sha(tensor_path),
        "branch_names": list(branches),
        "rows_per_branch": 7,
    }


def validate_comparisons(command_rows):
    branches = read(BRANCHES)
    expected_names = {
        "hf-selected": ("hf", "tt-selected"),
        "hf-full": ("hf", "tt-full"),
        "selected-full": ("tt-selected", "tt-full"),
    }
    checked = {}
    for row in command_rows:
        if not row["name"].startswith("compare-"):
            continue
        suffix = row["name"].removeprefix("compare-")
        left_mode, right_mode = expected_names[suffix]
        path = Path(row["argv"][row["argv"].index("--comparison-output") + 1])
        require(path.is_file(), f"comparison output missing: {path}")
        report = read(path)
        require(
            report.get("status") == "saved_logits_comparison"
            and report.get("case_sha256") == read(PROPOSAL)["fixture"]["sha256"],
            f"comparison identity mismatch: {suffix}",
        )
        for mode in (left_mode, right_mode):
            candidate = Path(
                next(
                    item["argv"][item["argv"].index("--output-prefix") + 1]
                    for item in command_rows
                    if item["name"] == mode
                )
            ).with_suffix(".report.json")
            require(
                report.get("left_report_sha256" if mode == left_mode else "right_report_sha256") == sha(candidate),
                f"comparison source report pin mismatch: {suffix}:{mode}",
            )
            tensor = Path(read(candidate).get("logits_pt", ""))
            require(
                tensor.is_file() and tensor.resolve().is_relative_to(OUTPUTS.resolve()),
                f"comparison tensor path missing or outside outputs: {mode}",
            )
            require(
                report.get("left_logits_sha256" if mode == left_mode else "right_logits_sha256") == sha(tensor),
                f"comparison source tensor pin mismatch: {suffix}:{mode}",
            )
        comparisons = report.get("comparisons", {})
        require(set(comparisons) == set(branches), f"comparison branch names mismatch: {suffix}")
        for name in branches:
            require(len(comparisons[name]) == 7, f"comparison row count mismatch: {suffix}:{name}")
        checked[suffix] = {"path": str(path), "sha256": sha(path), "branches": list(comparisons)}
    require(len(checked) == 3, "all three saved-logit comparisons were not validated")
    return checked


def run_attempt(
    mode,
    commands,
    forecast,
    preflight,
    admission_sha,
    h9_evidence,
    prior_forecasts,
    child_runner=subprocess.run,
    validator=None,
    record_dir=RUNS,
    now_fn=time.time,
):
    """Execute one explicit arm and always persist launch/completion after gates."""
    launch_path = record_dir / f"{mode}.launch.json"
    completion_path = record_dir / f"{mode}.completion.json"
    require(not launch_path.exists() and not completion_path.exists(), f"{mode} already has terminal records")
    started = now_fn()
    terminal = {
        "status": "preflight_failed",
        "mode": mode,
        "started_unix": started,
        "child_exit_codes": [],
        "driver_exit_code": 2,
        "candidate_admission_sha256": admission_sha,
        "h9_terminal_evidence": h9_evidence,
        "resource_preflight": preflight,
        "prior_forecast_gates": prior_forecasts,
    }
    launch_record = {
        "status": "launched",
        "mode": mode,
        "started_unix": started,
        "candidate_admission_sha256": admission_sha,
        "h9_terminal_evidence": h9_evidence,
        "resource_preflight": preflight,
        "commands": [
            {"name": row["name"], "argv": row["argv"], "timeout_seconds": row["timeout_seconds"]} for row in commands
        ],
    }
    launched = False
    try:
        write_exclusive(launch_path, launch_record)
        launched = True
        terminal["launch_sha256"] = sha(launch_path)
        terminal["status"] = "running"
        logs = []
        for row in commands:
            name = row["name"]
            stdout_path = record_dir / f"{mode}.{name}.stdout.log"
            stderr_path = record_dir / f"{mode}.{name}.stderr.log"
            try:
                with open(stdout_path, "xb") as stdout, open(stderr_path, "xb") as stderr:
                    result = child_runner(
                        row["argv"],
                        cwd=ROOT,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=row["timeout_seconds"] + TIMEOUT_GRACE_S,
                        check=False,
                    )
                code = int(result.returncode)
            except subprocess.TimeoutExpired as exc:
                code = 124
                terminal.setdefault("child_errors", []).append(
                    {
                        "command": name,
                        "type": "TimeoutExpired",
                        "timeout_seconds": row["timeout_seconds"] + TIMEOUT_GRACE_S,
                    }
                )
            except Exception as exc:
                code = 127
                terminal.setdefault("child_errors", []).append(
                    {"command": name, "type": type(exc).__name__, "message": str(exc)}
                )
            logs.append(
                {
                    "command": name,
                    "stdout_sha256": sha(stdout_path) if stdout_path.is_file() else None,
                    "stderr_sha256": sha(stderr_path) if stderr_path.is_file() else None,
                }
            )
            terminal["child_exit_codes"].append({"command": name, "exit_code": code})
            if code != 0:
                terminal["status"] = "child_failed"
                terminal["driver_exit_code"] = code
                break
        terminal["logs"] = logs
        if terminal["status"] == "running":
            try:
                validation = validator() if validator else {"status": "not_configured"}
                terminal.update({"status": "completed", "output_validation": validation, "driver_exit_code": 0})
            except Exception as exc:
                terminal.update(
                    {
                        "status": "output_validation_failed",
                        "output_validation_error": f"{type(exc).__name__}: {exc}",
                        "driver_exit_code": 2,
                    }
                )
    except Exception as exc:
        terminal.update(
            {"status": "launch_or_wrapper_failed", "error": f"{type(exc).__name__}: {exc}", "driver_exit_code": 2}
        )
    finally:
        finished = now_fn()
        terminal["finished_unix"] = finished
        terminal["elapsed_seconds"] = max(0.0, finished - started)
        if launched and terminal.get("status") == "completed":
            lo, hi = forecast["range"]
            gate_status = (
                "range_hit"
                if lo <= terminal["elapsed_seconds"] <= hi
                else "root_acceptance_required_before_next_model_arm"
            )
            terminal["forecast_gate"] = {
                "status": gate_status,
                "measured_seconds": terminal["elapsed_seconds"],
                "range_seconds": [lo, hi],
                "root_acceptance_path": (
                    str(record_dir / f"forecast-miss-accepted-{mode}.json") if gate_status != "range_hit" else None
                ),
            }
        if not completion_path.exists():
            write_exclusive(completion_path, terminal)
    return int(terminal.get("driver_exit_code", 2)), terminal


def attempt_mode(mode, deps=None):
    deps = deps or {}
    static = (deps.get("candidate_static") or candidate_static)()
    admission = (deps.get("verify_admission") or verify_root_admission)(mode, now=deps.get("now"))
    require(not STOP.exists(), "STOP_BEFORE_LAUNCH is present")
    require(
        RUNS.exists() or not any((RUNS / f"{mode}.{suffix}").exists() for suffix in ("launch.json", "completion.json")),
        "mode already has run records",
    )
    RUNS.mkdir(parents=True, exist_ok=True)
    lock_path = RUNS / ".probe.lock"
    lock_handle = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another serial probe invocation holds the candidate lock") from exc
        static = (deps.get("candidate_static") or candidate_static)()
        admission = (deps.get("verify_admission") or verify_root_admission)(mode, now=deps.get("now"))
        prior = (deps.get("check_prior_forecasts") or check_prior_forecasts)(
            mode, record_dir=deps.get("record_dir", RUNS)
        )
        ensure_outputs_fresh(mode, static["commands"])
        hf_processes, owners = (deps.get("process_snapshot") or process_snapshot)()
        memory = (deps.get("available_memory") or available_memory)() if mode == "hf" else None
        fuser_result = None
        if mode.startswith("tt-"):
            fuser_result = (deps.get("fuser_probe") or fuser_probe)()
        preflight = resource_preflight(mode, hf_processes, owners, memory, fuser_result)
        selected_rows = (
            [row for row in static["commands"] if row["name"].startswith("compare-")]
            if mode == "compare"
            else [next(row for row in static["commands"] if row["name"] == mode)]
        )
        forecast_key = "three_comparisons" if mode == "compare" else mode
        forecast = static["forecast"]["per_command_seconds"][forecast_key]
        if mode == "compare":
            timeout_each = forecast["timeout_each"]
            for row in selected_rows:
                row["timeout_seconds"] = timeout_each
        validator = (
            (lambda: {"status": "saved_comparisons_validated", "comparisons": validate_comparisons(static["commands"])})
            if mode == "compare"
            else (
                lambda: {
                    "status": "capture_tensor_and_branch_identity_validated",
                    "capture": validate_capture(mode, selected_rows[0]["argv"]),
                }
            )
        )
        return run_attempt(
            mode,
            selected_rows,
            {"range": forecast["range"]},
            preflight,
            admission["admission_sha256"],
            admission["h9_terminal"],
            prior,
            child_runner=deps.get("child_runner", subprocess.run),
            validator=validator,
            record_dir=deps.get("record_dir", RUNS),
            now_fn=deps.get("now_fn", time.time),
        )
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def record_preflight_denial(mode, error, record_dir=RUNS):
    """Keep a denial audit without reserving a model-arm launch/completion slot."""
    record_dir = Path(record_dir)
    if (record_dir / f"{mode}.launch.json").exists() or (record_dir / f"{mode}.completion.json").exists():
        return None
    record_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = record_dir / f"{mode}.preflight-denied.{stamp}.json"
    write_exclusive(
        path,
        {
            "status": "preflight_denied_no_launch_slot_reserved",
            "mode": mode,
            "at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
            "runner_invoked": False,
        },
    )
    return path


def mock_tests():
    global OUTPUTS
    import tempfile

    # Terminal H9 schema and root reviews are checked without reading the live queue.
    with tempfile.TemporaryDirectory() as temp:
        h9 = Path(temp) / "h9"
        h9.mkdir()
        (h9 / "admission.json").write_text("{}\n")
        progress = {"status": "completed", "candidate_admission_sha256": "adm", "active_group": None, "groups": []}
        for i in range(1, 9):
            completion = {"exit_code": 0}
            launch = {"candidate_admission_sha256": "adm"}
            cp, lp = h9 / f"completion-group-{i:02}.json", h9 / f"launch-group-{i:02}.json"
            cp.write_text(json.dumps(completion))
            lp.write_text(json.dumps(launch))
            progress["groups"].append(
                {"serial_order": i, "status": "completed", "completion_sha256": sha(cp), "launch_sha256": sha(lp)}
            )
        pp = h9 / "progress.json"
        pp.write_text(json.dumps(progress))
        root_reviews = {}
        for i, group in enumerate(progress["groups"], start=1):
            review = {
                "reviewer": "root",
                "accepted": True,
                "serial_order": i,
                "completion_sha256": group["completion_sha256"],
                "launch_sha256": group["launch_sha256"],
            }
            rp = h9 / f"root-group-{i:02}-review.json"
            rp.write_text(json.dumps(review))
            root_reviews[f"{i:02}"] = sha(rp)
        terminal_review_path = h9 / "root-terminal-review.json"
        terminal_review = {
            "reviewer": "root",
            "accepted": True,
            "progress_sha256": sha(pp),
            "admission_sha256": sha(h9 / "admission.json"),
            "candidate_admission_sha256": sha(h9 / "admission.json"),
            "root_group_reviews": root_reviews,
            "launch_sha256": {
                str((h9 / f"launch-group-{i:02}.json").relative_to(Path(temp))): g["launch_sha256"]
                for i, g in enumerate(progress["groups"], start=1)
            },
            "completion_sha256": {
                str((h9 / f"completion-group-{i:02}.json").relative_to(Path(temp))): g["completion_sha256"]
                for i, g in enumerate(progress["groups"], start=1)
            },
            "output_sha256": {},
        }
        terminal_review_path.write_text(json.dumps(terminal_review))
        fake_admission = {
            "h9_terminal": {
                "admission_sha256": sha(h9 / "admission.json"),
                "progress_sha256": sha(pp),
                "root_reviews": root_reviews,
                "root_terminal_review": {"path": str(terminal_review_path), "sha256": sha(terminal_review_path)},
                "root_terminal_review_sha256": sha(terminal_review_path),
            }
        }
        checked_h9 = verify_h9_terminal(fake_admission, root=Path(temp), h9=h9)
        assert checked_h9["group_count"] == 8 and len(checked_h9["root_reviews"]) == 8
        progress["groups"][7]["status"] = "running"
        pp.write_text(json.dumps(progress))
        try:
            verify_h9_terminal(fake_admission, root=Path(temp), h9=h9)
        except RuntimeError:
            h9_running_rejected = True
        else:
            h9_running_rejected = False
        assert h9_running_rejected

    # Exercise the producer-facing capture and comparison validators with exact
    # fixture-bound IDs and report-declared tensor paths (without importing torch).
    class TensorStub:
        shape = (7, 237000)

    with tempfile.TemporaryDirectory() as temp:
        outdir = Path(temp)
        fixture_path = HERE / "fixture/anomaly.psf__requests-5414.call11.onset.suffix2048.private.json"
        fixture = read(fixture_path)
        branches = read(BRANCHES)
        vocab = 237000
        branch_rows = {}
        for name, branch in branches.items():
            branch_rows[name] = []
            for index in range(7):
                branch_rows[name].append(
                    {
                        "row_index": index,
                        "absolute_input_position": 2047 + index,
                        "input_token_id": [fixture["prompt_token_ids"][-1], *branch][index],
                        "history_after_input": [fixture["prompt_token_ids"][-1], *branch[:index]],
                        "expected_next_token_id": branch[index] if index < 6 else None,
                        "summary": {"vocab_size": vocab, "top32": [{"token_id": i} for i in range(32)]},
                    }
                )
        canonical_branch_hash = hashlib.sha256(
            json.dumps(branches, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        capture_tensor = outdir / "hf.logits.pt"
        capture_tensor.write_bytes(b"mock tensor bytes")
        capture_report_path = outdir / "hf.report.json"
        capture_report = {
            "status": "completed_logits_capture",
            "mode": "hf",
            "case_sha256": sha(fixture_path),
            "source_prompt_tokens": 2048,
            "prompt_token_ids_sha256": hashlib.sha256(
                json.dumps(fixture["prompt_token_ids"], separators=(",", ":")).encode()
            ).hexdigest(),
            "hf_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "branches_sha256": canonical_branch_hash,
            "branches": branches,
            "logits_pt": str(capture_tensor),
            "logits_pt_sha256": sha(capture_tensor),
            "precision": {"reference": "pinned Hugging Face CPU bfloat16"},
            "rows": branch_rows,
        }
        capture_report_path.write_text(json.dumps(capture_report))
        os.chmod(capture_report_path, 0o600)
        os.chmod(capture_tensor, 0o600)
        tensor_blob = {
            "format": "actual-branch-logits-v1",
            "mode": "hf",
            "case_sha256": sha(fixture_path),
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "rows": {name: TensorStub() for name in branches},
        }
        fake_load = lambda path, **kwargs: tensor_blob
        good_capture = validate_capture("hf", ["--output-prefix", str(outdir / "hf")], torch_load=fake_load)
        assert good_capture["rows_per_branch"] == 7
        wrong_report = dict(capture_report)
        wrong_rows = {name: [dict(row) for row in rows] for name, rows in branch_rows.items()}
        wrong_rows[next(iter(branches))][0]["input_token_id"] = 2047
        wrong_report["rows"] = wrong_rows
        capture_report_path.write_text(json.dumps(wrong_report))
        try:
            validate_capture("hf", ["--output-prefix", str(outdir / "hf")], torch_load=fake_load)
        except RuntimeError:
            bad_prefix_rejected = True
        else:
            bad_prefix_rejected = False
        assert bad_prefix_rejected

        # Three comparison outputs pin the report-declared tensor artifacts.
        old_outputs = OUTPUTS
        OUTPUTS = outdir
        try:
            report_paths, tensor_paths, command_rows = {}, {}, []
            for mode in MODEL_ORDER:
                rpt = outdir / f"{mode}.report.json"
                tensor = outdir / f"{mode}.logits.pt"
                tensor.write_bytes((mode + " tensor").encode())
                os.chmod(tensor, 0o600)
                rpt.write_text(json.dumps({"logits_pt": str(tensor)}))
                os.chmod(rpt, 0o600)
                report_paths[mode], tensor_paths[mode] = rpt, tensor
                command_rows.append({"name": mode, "argv": ["--output-prefix", str(outdir / mode)]})
            compare_specs = {
                "hf-selected": ("hf", "tt-selected"),
                "hf-full": ("hf", "tt-full"),
                "selected-full": ("tt-selected", "tt-full"),
            }
            for suffix, (left, right) in compare_specs.items():
                dest = outdir / f"compare-{suffix}.json"
                comparison = {
                    "status": "saved_logits_comparison",
                    "case_sha256": read(PROPOSAL)["fixture"]["sha256"],
                    "left_report_sha256": sha(report_paths[left]),
                    "right_report_sha256": sha(report_paths[right]),
                    "left_logits_sha256": sha(tensor_paths[left]),
                    "right_logits_sha256": sha(tensor_paths[right]),
                    "comparisons": {name: [{} for _ in range(7)] for name in branches},
                }
                dest.write_text(json.dumps(comparison))
                command_rows.append({"name": f"compare-{suffix}", "argv": ["--comparison-output", str(dest)]})
            checked = validate_comparisons(command_rows)
            assert len(checked) == 3
            for row in command_rows:
                if row["name"].startswith("compare-"):
                    Path(row["argv"][row["argv"].index("--comparison-output") + 1]).unlink()
            ensure_outputs_fresh("compare", command_rows)
            occupied = outdir / "compare-hf-selected.json"
            occupied.write_text("occupied")
            try:
                ensure_outputs_fresh("compare", command_rows)
            except RuntimeError:
                existing_compare_rejected = True
            else:
                existing_compare_rejected = False
            assert existing_compare_rejected
        finally:
            OUTPUTS = old_outputs

    for mode, kwargs_name in (
        ("tt-selected", "head_bfp8_build_kwargs.json"),
        ("tt-full", "full_precision_build_kwargs.json"),
    ):
        kwargs_path = ROOT / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals/setup" / kwargs_name
        expected_runtime = read(ROOT / read(kwargs_path)["precision_config"])
        validate_runtime_precision(
            {
                "precision": expected_runtime,
                "precision_config_path": str((ROOT / read(kwargs_path)["precision_config"]).resolve()),
                "precision_config_sha256": sha(ROOT / read(kwargs_path)["precision_config"]),
            },
            mode,
            kwargs_path,
        )
        try:
            validate_runtime_precision(
                {
                    "precision": {"config_id": "wrong"},
                    "precision_config_path": str((ROOT / read(kwargs_path)["precision_config"]).resolve()),
                    "precision_config_sha256": sha(ROOT / read(kwargs_path)["precision_config"]),
                },
                mode,
                kwargs_path,
            )
        except RuntimeError:
            runtime_policy_rejected = True
        else:
            runtime_policy_rejected = False
        assert runtime_policy_rejected

    def attempt(name, runner, validator=None):
        with tempfile.TemporaryDirectory() as temp:
            outdir = Path(temp)
            ticks = iter((100.0, 105.0))
            code, record = run_attempt(
                name,
                [{"name": name, "argv": ["host-mock"], "timeout_seconds": 10}],
                {"range": [1.0, 20.0]},
                {"mock": True},
                "admission",
                {},
                {},
                child_runner=runner,
                validator=validator or (lambda: {"ok": True}),
                record_dir=outdir,
                now_fn=lambda: next(ticks),
            )
            completion_exists = (outdir / f"{name}.completion.json").is_file()
            completion = read(outdir / f"{name}.completion.json") if completion_exists else None
            return code, record, completion

    class Completed:
        returncode = 0

    code, record, completion = attempt("success", lambda *a, **k: Completed())
    assert code == 0 and record["status"] == "completed" and completion["status"] == "completed"
    code, record, completion = attempt("child-error", lambda *a, **k: (_ for _ in ()).throw(OSError("spawn")))
    assert code == 127 and record["status"] == "child_failed" and completion["child_exit_codes"][0]["exit_code"] == 127
    code, record, completion = attempt(
        "child-timeout", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(["mock"], 70))
    )
    assert (
        code == 124 and record["status"] == "child_failed" and completion["child_errors"][0]["type"] == "TimeoutExpired"
    )
    code, record, completion = attempt(
        "bad-output", lambda *a, **k: Completed(), lambda: (_ for _ in ()).throw(ValueError("identity mismatch"))
    )
    assert (
        code == 2
        and record["status"] == "output_validation_failed"
        and "identity mismatch" in completion["output_validation_error"]
    )
    with tempfile.TemporaryDirectory() as temp:
        denied_dir = Path(temp)
        denial = record_preflight_denial("hf", RuntimeError("not admitted"), denied_dir)
        denial_record = read(denial)
        assert denial_record["status"] == "preflight_denied_no_launch_slot_reserved"
        assert denial_record["runner_invoked"] is False
        assert not (denied_dir / "hf.launch.json").exists()
        assert not (denied_dir / "hf.completion.json").exists()
    try:
        resource_preflight("tt-selected", [], [], None, {"active": True})
    except RuntimeError:
        fuser_gate = True
    else:
        fuser_gate = False
    assert fuser_gate
    try:
        resource_preflight("hf", [], [], MIN_MEM_BYTES - 1)
    except RuntimeError:
        mem_gate = True
    else:
        mem_gate = False
    assert mem_gate
    check_deadline_budget(100, 2560, 2460)
    try:
        check_deadline_budget(100, 2559, 2460)
    except RuntimeError:
        deadline_gate = True
    else:
        deadline_gate = False
    assert deadline_gate

    # A forecast miss blocks the next model arm until a root record binds its completion hash.
    with tempfile.TemporaryDirectory() as temp:
        run_dir = Path(temp)
        hf_completion = run_dir / "hf.completion.json"
        hf_completion.write_text(
            json.dumps(
                {"status": "completed", "forecast_gate": {"status": "root_acceptance_required_before_next_model_arm"}}
            )
        )
        try:
            check_prior_forecasts("tt-selected", run_dir)
        except RuntimeError:
            miss_blocked = True
        else:
            miss_blocked = False
        assert miss_blocked
        accepted_path = run_dir / "forecast-miss-accepted-hf.json"
        accepted_path.write_text(
            json.dumps(
                {
                    "reviewer": "root",
                    "accepted": True,
                    "status": "root_accepted_forecast_miss",
                    "completion_sha256": sha(hf_completion),
                }
            )
        )
        assert "hf" in check_prior_forecasts("tt-selected", run_dir)
    return {
        "child_success_completion": True,
        "child_spawn_error_completion": True,
        "child_timeout_completion": True,
        "invalid_output_completion": True,
        "fuser_and_hf_memory_gates": True,
        "eight_group_terminal_root_review_gate": True,
        "stage_deadline_and_forecast_miss_gates": True,
        "preflight_denial_logged_without_launch_slot": True,
        "capture_fixture_boundary_and_canonical_branch_hash": True,
        "comparison_report_declared_tensor_paths": True,
        "tt_runtime_precision_policy_match_and_mismatch": True,
        "comparison_freshness_ignores_existing_model_captures": True,
        "no_model_or_device_calls": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--mock-tests", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            require(args.mode is None, "--mode is only used with --execute")
            static = candidate_static()
            tests = mock_tests() if args.mock_tests else None
            print(
                json.dumps(
                    {
                        "status": "validated_inert_no_runner_no_model_no_device",
                        "candidate_pins": static["candidate_pins"],
                        "mode_count": len(MODES),
                        "h9_current_status_not_used_for_validation": read(H9_PROGRESS).get("status"),
                        "mock_tests": tests,
                    },
                    sort_keys=True,
                )
            )
            return 0
        require(
            args.mode is not None and not args.mock_tests,
            "--execute requires exactly one --mode; mock tests are validate-only",
        )
        code, result = attempt_mode(args.mode)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "status": result.get("status"),
                    "driver_exit_code": code,
                    "forecast_gate": result.get("forecast_gate"),
                },
                sort_keys=True,
            )
        )
        return code
    except Exception as exc:
        if args.execute and args.mode is not None:
            try:
                record_preflight_denial(args.mode, exc)
            except Exception as record_exc:
                print(f"run_serial_probe: also failed to record preflight denial: {record_exc}", file=sys.stderr)
        print(f"run_serial_probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
