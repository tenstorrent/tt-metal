"""Snapshot reviewable diagnostic commands and CPU schema checks; never runs models."""
import datetime
import hashlib
import json
import pathlib
import statistics
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
A = D / "anomalies"
O = D / "setup/swe85-tt-execution-plan"
O.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, data):
    p = O / name
    p.write_text(json.dumps(data, indent=2) + "\n")
    p.chmod(0o600)


anomaly = []
for p in sorted((A / "swe85-precision").glob("*/canonicalprefix/cases.json")):
    for job in json.loads(p.read_text()):
        j = dict(job)
        j["name"] = "anomaly." + job["name"]
        j["branches"] = ["chosen"]
        anomaly.append(j)
for onset_file in [
    "swe85-pytest5787-controls/cases.json",
    "swe85-getall-call1-controls/cases.json",
    "swe85-sympy13480-controls/cases.json",
    "swe85-sphinx10323-controls/cases.json",
    "swe85-extra-contraction-controls/cases.json",
    "swe85-sympy13551-controls/cases.json",
    "swe85-18763-7432-controls/cases.json",
    "swe85-sympy13878-controls/cases.json",
    "swe85-sphinx11510-controls/cases.json",
    "swe85-sphinx11510-protocol-controls/cases.json",
    "swe85-sphinx11510-contraction101-controls/cases.json",
    "swe85-protocol-scan83-controls/cases.json",
    "swe85-requests5414-caret-controls/cases.json",
    "swe85-sympy14248-caret-controls/cases.json",
]:
    for job in json.loads((A / onset_file).read_text()):
        j = dict(job)
        j["name"] = "anomaly." + job["name"]
        j["branches"] = ["chosen"]
        anomaly.append(j)
# Authoritative coordinator-selected cases, never directory glob admission.
roster_path = D / "setup/swe85-full-reply-control-roster.json"
roster = json.loads(roster_path.read_text())
full = []
seen = set()
for row in roster["cases"]:
    p = pathlib.Path(row["case"])
    data = json.loads(p.read_text())
    key = (row["task"], row["call"])
    assert key not in seen, key
    assert data["metadata"]["task"] == row["task"] and data["metadata"]["call"] == row["call"]
    assert len(data["continuations"]["chosen"]) == row["chosen_tokens"]
    seen.add(key)
    full.append(
        {"name": "fullreply." + p.name.removesuffix(".case.private.json"), "case": str(p), "branches": ["chosen"]}
    )
assert len(full) == roster["case_count"]
# Candidates are not semantically validated merely because syntax/application passes.
candidates = []
candidate_rosters = {}
for kind, manifest in [
    ("wrong", A / "swe85-below-band/correct-actions/all-candidate-action-cases.json"),
    ("fetch", A / "swe85-below-band/fetch-controls/candidate-action-cases.json"),
]:
    candidate_rosters[kind] = {"path": str(manifest), "sha256": sha(manifest)}
    for row in json.loads(manifest.read_text()):
        p = pathlib.Path(row["case"])
        d = json.loads(p.read_text())
        candidates.append(
            {
                "case": str(p),
                "task": d["metadata"].get("task"),
                "name": row["name"],
                "kind": kind,
                "status": "deferred_requires_semantic_validation",
                "metadata_status": d["metadata"].get("correct_semantic_validation"),
            }
        )
assert len({r["case"] for r in candidates}) == len(candidates)
reconciliation = {
    "authoritative_roster": str(roster_path),
    "sha256": sha(roster_path),
    "snapshot_utc": roster["snapshot_utc"],
    "whole_reply_task_count": roster["task_count"],
    "whole_reply_case_count": roster["case_count"],
    "previous_whole_reply_count": 52,
    "duplicates_by_task_call": [],
    "excluded_previous_extras": [
        {
            "name": "fullreply.pytest-dev__pytest-5787.call1",
            "chosen_tokens": 9690,
            "reason": "Archival degenerate full reply; canonical onset case remains measured separately.",
        },
        {
            "name": "fullreply.pytest-dev__pytest-5787.call65",
            "chosen_tokens": 1907,
            "reason": "Archival anomaly full reply; canonical onset case remains measured separately.",
        },
    ],
    "scope": "Exactly authoritative complete chosen replies plus independently selected canonical onset controls (counts recorded separately). Extras are not silently included in whole-reply closure. No whole-reply truncation. Requests2931 capped anomaly reply excluded; semanticcall29 retained by authoritative roster.",
    "action_rosters": candidate_rosters,
    "action_counts": {
        kind: {
            "cases": sum(c["kind"] == kind for c in candidates),
            "tasks": len({c["task"] for c in candidates if c["kind"] == kind}),
        }
        for kind in candidate_rosters
    },
    "reviewed_task_snapshot": len(list((D / "current/swe-extended85-head-bfp8/live-tasks").glob("*/coordinator.json"))),
    "new_pass_no_control": "sphinx-doc__sphinx-9281",
    "semantic_admission": "All listed action candidates remain deferred until root admits reviewed semantic evidence; they are not in device execution manifest.",
}
write("roster-reconciliation.json", reconciliation)
rows = []
for group, jobs in [("anomaly", anomaly), ("fullreply", full)]:
    for j in jobs:
        p = pathlib.Path(j["case"])
        d = json.loads(p.read_text())
        prompt = len(d["prompt_token_ids"])
        tf = sum(len(d["continuations"][b]) for b in j["branches"])
        free = j.get("free_run_tokens", 0) * j.get("free_run_trials", 3)
        longest = prompt + max([len(d["continuations"][b]) for b in j["branches"]] + [j.get("free_run_tokens", 0)])
        rows.append(
            {
                "group": group,
                "name": j["name"],
                "case": str(p),
                "sha256": sha(p),
                "prompt_tokens": prompt,
                "teacher_forcing_positions": tf,
                "free_run_token_cap": free,
                "total_context_max": longest,
                "fits_config_context": longest <= 262144,
                "prefill_repetitions": len(j["branches"])
                + (j.get("free_run_trials", 3) if j.get("free_run_tokens") else 0),
            }
        )
assert all(r["fits_config_context"] for r in rows)
# Shorter prompts first gives an early diagnostic health check before large requests.
anomaly.sort(key=lambda j: len(json.loads(pathlib.Path(j["case"]).read_text())["prompt_token_ids"]))
full.sort(key=lambda j: len(json.loads(pathlib.Path(j["case"]).read_text())["prompt_token_ids"]))
combined = anomaly + full
write("anomaly-cases.json", anomaly)
write("whole-reply-cases.json", full)
write("combined-cases.json", combined)
write("deferred-action-candidates.json", candidates)
emp = {}
for mode in ["served", "full"]:
    p = A / "gpqa-extended-precision" / ("first-wave-" + mode)
    rates = []
    decode = []
    tt = []
    for f in p.glob("*.json"):
        d = json.loads(f.read_text())
        for c in d.get("continuations", {}).values():
            rates.append(c["tokens"] / c["diagnostic_wall_s_including_prefill"])
        for c in d.get("free_running", []):
            value = c.get("decode_timing", {}).get("decode_tok_s_user")
            if value:
                decode.append(value)
    emp[mode] = {
        "source": str(p),
        "load_seconds": json.loads((p / "load.json").read_text())["wall_s"],
        "short_prefix_tf_tokens_per_second_median_including_prefill": statistics.median(rates),
        "traced_free_decode_tokens_per_second_median": statistics.median(decode),
        "scope": "GPQA suffix512 controls; not a measured long-context SWE rate.",
    }
    tf = sum(r["teacher_forcing_positions"] for r in rows)
    free = sum(r["free_run_token_cap"] for r in rows)
    emp[mode]["decode_dominated_proxy_seconds"] = (
        emp[mode]["load_seconds"] + tf / statistics.median(rates) + free / statistics.median(decode)
    )
base = json.loads((A / "swe85-precision/call88/canonicalprefix/commands.json").read_text())["device_controls"]
commands = {}
validation = []
for mode, cmd in base.items():
    cmd = cmd[:]
    cmd[cmd.index("--cases") + 1] = str(O / "combined-cases.json")
    cmd[cmd.index("--output-dir") + 1] = str(A / "swe85-executed-controls" / mode)
    if "--compare-dir" in cmd:
        cmd[cmd.index("--compare-dir") + 1] = str(A / "swe85-executed-controls/served")
    commands[mode] = {
        "argv": ["timeout", "14400"] + cmd,
        "external_timeout_seconds": 14400,
        "timeout_policy": "Mandatory bounded device job per tt-device-usage. Automatic timeout ends the job at14400s; preserve partial outputs/progress and capture evidence before recovery. Do not manually SIGTERM an active job. Admit launch only if updated estimate fits the remaining phase budget. Runner has no stop-file support; no graceful-between-jobs stop facility is claimed.",
    }
    r = subprocess.run(cmd + ["--validate-only"], capture_output=True, text=True)
    validation.append({"mode": mode, "exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr})
    assert r.returncode == 0
configs = {}
for name in [
    "head_bfp8_build_kwargs.json",
    "head_bfp8_precision_config.json",
    "full_precision_build_kwargs.json",
    "full_precision_config.json",
]:
    p = D / "setup" / name
    configs[str(p)] = {"sha256": sha(p), "content": json.loads(p.read_text())}
summary = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "reviewable_CPU_only_not_executed",
    "roster_reconciliation": reconciliation,
    "anomaly_jobs": len(anomaly),
    "whole_reply_jobs": len(full),
    "deferred_action_candidates": len(candidates),
    "teacher_forcing_positions_per_precision": sum(r["teacher_forcing_positions"] for r in rows),
    "free_run_token_cap_per_precision": sum(r["free_run_token_cap"] for r in rows),
    "prefill_input_tokens_per_precision": sum(r["prompt_tokens"] * r["prefill_repetitions"] for r in rows),
    "max_context": max(r["total_context_max"] for r in rows),
    "measurements": emp,
    "cost_warning": "Decode proxy excludes long-prefix prefill scaling and extra compile/warmup. Not an admission-grade wall estimate; recalibrate after first long-prefix case. Native full262144 context remains configured. No result or performance claim.",
    "sequence": [
        "Wait until the entire current eval queue and task retries are finished; prevent new submissions. Save results/response accounting and coordinator evidence.",
        "Use existing benchmark_server.py --stop only after checking recorded owner and closed launchers. It checks PID/start ticks, foreign owners, three idle metrics polls and signals only idle readiness driver; waits180s for owners to release; refuses forced termination.",
        "Verify no device owners remain; follow tt-device-usage before any recovery. No reset authorized by this plan.",
        "Before launch, establish that the updated long-prefix estimate fits the remaining phase budget and14400s bound. Run served combined manifest serially (single model load) under mandatory timeout14400, then clean close and verify no owners. Treat the first long-prefix normal case as the cost pilot; do not add a benchmark or change the case.",
        "Run full combined manifest serially under mandatory timeout14400 with compare-dir pointed to complete served outputs. Partial/missing served outputs must be resolved before full. Retain stdout TF progress and completed-branch private result snapshots; timeout or hang requires evidence capture and skill recovery before another owner.",
        "Interpret anomalies semantically. Full agreement alone is not sufficient for a below-band precision closure without valid correct continuation probability. Promote only separately validated action cases into a regenerated manifest.",
        "If full still degenerates, use pinned HF controls (shortened only if anomaly persists) per skill. If full fixes disagreement, bisect before any proposed scored rerun. Restart selected serving config via established launcher only after clean diagnostic close.",
    ],
    "stop_argv": ["python", "models/autoports/google_gemma_4_31b_it/tools/benchmark_server.py", "--stop"],
    "commands": commands,
    "configs": configs,
    "runner_restrictions": [
        "batch_size1; P300x2 FABRIC_1D_RING; exactly one device owner workload at a time",
        "teacher forcing uses traced model with host full-vocabulary logit readbacks, untempered probabilities; not production sampling/performance",
        "free generation uses device canonical sampler, top_k20 (within32), temperature1/top_p.95 and diagnostic seeds42/43/44; native SWE seeds generally absent",
        "buffered=False; output_capacity1024 does not truncate long teacher forcing; generator checks prompt+requested output against262144; next_input disables EOS early termination",
        "whole replies are never cropped; source canonical reconstructions retain count/serialization uncertainty, especially malformed protocol and flattened reasoning",
        "canonical onset controls keep full reconstructed prefix; no suffix512 substitution unless validated to preserve anomaly; no shortened control implies native-prompt equivalence",
        "correct branches withheld: candidate source applicability, bash syntax and expected-output specs alone do not establish semantic validity",
    ],
    "schema_validation": validation,
    "case_inventory": rows,
}
write("plan.json", summary)
print(
    json.dumps(
        {
            k: summary[k]
            for k in [
                "anomaly_jobs",
                "whole_reply_jobs",
                "deferred_action_candidates",
                "teacher_forcing_positions_per_precision",
                "free_run_token_cap_per_precision",
                "prefill_input_tokens_per_precision",
                "max_context",
                "measurements",
            ]
        },
        indent=2,
    )
)
