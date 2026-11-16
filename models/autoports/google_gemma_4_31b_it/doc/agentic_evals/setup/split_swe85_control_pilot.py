"""CPU-only lossless split of an admitted control list; stages commands but executes none."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-pilot-split",
        type=Path,
        help="Prior hash-bound split whose exact pilot is already executing or measured.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[6]
    source = args.input_manifest.resolve()
    out = args.output_dir.resolve()
    raw = source.read_bytes()
    jobs = json.loads(raw)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Input must be a nonempty runner job list.")
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)), "Duplicate input job names"
    records = []
    paths = []
    for i, job in enumerate(jobs):
        path = Path(job["case"])
        path = path if path.is_absolute() else root / path
        path = path.resolve()
        payload = path.read_bytes()
        case = json.loads(payload)
        branches = job["branches"]
        assert branches and len(branches) == len(set(branches))
        lengths = {b: len(case["continuations"][b]) for b in branches}
        assert all(n > 0 for n in lengths.values())
        free = job.get("free_run_tokens", 0)
        trials = job.get("free_run_trials", 3) if free else 0
        assert free >= 0 and trials >= 0
        prompt = len(case["prompt_token_ids"])
        paths.append(str(path))
        records.append(
            {
                "input_index": i,
                "name": job["name"],
                "group": job["name"].split(".", 1)[0],
                "job_sha256": digest(canonical(job)),
                "case": str(path),
                "case_sha256": digest(payload),
                "prompt_tokens": prompt,
                "branches": lengths,
                "teacher_forcing_positions": sum(lengths.values()),
                "free_run_trials": trials,
                "free_run_token_cap": free * trials,
                "prefill_calls_proxy": len(branches) + trials,
                "prefill_input_tokens_proxy": prompt * (len(branches) + trials),
                "max_context_tokens": prompt + max(max(lengths.values()), free),
            }
        )
    assert len(paths) == len(set(paths)), "Duplicate input case paths"
    eligible = [r for r in records if r["group"] == "fullreply"]
    assert eligible, "No fullreply.* job eligible for pilot"
    # Longest conditioning input probes the unresolved long-context prefill cost.
    # Tie-break by context including full reply, then stable original order.
    selected = max(eligible, key=lambda r: (r["prompt_tokens"], r["max_context_tokens"], -r["input_index"]))
    index = selected["input_index"]
    pilot, rest = [jobs[index]], [j for i, j in enumerate(jobs) if i != index]
    reunited = {j["name"]: j for j in pilot + rest}
    assert len(reunited) == len(jobs) and all(reunited[j["name"]] == j for j in jobs)
    assert {j["name"] for j in pilot}.isdisjoint(j["name"] for j in rest)
    assert source.read_bytes() == raw
    assert all(digest(Path(r["case"]).read_bytes()) == r["case_sha256"] for r in records)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, obj):
        p = out / name
        data = (json.dumps(obj, indent=2) + "\n").encode()
        if p.exists():
            assert p.read_bytes() == data, f"Refusing to overwrite changed output: {p}"
        else:
            p.write_bytes(data)
            p.chmod(0o600)
        return {"path": str(p), "sha256": digest(data)}

    artifacts = {"pilot": save("pilot.json", pilot), "rest": save("rest.json", rest)}

    def counts(rows):
        groups = {}
        for row in rows:
            group = groups.setdefault(
                row["group"],
                {
                    k: 0
                    for k in [
                        "jobs",
                        "teacher_forcing_positions",
                        "free_run_token_cap",
                        "prefill_calls_proxy",
                        "prefill_input_tokens_proxy",
                    ]
                },
            )
            group["jobs"] += 1
            for k in group:
                if k != "jobs":
                    group[k] += row[k]
        totals = {
            k: sum(g[k] for g in groups.values())
            for k in [
                "jobs",
                "teacher_forcing_positions",
                "free_run_token_cap",
                "prefill_calls_proxy",
                "prefill_input_tokens_proxy",
            ]
        }
        return {"groups": groups, "total": totals}

    split_counts = {
        "pilot": counts([selected]),
        "rest": counts([r for r in records if r is not selected]),
        "input": counts(records),
    }
    assert all(
        split_counts["pilot"]["total"][k] + split_counts["rest"]["total"][k] == v
        for k, v in split_counts["input"]["total"].items()
    )
    d = root / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
    skill_runtime = Path(
        "/home/mvasiljevic/v5-agentic/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.27/runtime"
    )
    assert (skill_runtime / "readiness_check").is_dir(), "Pinned skill runtime unavailable"
    required_env = {"PYTHONPATH": str(root) + ":" + str(skill_runtime), "READINESS_TRACE_REGION_SIZE": "134217728"}
    commands = {}
    for part in ["pilot", "rest"]:
        commands[part] = {}
        for mode, cfg in [("served", "head_bfp8_build_kwargs.json"), ("full", "full_precision_build_kwargs.json")]:
            command = [
                "timeout",
                "14400",
                "env",
                *[k + "=" + v for k, v in required_env.items()],
                str(root / "python_env/bin/python"),
                "-m",
                "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
                "--cases",
                str(out / (part + ".json")),
                "--build-kwargs-file",
                str(d / "setup" / cfg),
                "--output-dir",
                str(out / "runs" / part / mode),
            ]
            if mode == "full":
                command += ["--compare-dir", str(out / "runs" / part / "served")]
            commands[part][mode] = {
                "cwd": str(root),
                "argv": command,
                "timeout_seconds": 14400,
                "required_environment": required_env,
                "build_kwargs_sha256": digest((d / "setup" / cfg).read_bytes()),
            }
    reuse = None
    if args.reuse_pilot_split:
        prior = args.reuse_pilot_split.resolve()
        prior_report = json.loads((prior / "split-report.json").read_text())
        assert json.loads((prior / "pilot.json").read_text()) == pilot
        assert prior_report["selection"]["pilot"]["case_sha256"] == selected["case_sha256"]
        reuse = {
            "split": str(prior),
            "report_sha256": digest((prior / "split-report.json").read_bytes()),
            "pilot_manifest_sha256": digest((prior / "pilot.json").read_bytes()),
            "outputs": {mode: str(prior / "runs/pilot" / mode) for mode in ["served", "full"]},
            "status": "Reuse exact pilot; completion/results remain root responsibility. Do not rerun.",
        }
        commands["pilot"] = {"reuse_only": reuse}
    report = {
        "status": "CPU_split_only_no_execution_no_new_admission",
        "reused_pilot": reuse,
        "input_manifest": str(source),
        "input_sha256": digest(raw),
        "output_manifests": artifacts,
        "selection": {
            "rule": "Maximum fullreply conditioning-token count; ties by maximum context then earliest input index. Full chosen reply remains untruncated.",
            "pilot": selected,
        },
        "coverage": {
            "exact_job_equality": True,
            "no_duplicates": True,
            "no_omissions": True,
            "input_order_preserved_in_remainder": True,
            "source_and_case_hashes_unchanged": True,
        },
        "counts_per_precision": split_counts,
        "case_records": records,
        "staged_commands_not_executed": commands,
        "execution_gate": "Root reviews measured pilot served/full wall times and load/prefill/TF accounting before admitting remainder execution. Original input admission is not permission to skip this cost gate.",
        "cost_limits": "Token-count proxies only, not time estimates. One prefill per forced branch and per free trial. Two parts incur two loads per precision (four loads if served+full both run), one additional reload per precision versus unsplit execution. Outputs are disjoint; full comparisons use only matching served part. No optional suffix512 jobs added.",
        "learned_basis": {
            "path": str(d / "learned.json"),
            "sha256": digest((d / "learned.json").read_bytes()),
            "applied": "Historical long-prefix controls timed out; shortened controls do not establish native equivalence. Measure longest original prefix before extrapolating remainder costs.",
        },
    }
    save("split-report.json", report)
    print(json.dumps({"pilot": selected["name"], "counts_per_precision": split_counts, "coverage": report["coverage"]}))


if __name__ == "__main__":
    main()
