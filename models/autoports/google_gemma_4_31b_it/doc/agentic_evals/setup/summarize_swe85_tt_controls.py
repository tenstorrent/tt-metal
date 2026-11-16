#!/usr/bin/env python3
"""Summarize completed SWE85 diagnostics without exposing private text or token IDs.

Default output is compact JSON on stdout. --output writes a summary atomically.
Exit 2 means schema/alignment errors, or incomplete coverage with --require-complete.
Partial outputs never contribute metrics. Mechanical flags require semantic review.
"""
import argparse
import collections
import hashlib
import json
import math
import zlib
from pathlib import Path

D = Path(__file__).resolve().parent.parent
REPO = D.parents[4]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def read(path):
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def action_divergence(raw, case):
    """Compare targets only where both forced branches have the same context."""
    branches = raw["continuations"]
    if not {"chosen", "correct"} <= branches.keys():
        return {"status": "not_applicable"}
    a, b = (branches[k] for k in ("chosen", "correct"))
    if (
        a["conditioning_token_ids"] != b["conditioning_token_ids"]
        or a["conditioning_token_ids"] != case["prompt_token_ids"]
    ):
        raise ValueError("action_conditioning_mismatch")
    for name in ("chosen", "correct"):
        rows = branches[name]["positions"]
        if [(r["position"], r["forced_token"]) for r in rows] != list(enumerate(case["continuations"][name])):
            raise ValueError("action_forced_prefix_or_positions_mismatch")
    limit = min(len(a["positions"]), len(b["positions"]))
    index = next(
        (i for i in range(limit) if a["positions"][i]["forced_token"] != b["positions"][i]["forced_token"]), None
    )
    if index is None:
        return {
            "status": "no_shared_position_with_different_targets",
            "shared_forced_prefix_tokens": limit,
            "reason": "Identical alternatives or one is a strict prefix; no local target comparison.",
        }
    if any(a["positions"][i]["forced_token"] != b["positions"][i]["forced_token"] for i in range(index)):
        raise ValueError("action_forced_prefix_mismatch")
    out = {
        "status": "verified_common_prefix",
        "branch_position_zero_based": index,
        "absolute_next_token_index_zero_based": len(a["conditioning_token_ids"]) + index,
        "conditioning_tokens": len(a["conditioning_token_ids"]),
        "shared_forced_prefix_tokens": index,
        "conditioning_and_preceding_forced_prefix_identical": True,
        "targets": {},
        "scope": "Local alternative-token probabilities under a common prefix; not sequence preference or task success.",
    }
    for name in ("chosen", "correct"):
        row = branches[name]["positions"][index]
        lp, prob = row["forced_token_logprob"], row["forced_token_probability"]
        if (
            not finite(lp)
            or lp > 0
            or not finite(prob)
            or not 0 <= prob <= 1
            or not math.isclose(math.exp(lp), prob, rel_tol=1e-4, abs_tol=1e-7)
        ):
            raise ValueError("invalid_target_probability")
        out["targets"][name] = {
            "target_log_probability": lp,
            "target_probability": prob,
            "target_is_top1": row["forced_token"] == row["top1_token"],
        }
    out["same_top1_across_branch_replays"] = a["positions"][index]["top1_token"] == b["positions"][index]["top1_token"]
    return out


def validate_precision(policy, mode):
    """Check reported loaded policy, not just the diagnostic output directory."""
    full = mode == "full"
    expected = {
        "config_id": "stage12_bf16_hifi4_diagnostic" if full else "stage12_head_bfp8_lofi",
        "weight_groups": dict(
            attention="bf16" if full else "bfp4",
            mlp="bf16" if full else "bfp4",
            down="bf16" if full else "bfp4",
            lm_head="bf16" if full else "bfp8_lofi",
        ),
        "layer_exceptions": {},
        "compute_fidelities": {k: "HiFi4" if full else "LoFi" for k in ("attention", "mlp", "down", "lm_head")},
        "activation_dtype": "bfloat16" if full else "bfloat8_b",
        "residual_dtype": "bfloat16",
        "ccl_dtype": {"prefill": "bfloat16" if full else "bfloat8_b", "decode": "bfloat16"},
        "kv_cache_dtype": "bfloat16" if full else "bfloat8_b",
        "max_seq_len": 262144,
        **{k: "bfloat16" for k in ("logits_dtype", "sampling_dtype", "embedding_dtype", "norm_dtype")},
    }
    if full:
        expected["sdpa_compute_fidelity"] = "HiFi4"
    if mode not in ("served", "full") or any(policy.get(k) != v for k, v in expected.items()):
        raise ValueError("loaded_precision_policy_mismatch")


def inspect(path, job, case, mode):
    if not path.exists():
        return {"status": "missing", "path": str(path)}, None
    try:
        raw = read(path)
    except (OSError, ValueError):
        return {"status": "unreadable_or_in_progress", "path": str(path)}, None
    if "diagnostic_total_wall_s" not in raw:
        return {"status": "in_progress", "path": str(path)}, None
    errors = []
    branches = {}
    free = []
    try:
        validate_precision(raw["runtime_precision"], mode)
        assert finite(raw["diagnostic_total_wall_s"]) and raw["diagnostic_total_wall_s"] >= 0
        assert resolve(raw["case"]).resolve() == resolve(job["case"]).resolve()
        assert set(raw["continuations"]) == set(job["branches"])
        for name in job["branches"]:
            b = raw["continuations"][name]
            rows = b["positions"]
            expected = case["continuations"][name]
            assert b["conditioning_token_ids"] == case["prompt_token_ids"]
            assert len(rows) == len(expected) == b["tokens"] and rows
            assert [(r["position"], r["forced_token"]) for r in rows] == list(enumerate(expected))
            assert all(finite(r["forced_token_logprob"]) for r in rows)
            total = sum(r["forced_token_logprob"] for r in rows)
            assert math.isclose(total, b["sequence_log_probability"], rel_tol=1e-9, abs_tol=1e-7)
            assert finite(b["diagnostic_wall_s_including_prefill"])
            branches[name] = {
                "tokens": len(rows),
                "top1_matches_forced_fraction": sum(r["top1_token"] == r["forced_token"] for r in rows) / len(rows),
                "sequence_log_probability": total,
                "mean_token_log_probability": total / len(rows),
                "sequence_comparison_caveat": "Unequal-length sums are confounded by length; mean logprob is descriptive, not a task-level preference.",
                "wall_s_including_prefill": b["diagnostic_wall_s_including_prefill"],
            }
        divergence = action_divergence(raw, case)
        trials = raw.get("free_running", [])
        expected_trials = job.get("free_run_trials", 3) if job.get("free_run_tokens") else 0
        assert len(trials) == expected_trials
        for i, trial in enumerate(trials):
            text = trial["text"]
            tail = text[-4096:]
            ratio = len(zlib.compress(tail.encode())) / max(1, len(tail.encode()))
            flag = len(tail) >= 1000 and (ratio < 0.05 or len(set(tail)) <= 10)
            assert trial["flag"] == flag
            assert math.isclose(trial["tail_zlib_ratio"], ratio, abs_tol=1e-12)
            assert trial["tail_distinct_chars"] == len(set(tail))
            assert trial["digest_min_chars_met"] == (len(tail) >= 1000)
            assert len(trial["token_ids"]) <= job["free_run_tokens"]
            params = dict(case["metadata"]["sampling"], seed=case["metadata"].get("seed", 42) + i)
            assert trial["sampling"] == params
            free.append(
                {
                    "trial": i,
                    "sampling": params,
                    "tokens": len(trial["token_ids"]),
                    "mechanical_flag": flag,
                    "digest_min_chars_met": len(tail) >= 1000,
                    "tail_zlib_ratio": ratio,
                    "tail_distinct_chars": len(set(tail)),
                    "wall_s_including_prefill": trial["diagnostic_wall_s_including_prefill"],
                    "raw_text_reference": {"path": str(path), "json_pointer": f"/free_running/{i}/text"},
                    "semantic_review": "pending; raw tool/channel markers can be normal runner serialization, not evidence of visible-assistant protocol failure",
                }
            )
    except (AssertionError, KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("completed_output_schema_or_fixture_mismatch")
    if errors:
        return {"status": "invalid", "path": str(path), "errors": errors}, None
    return {
        "status": "complete",
        "path": str(path),
        "runtime_config_id": raw.get("runtime_precision", {}).get("config_id"),
        "wall_s": raw["diagnostic_total_wall_s"],
        "branches": branches,
        "free_running": free,
        "action_divergence": divergence,
    }, raw


def supplement_inputs(manifest, runs):
    """Bind optional inputs to their reviewed admission and staged output paths."""
    admission_path = manifest.parent / "admission.json"
    admission = read(admission_path)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    jobs = read(manifest)
    if admission["manifest_sha256"] != digest or admission["jobs"] != len(jobs):
        raise ValueError("Supplement manifest does not match admission")
    if not admission["status"].startswith("admitted") or admission["frozen_161_changed"] is not False:
        raise ValueError("Supplement is not an independent admitted fixture")
    review = resolve(admission["review"])
    if hashlib.sha256(review.read_bytes()).hexdigest() != admission["review_sha256"]:
        raise ValueError("Supplement review hash mismatch")
    for mode in ("served", "full"):
        argv = admission["staged_commands_not_executed"][mode]
        if resolve(argv[argv.index("--cases") + 1]).resolve() != manifest.resolve():
            raise ValueError("Supplement command manifest mismatch")
        if resolve(argv[argv.index("--output-dir") + 1]).resolve() != (runs / mode).resolve():
            raise ValueError("Supplement runs directory does not match admission")
    preparation_path = manifest.parent / "preparation.json"
    preparation = read(preparation_path)
    normalization = {r["original_name"]: r["execution_name"] for r in preparation.get("name_normalization", [])}
    source_jobs = []
    for source in preparation["sources"]:
        path = resolve(source["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Supplement source manifest hash mismatch")
        for job in read(path):
            job = dict(job)
            job["name"] = normalization.get(job["name"], job["name"])
            source_jobs.append(job)
    if source_jobs != jobs:
        raise ValueError("Supplement jobs differ from reviewed source manifests")
    return jobs, {
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "admission": str(admission_path),
        "admission_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "review": str(review),
        "review_sha256": admission["review_sha256"],
        "preparation": str(preparation_path),
        "preparation_sha256": hashlib.sha256(preparation_path.read_bytes()).hexdigest(),
        "sources": preparation["sources"],
        "runs": str(runs),
        "binding_verified": True,
    }


def summarize(rest_manifest, pilot_manifest, rest_dir, pilot_dir, supplement_manifest=None, supplement_runs=None):
    jobs = [(j, rest_dir, "rest") for j in read(rest_manifest)] + [
        (j, pilot_dir, "reused_pilot") for j in read(pilot_manifest)
    ]
    if (supplement_manifest is None) != (supplement_runs is None):
        raise ValueError("Both supplement manifest and runs arguments are required")
    evidence = None
    if supplement_manifest is not None:
        supplemental, evidence = supplement_inputs(supplement_manifest, supplement_runs)
        jobs.extend((j, supplement_runs, "supplement") for j in supplemental)
    if len({j["name"] for j, _, _ in jobs}) != len(jobs):
        raise ValueError("Duplicate job names")
    identities = [(str(resolve(j["case"]).resolve()), tuple(j["branches"])) for j, _, _ in jobs]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate case/branch inputs under different names")
    rows = []
    counts = {mode: collections.Counter() for mode in ("served", "full")}
    for job, directory, group in jobs:
        case = read(resolve(job["case"]))
        metadata = case.get("metadata", {})
        task = metadata.get("task") or metadata.get("task_id") or job["name"].split(".")[1]
        row = {
            "name": job["name"],
            "task": task,
            "call": metadata.get("call"),
            "group": group,
            "case": job["case"],
            "case_sha256": hashlib.sha256(resolve(job["case"]).read_bytes()).hexdigest(),
        }
        raw = {}
        for mode in counts:
            row[mode], raw[mode] = inspect(directory / mode / (job["name"] + ".json"), job, case, mode)
            counts[mode][row[mode]["status"]] += 1
        row["comparison_status"] = "pending"
        if all(raw.values()):
            comparison = {}
            try:
                for name in job["branches"]:
                    a, b = (raw[m]["continuations"][name] for m in ("served", "full"))
                    assert a["conditioning_token_ids"] == b["conditioning_token_ids"]
                    assert [(r["position"], r["forced_token"]) for r in a["positions"]] == [
                        (r["position"], r["forced_token"]) for r in b["positions"]
                    ]
                    computed = {
                        "tokens": len(a["positions"]),
                        "top1_agreement": sum(
                            x["top1_token"] == y["top1_token"] for x, y in zip(a["positions"], b["positions"])
                        )
                        / len(a["positions"]),
                        "served_sequence_log_probability": a["sequence_log_probability"],
                        "full_sequence_log_probability": b["sequence_log_probability"],
                    }
                    saved = raw["full"]["comparison"][name]
                    assert all(
                        finite(saved[k]) and math.isclose(saved[k], v, rel_tol=1e-9, abs_tol=1e-7)
                        for k, v in computed.items()
                    )
                    comparison[name] = computed
                row.update(comparison_status="verified", comparison=comparison)
                da, db = (row[m]["action_divergence"] for m in ("served", "full"))
                if da["status"] == db["status"] == "verified_common_prefix":
                    assert da["branch_position_zero_based"] == db["branch_position_zero_based"]
                    i = da["branch_position_zero_based"]
                    row["action_divergence_comparison"] = {
                        "status": "verified_common_prefix_across_precisions",
                        "branch_position_zero_based": i,
                        "absolute_next_token_index_zero_based": da["absolute_next_token_index_zero_based"],
                        "served_full_top1_agreement_by_branch": {
                            name: raw["served"]["continuations"][name]["positions"][i]["top1_token"]
                            == raw["full"]["continuations"][name]["positions"][i]["top1_token"]
                            for name in ("chosen", "correct")
                        },
                        "scope": "Same conditioning and forced prefix verified; see each mode action_divergence for local target probabilities.",
                    }

            except (AssertionError, KeyError, TypeError, ValueError):
                row["comparison_status"] = "invalid_saved_comparison_or_alignment"
        rows.append(row)
    timing = {}
    for mode in counts:
        load = {}
        directories = [("rest", rest_dir), ("reused_pilot", pilot_dir)]
        if supplement_runs is not None:
            directories.append(("supplement", supplement_runs))
        for group, directory in directories:
            try:
                value = read(directory / mode / "load.json")["wall_s"]
                if finite(value):
                    load[group] = value
            except (OSError, ValueError, KeyError, TypeError):
                pass
        timing[mode] = {
            "completed_case_wall_s_sum": sum(r[mode]["wall_s"] for r in rows if r[mode]["status"] == "complete"),
            "observed_load_wall_s": load,
            "scope": "Diagnostic host wall accounting; not serving rates or elapsed run wall.",
        }
    invalid = any(r[m]["status"] == "invalid" for r in rows for m in counts) or any(
        r["comparison_status"].startswith("invalid") for r in rows
    )
    complete = all(r["comparison_status"] == "verified" for r in rows)
    return {
        "status": "invalid" if invalid else ("complete" if complete else "partial"),
        "supplemental_admission_evidence": evidence,
        "scope": (
            "Frozen rest85 plus reused pilot83; explicitly selected reviewed supplemental controls included. "
            if evidence
            else "Frozen rest85 plus reused pilot83 only; supplemental controls excluded. "
        )
        + "Agreement and mechanical flags do not resolve anomalies or rule out precision effects. Semantic review pending. Raw tool/channel markers may be canonical runner syntax; native visible-assistant protocol failures require conditioning/channel and tokenizer/parser-aware review.",
        "expected_jobs": len(rows),
        "expected_unique_tasks": len({r["task"] for r in rows}),
        "completion_counts": {m: dict(c) for m, c in counts.items()},
        "verified_comparisons": sum(r["comparison_status"] == "verified" for r in rows),
        "completed_unique_tasks": {m: len({r["task"] for r in rows if r[m]["status"] == "complete"}) for m in counts},
        "timing": timing,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest-manifest", type=Path, default=D / "setup/swe85-pilot-split85/rest.json")
    parser.add_argument("--pilot-manifest", type=Path, default=D / "setup/swe85-pilot-split83/pilot.json")
    parser.add_argument("--rest-dir", type=Path, default=D / "setup/swe85-pilot-split85/runs/rest")
    parser.add_argument("--pilot-dir", type=Path, default=D / "setup/swe85-pilot-split83/runs/pilot")
    parser.add_argument(
        "--supplement-manifest", type=Path, help="Explicit reviewed admitted-cases.json; requires --supplement-runs"
    )
    parser.add_argument(
        "--supplement-runs", type=Path, help="Admitted supplement runs directory containing served/full"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        result = summarize(
            args.rest_manifest,
            args.pilot_manifest,
            args.rest_dir,
            args.pilot_dir,
            args.supplement_manifest,
            args.supplement_runs,
        )
    except (ValueError, KeyError, OSError) as exc:
        parser.error("Input admission/schema validation failed: " + type(exc).__name__)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(payload)
        temp.replace(args.output)
        print(
            json.dumps({k: result[k] for k in ("status", "expected_jobs", "completion_counts", "verified_comparisons")})
        )
    else:
        print(payload, end="")
    return 2 if result["status"] == "invalid" or (args.require_complete and result["status"] != "complete") else 0


if __name__ == "__main__":
    raise SystemExit(main())
