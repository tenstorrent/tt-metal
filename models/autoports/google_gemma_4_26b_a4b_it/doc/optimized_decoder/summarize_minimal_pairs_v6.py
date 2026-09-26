# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate paired prefill timing artifacts without importing torch or TTNN."""

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / "minimal_pairs_v6"
EXPECTED_RUNTIME = "b585a21f0b66144f69a823fa2d1088b130e34928a65fc91a38fd1bc5c2526846"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def value(command, flag):
    return command[command.index(flag) + 1]


def accurate(row):
    return (
        row.get("passed") is True
        and isinstance(row.get("pcc"), (float, int))
        and math.isfinite(row["pcc"])
        and 0.995 <= row["pcc"] <= 1.000000000001
    )


def close(left, right):
    return math.isclose(left, right, abs_tol=1e-8, rel_tol=0)


def check(job, entry, plan):
    require(entry["command"] == job["command"] and entry["returncode"] == 0, "Paired command failed/differs")
    require(entry["hashes"] == plan["hashes"], "Paired source hashes differ")
    command = entry["command"]
    layer, candidate = job["layer"], job["candidate"]
    require(
        value(command, "--paired-minimal-advice") == candidate and value(command, "--pairs") == "8",
        "Wrong pair control",
    )
    require(value(command, "--length") == "4096" and value(command, "--steps") == "128", "Wrong workload")
    path = Path(job["output"]).resolve()
    require(
        path.resolve().parent == DIRECTORY.resolve() and digest(path) == entry["report_sha256"],
        "Report hash/path differs",
    )
    report = read(path)
    require(report["runtime_sha256"] == EXPECTED_RUNTIME and report["real_weights"] is True, "Wrong runtime/weights")
    require(
        report["layer_type"] == ("sliding_attention" if layer == 0 else "full_attention") and report["length"] == 4096,
        "Wrong kind/length",
    )
    require(report.get("candidate") == {"defaults": True, "overrides": {}}, "Unexpected factory override")
    require(
        accurate(report) and report["runtime_prefill_audit"] == "clean" and report["program_cache_miss_guard"] is True,
        "Prefill accuracy/audit failed",
    )
    decode = report["decode"]
    require(
        decode["passed"] is True
        and decode["traced"] is True
        and decode["repeated_equal"] is True
        and decode["runtime_decode_audit"] == "clean",
        "Decode audit/determinism failed",
    )
    checks = decode["checks"]
    require(
        [row["position"] for row in checks] == [4096, *range(4096, 4224)] and all(accurate(row) for row in checks),
        "Decode coverage/PCC failed",
    )
    require(
        decode["steps"] == 128
        and decode["positions"] == [4096, 4223]
        and close(decode["min_pcc"], min(row["pcc"] for row in checks)),
        "Decode range/minimum differs",
    )
    fixture = ROOT / f"actual_text_layer{layer}_4096_128.pt"
    require(Path(report["input_fixture"]).resolve() == fixture.resolve(), "Wrong fixture path")
    require(
        report["input_fixture_sha256"] == read(fixture.with_suffix(".json"))["fixture_sha256"]
        and report["input_source"]["source"]["kind"] == "recorded_real_text_hf_layer_inputs",
        "Wrong fixture hash/source",
    )
    source_hashes = report["paired_source_hashes"]
    for source, expected in source_hashes.items():
        source = Path(source)
        if source.name == "optimized_decoder.py":
            require(expected == EXPECTED_RUNTIME, "Runtime provenance differs")
        else:
            require(digest(source) == expected, "Probe dependency changed")
    require(
        report["paired_candidate"] == {"paired_minimal_advice": candidate, "paired_placement": None, "pairs": 8},
        "Wrong candidate metadata",
    )
    policy = report["paired_policy"]
    before, after = policy["baseline"], policy["alternate"]
    baseline_path = ROOT / "minimal_advice_v6" / f"layer{layer}_baseline.json"
    baseline_report = read(baseline_path)
    require(
        baseline_report["runtime_sha256"] == EXPECTED_RUNTIME
        and baseline_report["input_fixture_sha256"] == report["input_fixture_sha256"]
        and baseline_report["minimal_advice_runtime"]["after"] == before,
        "Separate baseline control does not match paired baseline",
    )
    require(
        accurate(baseline_report)
        and baseline_report["decode"]["repeated_equal"] is True
        and [row["position"] for row in baseline_report["decode"]["checks"]] == [4096, *range(4096, 4224)]
        and all(accurate(row) for row in baseline_report["decode"]["checks"]),
        "Separate baseline accuracy failed",
    )
    require(
        before["weight_dtype"] == "DataType.BFLOAT8_B" and before["output_dtype"] == "float32", "Wrong projection dtype"
    )
    require(before["weight_shape"] == [1, 1, 2816, 8192 if layer == 0 else 9216], "Wrong projection shape")
    for key in ("weight_dtype", "weight_shape", "weight_memory", "output_dtype", "input_dtype"):
        require(before[key] == after[key], "Projection storage/precision changed unexpectedly")
    compute = dict(before["compute"])
    require(
        compute["math_fidelity"] == "MathFidelity.HiFi4" and compute["fp32_dest_acc_en"] == "True",
        "Wrong baseline compute",
    )
    if candidate == "hifi2":
        compute["math_fidelity"] = "MathFidelity.HiFi2"
        require(before["programs"] == after["programs"], "HiFi2 geometry changed")
    else:
        require(
            [program.replace("11-8", "11-10") for program in before["programs"]] == after["programs"],
            "Grid control changed other program fields",
        )
    require(compute == after["compute"], "Unintended compute change")
    selected = report["precision_policy"]["prefill_qkv_projection"]
    require(
        selected["fidelity"] == compute["math_fidelity"]
        and selected["k_block"] == (8 if layer == 0 else 16)
        and selected["grid"] == ([11, 10] if candidate == "grid110" else [11, 8]),
        "Reported final policy differs",
    )
    pairs = report["prefill_pairing"]
    require(
        pairs["pairs_requested"] == 8 and pairs["program_cache_misses_forbidden"] is True, "Pair guard/count differs"
    )
    require(
        pairs["cache_entries_after_warmup"] == pairs["cache_entries_after_samples"] > 0,
        "Cache grew during paired samples",
    )
    require(
        [(row["round"], row["candidate"]) for row in pairs["warmups"]]
        == [(0, False), (0, True), (1, False), (1, True)],
        "Both variants not warmed twice",
    )
    samples = pairs["samples"]
    require(
        [(row["pair"], row["candidate"]) for row in samples]
        == [(index, enabled) for index in range(8) for enabled in ((False, True) if index % 2 == 0 else (True, False))],
        "Pair order differs",
    )
    require(
        all(math.isfinite(row["whole_prefill_host_us"]) and row["whole_prefill_host_us"] > 0 for row in samples),
        "Invalid timing",
    )
    baseline = [row["whole_prefill_host_us"] for row in samples if not row["candidate"]]
    alternate = [row["whole_prefill_host_us"] for row in samples if row["candidate"]]
    deltas = [new - old for old, new in zip(baseline, alternate)]
    require(
        close(pairs["baseline_median_us"], statistics.median(baseline))
        and close(pairs["candidate_median_us"], statistics.median(alternate)),
        "Medians differ",
    )
    require(
        pairs["paired_candidate_minus_baseline_us"] == deltas
        and close(pairs["median_paired_delta_us"], statistics.median(deltas)),
        "Paired deltas differ",
    )
    wins = sum(delta < 0 for delta in deltas)
    require(pairs["candidate_faster_pairs"] == wins, "Pair win count differs")
    return dict(
        layer=layer,
        candidate=candidate,
        artifact=str(path.relative_to(ROOT)),
        artifact_sha256=digest(path),
        log_sha256=digest(path.with_suffix(".log")),
        runtime_sha256=EXPECTED_RUNTIME,
        source_hashes=source_hashes,
        input_fixture_sha256=report["input_fixture_sha256"],
        baseline_screen=str(baseline_path.relative_to(ROOT)),
        baseline_screen_sha256=digest(baseline_path),
        prefill_pcc=report["pcc"],
        minimum_decode_pcc=decode["min_pcc"],
        repeated_equal=True,
        baseline_samples_us=baseline,
        candidate_samples_us=alternate,
        baseline_median_us=statistics.median(baseline),
        candidate_median_us=statistics.median(alternate),
        paired_delta_us=deltas,
        median_paired_delta_us=statistics.median(deltas),
        candidate_faster_pairs=wins,
        program_cache_entries=pairs["cache_entries_after_samples"],
        policy=policy,
        adoption="not selected by this timing-only summary; any fidelity winner still requires long/stress accuracy controls",
    )


def main():
    plan = read(DIRECTORY / "plan.json")
    require(plan["hashes"]["runtime"] == EXPECTED_RUNTIME, "Plan runtime differs")
    require(digest(DIRECTORY / "runtime_snapshot.py.txt") == EXPECTED_RUNTIME, "Saved runtime snapshot differs")
    jobs = plan["jobs"]
    require(
        [(job["layer"], job["candidate"]) for job in jobs]
        == [(layer, candidate) for layer in (0, 5) for candidate in ("hifi2", "grid110")],
        "Plan coverage differs",
    )
    journal = read(DIRECTORY / "commands.json") if (DIRECTORY / "commands.json").is_file() else []
    results, pending, errors = [], [], []
    for job in jobs:
        entries = [entry for entry in journal if entry["output"] == job["output"]]
        if not entries:
            pending.append(dict(layer=job["layer"], candidate=job["candidate"]))
            continue
        try:
            require(len(entries) == 1, "Duplicate command")
            results.append(check(job, entries[0], plan))
        except (KeyError, ValueError, OSError, TypeError) as error:
            errors.append(dict(layer=job["layer"], candidate=job["candidate"], error=str(error)))
    summary = dict(
        status="failed" if errors else "pending" if pending else "paired_controls_passed_selection_pending",
        runtime_sha256=EXPECTED_RUNTIME,
        generator_sha256=digest(Path(__file__)),
        plan_sha256=digest(DIRECTORY / "plan.json"),
        command_journal_sha256=digest(DIRECTORY / "commands.json") if journal else None,
        results=results,
        pending=pending,
        errors=errors,
        scope="Eight alternating synchronous whole-prefill host pairs per real4096/128 control. Both variants warmed twice; samples forbid cache misses and exclude initial sync/deallocation. Final candidate only is checked against HF in each paired run; original matched baseline screen is separate. This is not per-op device timing or proof of maximum/stress accuracy.",
    )
    (ROOT / "minimal_pairs_v6_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Minimal QKV paired timing controls",
        "",
        f'Runtime: `{EXPECTED_RUNTIME}`. Status: **{summary["status"]}**.',
        "",
        summary["scope"],
        "",
        "| Layer | Candidate | Baseline median µs | Candidate median µs | Median paired delta µs | Faster pairs | Prefill PCC | Minimum decode PCC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f'| {row["layer"]} | {row["candidate"]} | {row["baseline_median_us"]:.3f} | {row["candidate_median_us"]:.3f} | {row["median_paired_delta_us"]:+.3f} | {row["candidate_faster_pairs"]}/8 | {row["prefill_pcc"]:.10f} | {row["minimum_decode_pcc"]:.10f} |'
        )
    lines += [
        "",
        "Negative paired deltas favor the candidate. Every sample is retained; medians and win counts describe these eight pairs without a confidence or hardware-cause claim. Candidate precision, full source hashes, raw samples and cache counts are retained in [the JSON](minimal_pairs_v6_summary.json). No policy is adopted by this report; fidelity winners require long-context and512-step stress checks before integration.",
    ]
    if pending:
        lines += ["", f"Pending controls: {pending}."]
    if errors:
        lines += ["", f"Artifact validation errors: {errors}."]
    (ROOT / "minimal_pairs_v6_summary.md").write_text("\n".join(lines) + "\n")
    print(summary["status"], len(results), "complete;", len(pending), "pending;", len(errors), "errors")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
