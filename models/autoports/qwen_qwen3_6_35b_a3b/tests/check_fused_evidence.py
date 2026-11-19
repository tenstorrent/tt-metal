# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Preserve every functional hard gate and add fused-path/performance gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import check_evidence
from models.autoports.qwen_qwen3_6_35b_a3b.tests.audit_fused_source import main as audit_source

ROOT = Path(__file__).parents[1]
DOC = ROOT / "doc/fused_decoder"


def artifact_identity(path):
    provenance = json.loads(Path(str(path) + ".provenance.json").read_text())
    # Snapshot harness provenance also records the live file, which it did not execute.
    source_sha = provenance.get(
        "measured_decoder_source_sha256", provenance["source_sha256"].get("tt/fused_decoder.py")
    )
    return provenance, source_sha


def check_stress(path, *, layer, seed, decode_length, current_sha):
    """Validate complete final acceptance runs, including the known seed regressions."""
    evidence = json.loads(path.read_text())
    provenance, measured_sha = artifact_identity(path)
    expected = dict(layer=layer, seed=seed, prompt_length=128, decode_length=decode_length, continuation_length=33)
    for key, value in expected.items():
        assert evidence[key] == value, (path, key, evidence[key], value)
    assert evidence["decoder"] == "FusedDecoder" and evidence["gate_pass"], path
    assert evidence["failed_rows"] == [], (path, "failed rows")
    assert math.isfinite(evidence["prefill_pcc"]) and evidence["prefill_pcc"] >= 0.995, (path, "prefill")
    repeats = evidence["repeats"]
    assert type(repeats) is int and repeats >= 3, (path, "three or more repeats required")
    assert evidence["captures"] == 1 and evidence["repeated_bit_equal"] is True, path
    assert evidence["tracking"] is True and evidence["program_cache_included"] is True, path
    # Final acceptance must recompute HF, unlike cached-oracle diagnostic A/B runs.
    assert evidence["cpu_oracle_cache"] is None, (path, "fresh HF reference required")
    assert evidence["cpu_oracle_key"]["parameters"] == expected, (path, "oracle parameters")
    command = provenance["command"]
    assert command and Path(command[0]).name == "stress_fused_decoder.py", (path, "acceptance harness")
    assert not any(arg == "--oracle-cache" or arg.startswith("--oracle-cache=") for arg in command), path
    assert provenance["decoder"] == "fused" and measured_sha == current_sha, (path, "stale source")
    assert provenance["source_sha256"]["tt/fused_decoder.py"] == current_sha, (path, "stale live source")
    for name in ("tests/stress_fused_decoder.py", "tests/reference.py"):
        assert provenance["source_sha256"][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), (
            path,
            name,
            "stale harness/reference",
        )
    assert provenance["env"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1", (path, "tracking disabled")
    assert provenance["env"].get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") in (None, "0"), (
        path,
        "program cache excluded",
    )
    rows = evidence["results"]
    assert len(rows) == repeats * decode_length, (path, "decode row count")
    assert all(type(r["repeat"]) is int and type(r["position"]) is int for r in rows), path
    actual = {(r["repeat"], r["position"]) for r in rows}
    wanted = {(repeat, position) for repeat in range(repeats) for position in range(128, 128 + decode_length)}
    assert actual == wanted, (path, "missing, duplicate, or out-of-range decode positions")
    continuation = evidence["continuation_results"]
    assert len(continuation) == 33, (path, "continuation row count")
    assert all(type(r["position"]) is int for r in continuation), path
    assert {r["position"] for r in continuation} == set(range(128 + decode_length, 128 + decode_length + 33)), (
        path,
        "missing, duplicate, or out-of-range continuation positions",
    )
    assert all(
        math.isfinite(r[key]) and r[key] >= 0.995 for r in rows + continuation for key in ("pcc", "contribution_pcc")
    ), (path, "accuracy")


def check_comparator_endurance(path, *, source_sha, layer):
    """Comparator selection permits cached HF outputs, never state/output substitution."""
    path = Path(path)
    evidence = json.loads(path.read_text())
    provenance, measured_sha = artifact_identity(path)
    assert measured_sha == source_sha, (path, "correctness belongs to another comparator source")
    for document in (evidence, provenance):
        assert document.get("oracle_assisted", False) is False, (path, "oracle-assisted comparator")
        if "measured_decoder_source_sha256" in document:
            assert document["measured_decoder_source_sha256"] == source_sha, (path, "inconsistent source identity")
    assert evidence["decoder"] == "FusedDecoder" and evidence["gate_pass"] is True
    assert evidence["failed_rows"] == [] and evidence["layer"] == layer, (path, "ineligible comparator")
    assert type(evidence["seed"]) is int
    assert evidence["prompt_length"] == 128 and evidence["decode_length"] == 8192
    assert evidence["continuation_length"] == 33
    assert math.isfinite(evidence["prefill_pcc"]) and evidence["prefill_pcc"] >= 0.995
    repeats = evidence["repeats"]
    assert type(repeats) is int and repeats >= 1, (path, "comparator requires at least one complete stream")
    assert evidence["captures"] == 1 and evidence["repeated_bit_equal"] is True
    assert evidence["tracking"] is True and evidence["program_cache_included"] is True
    assert provenance["decoder"] == "fused" and provenance["execution"] == "TTNN device"
    assert provenance["regime"] == "serial" and provenance["device"]
    assert provenance["env"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert provenance["env"].get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") in (None, "0")
    command = provenance["command"]
    assert command and Path(command[0]).name in {"stress_fused_decoder.py", "stress_fused_snapshot.py"}, (
        path,
        "ordinary stress harness required; diagnostic interventions are ineligible",
    )
    if Path(command[0]).name == "stress_fused_snapshot.py":
        # Snapshot runs must agree on both explicit source identity fields.
        assert evidence["measured_decoder_source_sha256"] == provenance["measured_decoder_source_sha256"] == source_sha
        assert evidence["measured_decoder_source"] == provenance["measured_decoder_source"]
    # These unchanged harnesses compare TT results against CPU-only HF reference
    # outputs. They never transplant reference recurrent state or decoder outputs.
    harnesses = {"tests/stress_fused_decoder.py", "tests/reference.py", "tests/" + Path(command[0]).name}
    for name in harnesses:
        assert provenance["source_sha256"][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), (
            path,
            "unreviewed comparator harness/reference",
            name,
        )
    expected = dict(layer=layer, seed=evidence["seed"], prompt_length=128, decode_length=8192, continuation_length=33)
    assert evidence["cpu_oracle_key"]["parameters"] == expected, (path, "HF oracle parameter mismatch")
    for name in ("tests/stress_fused_decoder.py", "tests/reference.py"):
        assert (
            evidence["cpu_oracle_key"]["source_hashes"][str((ROOT / name).resolve())]
            == provenance["source_sha256"][name]
        ), (path, "HF oracle source mismatch", name)
    rows, continuation = evidence["results"], evidence["continuation_results"]
    assert len(rows) == repeats * 8192 and len(continuation) == 33, (path, "incomplete comparator stream")
    assert all(type(r["repeat"]) is int and type(r["position"]) is int for r in rows)
    assert {(r["repeat"], r["position"]) for r in rows} == {
        (repeat, position) for repeat in range(repeats) for position in range(128, 8320)
    }, (path, "missing/duplicate/out-of-range comparator decode position")
    assert all(type(r["position"]) is int for r in continuation)
    assert {r["position"] for r in continuation} == set(range(8320, 8353)), (
        path,
        "missing/duplicate/out-of-range continuation position",
    )
    assert all(
        math.isfinite(r[k]) and r[k] >= 0.995 for r in rows + continuation for k in ("pcc", "contribution_pcc")
    ), (path, "comparator accuracy")
    return dict(
        path=str(path),
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        provenance_sha256=hashlib.sha256(Path(str(path) + ".provenance.json").read_bytes()).hexdigest(),
        source_sha256=measured_sha,
        layer=layer,
        seed=evidence["seed"],
        prompt_length=128,
        decode_length=8192,
        continuation_length=33,
        repeats=repeats,
        cached_hf_reference=evidence["cpu_oracle_cache"] is not None,
        generic_provenance_decoder_label=provenance["decoder"],
        measured_decoder_identity="FusedDecoder with explicit snapshot hash"
        if Path(command[0]).name == "stress_fused_snapshot.py"
        else "FusedDecoder production harness",
    )


def comparator_requirements(item, *, layer, kind):
    """Manifest may request extra seeds; it cannot remove regression coverage."""
    assert (layer, kind) in {(0, "linear_attention"), (3, "full_attention")}
    minimum = {913, 42, 154} if layer == 0 else {913}
    supplied = item["required_endurance_seeds"]
    assert isinstance(supplied, list) and all(type(seed) is int for seed in supplied)
    required = set(supplied)
    assert minimum <= required, ("Manifest weakens mandatory endurance seeds", minimum - required)
    representatives = item.get("representative_comparator_correctness", [])
    assert isinstance(representatives, list)
    if layer == 0:
        assert representatives, "Linear comparator requires separate layer34 seed1548192+33 evidence"
    return required, representatives


def check_best_correct(manifest_path, current_sha):
    """Require explicit, source-bound corrected controls; never infer eligibility from latency."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("selection_rationale", "").strip(), "Explain selection of the best known correct controls"
    comparisons = manifest["comparisons"]
    assert comparisons, "No best-correct comparisons supplied"
    checked, layers = [], set()

    def resolve(name):
        path = Path(name)
        return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()

    def benchmark(path):
        value = json.loads(path.read_text())
        provenance, source_sha = artifact_identity(path)
        assert value["decoder"] == "FusedDecoder" and value["deterministic"]
        assert value["runtime_fallback_audit"] == "pass" and not value["latency_tracking_enabled"]
        for field in ("prefill_pcc", "prefill_contribution_pcc", "decode_eager_pcc", "decode_traced_pcc"):
            assert math.isfinite(value[field]) and value[field] >= 0.995, (path, field)
        for phase in ("prefill", "decode"):
            metric = "prefill_warmed_ms" if phase == "prefill" else "decode_traced_warmed_ms"
            assert math.isfinite(value[metric]) and value[metric] > 0, (path, metric)
            assert value[phase + "_samples_ms"] and all(
                math.isfinite(v) and v > 0 for v in value[phase + "_samples_ms"]
            )
        for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING"):
            assert provenance["env"].get(key) in (None, "", "0"), (path, "instrumented latency", key)
        assert provenance["regime"] == "serial" and provenance["device"], (path, "device/regime")
        return value, provenance, source_sha

    for item in comparisons:
        candidate_path, control_path = resolve(item["candidate"]), resolve(item["comparator"])
        candidate, cp, candidate_sha = benchmark(candidate_path)
        control, bp, control_sha = benchmark(control_path)
        assert candidate_sha == current_sha, (candidate_path, "stale candidate source")
        snapshot = resolve(item["comparator_source"])
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == control_sha, (snapshot, "wrong comparator snapshot")
        assert candidate_sha != control_sha, "A candidate cannot serve as its own historical comparator"
        for key in ("layer", "kind", "length", "context"):
            assert candidate[key] == control[key], (key, candidate_path, control_path)
        assert cp["device"] == bp["device"] and cp["versions"] == bp["versions"], "Comparator environment differs"
        for phase in ("prefill", "decode"):
            assert len(candidate[phase + "_samples_ms"]) == len(control[phase + "_samples_ms"]), (
                phase,
                "remeasure with matching repetition counts",
            )
        # Eligibility requires actual recurrent endurance, not a one-token PCC benchmark.
        correctness_paths = [resolve(name) for name in item["comparator_correctness"]]
        assert correctness_paths, "Comparator needs source-bound correctness evidence"
        required_seeds, representative_names = comparator_requirements(
            item, layer=control["layer"], kind=control["kind"]
        )
        matched_checks = [
            check_comparator_endurance(path, source_sha=control_sha, layer=control["layer"])
            for path in correctness_paths
        ]
        endurance_seeds = {row["seed"] for row in matched_checks}
        assert required_seeds <= endurance_seeds, (
            control_path,
            "missing mandatory/requested endurance seeds",
            required_seeds - endurance_seeds,
        )
        representative_paths = [resolve(name) for name in representative_names]
        representative_checks = [
            check_comparator_endurance(
                path, source_sha=control_sha, layer=34 if control["kind"] == "linear_attention" else control["layer"]
            )
            for path in representative_paths
        ]
        if control["kind"] == "linear_attention":
            assert any(
                row["layer"] == 34 and row["seed"] == 154 for row in representative_checks
            ), "Linear comparator lacks layer34 seed1548192+33 coverage"
        # The contract requires faster decode and a before/after prefill report.
        # A decode-only rewrite need not beat noise in unchanged prefill timing.
        assert candidate["decode_traced_warmed_ms"] < control["decode_traced_warmed_ms"], (
            candidate_path,
            "decode not improved against best-correct control",
        )
        layers.add(candidate["layer"])
        checked.append(
            dict(
                candidate=str(candidate_path),
                comparator=str(control_path),
                comparator_source=str(snapshot),
                candidate_artifact_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                comparator_artifact_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
                comparator_source_sha256=control_sha,
                comparator_correctness=[str(p) for p in correctness_paths],
                comparator_correctness_checks=matched_checks,
                representative_comparator_correctness=[str(p) for p in representative_paths],
                representative_comparator_checks=representative_checks,
                required_endurance_seeds=sorted(required_seeds),
                layer=candidate["layer"],
                length=candidate["length"],
                context=candidate["context"],
                prefill_before_ms=control["prefill_warmed_ms"],
                prefill_after_ms=candidate["prefill_warmed_ms"],
                prefill_delta_ms=candidate["prefill_warmed_ms"] - control["prefill_warmed_ms"],
                prefill_delta_percent=100 * (candidate["prefill_warmed_ms"] / control["prefill_warmed_ms"] - 1),
                prefill_speedup=control["prefill_warmed_ms"] / candidate["prefill_warmed_ms"],
                decode_speedup=control["decode_traced_warmed_ms"] / candidate["decode_traced_warmed_ms"],
            )
        )
    assert 0 in layers and layers <= {0, 3}, "The documented corrected linear path needs a best-correct comparison"
    return dict(
        manifest=str(manifest_path),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        selection_rationale=manifest["selection_rationale"],
        comparisons=checked,
    )


def check_all(best_correct_manifest):
    current_sha = hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest()
    best_correct = check_best_correct(best_correct_manifest, current_sha)
    audit_source()
    mapping = {
        "partition_0.json": "battery_0.json",
        "partition_3.json": "battery_3.json",
        "linear_battery.json": "battery_0.json",
        "full_resume_fixed.json": "battery_3.json",
        "linear_mixed_batch.json": "battery_0.json",
        "full_shared_pool.json": "battery_3.json",
    }

    def read(name):
        value = json.loads((DOC / mapping.get(name, name)).read_text())
        if name in ("partition_0.json", "partition_3.json"):
            # The fused runner combines boundaries, resume and batch checks in
            # one battery. Present the original boundaries-only schema to the
            # unchanged functional checker; resume/batch rows are checked by
            # their separate aliases below, with their own required fields.
            modes = {"prefill", "decode", "chunked_vs_unchunked", "chunk_partition_parity"}
            value["results"] = [row for row in value["results"] if row["mode"] in modes]
        return value

    check_evidence.DOC = DOC
    check_evidence.read = read
    check_evidence.main()
    for idx in (0, 3):
        final = read(f"final_{idx}.json")
        baseline = read(f"baseline_{idx}.json")
        assert final["decoder"] == "FusedDecoder"
        assert (
            min(
                final["prefill_pcc"],
                final["decode_traced_pcc"],
                final["unfused_prefill_pcc"],
                final["unfused_decode_pcc"],
                final["prefill_contribution_pcc"],
                final["unfused_prefill_contribution_pcc"],
                final["unfused_decode_contribution_pcc"],
            )
            >= 0.995
        )
        assert final["decode_traced_warmed_ms"] < baseline["decode_traced_warmed_ms"]
        assert not final["latency_tracking_enabled"]
        for stem, length in (("stress", 128), ("stress_extended", 1024), ("stress_endurance", 8192)):
            check_stress(DOC / f"{stem}_{idx}.json", layer=idx, seed=913, decode_length=length, current_sha=current_sha)
        names = [
            f"final_{idx}.json",
            f"battery_{idx}.json",
            f"trace_reuse_{idx}.json",
            f"stress_{idx}.json",
            f"stress_extended_{idx}.json",
            f"stress_endurance_{idx}.json",
            "linear_long.json" if idx == 0 else "full_long.json",
            f"tracy/layer_{idx}/correctness.json",
        ]
        for length in (127, 128, 129, 257, 4097):
            name = f"final_l{idx}_s{length}.json"
            measured, control = read(name), read(f"baseline_l{idx}_s{length}.json")
            assert measured["length"] == control["length"] == length
            assert measured["context"] == control["context"] == 8192
            assert (
                min(
                    measured["prefill_pcc"],
                    measured["decode_traced_pcc"],
                    measured["unfused_prefill_pcc"],
                    measured["unfused_decode_pcc"],
                    measured["prefill_contribution_pcc"],
                    measured["unfused_prefill_contribution_pcc"],
                    measured["unfused_decode_contribution_pcc"],
                )
                >= 0.995
            )
            assert measured["decode_traced_warmed_ms"] < control["decode_traced_warmed_ms"]
            names.append(name)
        for name in names:
            provenance = read(name + ".provenance.json")
            assert provenance["decoder"] == "fused", (name, provenance["decoder"])
            assert provenance["source_sha256"]["tt/fused_decoder.py"] == current_sha, (name, "stale source")
    for seed in (42, 154):
        check_stress(
            DOC / f"stress_endurance_seed{seed}_0.json", layer=0, seed=seed, decode_length=8192, current_sha=current_sha
        )
    # The actual layer34 failure is a required regression even when layers0/3 pass.
    # Reuse all fresh-reference, source-identity, completeness and quality checks.
    check_stress(
        DOC / "stress_representative_34_seed154.json",
        layer=34,
        seed=154,
        decode_length=8192,
        current_sha=current_sha,
    )
    for name in ("page_extent_regression.json", "padded_trace_regression.json"):
        assert read(name + ".provenance.json")["source_sha256"]["tt/fused_decoder.py"] == current_sha
    result = read("hard_gates.json")
    result.update(
        fused_path=True,
        unfused_equivalence=True,
        traced_decode_improved=True,
        stateful_128_in_128_out=True,
        stateful_1024_decode_and_33_continuation=True,
        stateful_8192_decode_and_33_continuation=True,
        linear_endurance_seeds=[913, 42, 154],
        full_endurance_seeds=[913],
        representative_endurance=[dict(layer=34, seed=154, decode_length=8192, continuation_length=33)],
        fresh_hf_stress_reference=True,
        current_source_sha256=current_sha,
        best_correct_performance=best_correct,
    )
    (DOC / "hard_gates.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FUSED_DECODER_HARD_GATES_PASS")


def main(best_correct_manifest):
    try:
        check_all(best_correct_manifest)
    except Exception as error:
        # The inherited functional checker writes PASS before fused checks finish.
        # Never leave that intermediate PASS masquerading as a fused-stage result.
        failure = dict(
            status="FAIL",
            fused_stage_pass=False,
            error=f"{type(error).__name__}: {error}",
            current_source_sha256=hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest(),
        )
        (DOC / "hard_gates.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-correct-manifest",
        required=True,
        type=Path,
        help="Explicit source-bound comparator manifest; schema documented in AUTOFIX_final_evidence_review.md",
    )
    main(parser.parse_args().best_correct_manifest)
