# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize the bounded full-prefill QKV geometry/placement controls; CPU only."""

import ast
import datetime
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = (
    ("original_m4", "prefill_qkv_l1_v7_command.json"),
    ("m2_l1_vs_m4_dram", "prefill_qkv_l1_m2_v7_command.json"),
    ("n4_l1_vs_m4_dram", "prefill_qkv_l1_n4_v7_command.json"),
    ("m2_placement", "prefill_qkv_m2_placement_v7_command.json"),
    ("m2_placement_65", "prefill_qkv_m2_placement_v7_65_command.json"),
    ("m2_grid", "prefill_qkv_m2_grid_v7_command.json"),
    ("m2_producer", "prefill_qkv_producer_v7_command.json"),
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(artifact=str(path.relative_to(ROOT)), sha256=sha(path))


def collect(name, journal_path):
    journal = read(journal_path)
    if "returncode" not in journal:
        return None
    argv = journal["command"]
    path = Path(argv[argv.index("--output") + 1]).resolve()
    assert sha(path) == journal["output_sha256"]
    raw = read(path)
    result = dict(
        name=name,
        journal=ref(journal_path),
        report=ref(path),
        command=argv,
        runtime_sha256=journal["runtime_sha256"],
        returncode=journal["returncode"],
        profiler_and_watcher_unset=journal["profiler_and_watcher_unset"],
        environment_overrides=journal["environment_overrides"],
        length=int(argv[argv.index("--length") + 1]),
        candidate_metadata={key: value for key, value in raw.items() if key.startswith("qkv_") and "error" not in key},
    )
    if result["returncode"]:
        assert name == "original_m4"
        error = raw["qkv_input_l1_error"]
        assert "1114112" in error and "1307648" in error
        result.update(
            status="capacity failure before candidate timing and PCC; not a placement-family rejection",
            error=error.split("backtrace:")[0].strip(),
            observed_lowest_live_l1_tensor_start=1114112,
            observed_static_cb_end=1307648,
        )
        return result
    assert raw["passed"] and raw["decode"]["passed"] and raw["decode"]["repeated_equal"]
    assert raw["runtime_sha256"] == journal["runtime_sha256"]
    pair = raw["paired_prefill"]
    assert pair["program_cache_misses_forbidden"]
    assert pair["cache_entries_after_warmup"] == pair["cache_entries_after_samples"]
    assert len(pair["samples"]) == 2 * pair["pairs_requested"]
    deltas = []
    for index in range(pair["pairs_requested"]):
        samples = {row["candidate"]: row["whole_prefill_host_us"] for row in pair["samples"] if row["pair"] == index}
        assert len(samples) == 2
        deltas.append(samples[True] - samples[False])
    assert deltas == pair["paired_candidate_minus_baseline_us"]
    assert statistics.median(deltas) == pair["median_paired_delta_us"]
    result.update(
        status="passed actual input and unchanged program-cache gate",
        input_fixture_sha256=raw["input_fixture_sha256"],
        prefill_pcc=raw["pcc"],
        minimum_decode_pcc=raw["decode"]["min_pcc"],
        pairs=pair,
        paired_delta_range_us=[min(deltas), max(deltas)],
        paired_delta_quartiles_us=statistics.quantiles(deltas, n=4),
    )
    return result


def main():
    results, pending = [], []
    for name, journal in CASES:
        path = ROOT / journal
        result = collect(name, path) if path.exists() else None
        if result is None:
            pending.append(name)
        else:
            results.append(result)
    helper = ROOT.parent.parent / "tests/probe_optimized_prefill_pairs.py"
    snapshot = ROOT / "probe_optimized_prefill_pairs_before_format.py.txt"
    historical_helper = dict(live_path=str(helper), live_sha256=sha(helper))
    if snapshot.exists():
        equivalent = ast.dump(ast.parse(snapshot.read_text())) == ast.dump(ast.parse(helper.read_text()))
        assert equivalent
        assert sha(snapshot) == "13f71dfb87a9cced61bd974effdb666dee972bade6591954fcb86a8c76a09376"
        historical_helper.update(snapshot=ref(snapshot), ast_equivalent_to_live=equivalent)
    report = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="candidate controls complete; selected integration validation/native profile separate"
        if not pending
        else "candidate controls pending",
        scope="Full attention only, exact recorded real activations, FP32 input/output, BFP8 weights and HiFi2. Same-process alternating whole-prefill host timing includes input movement. Device profiling and integrated-policy validation are separate.",
        output_placement_contract="The probes forward memory_config=None. Minimal matmul inherits the input memory config for its output: DRAM input yields DRAM output; L1 input yields L1 output. The matched placement control changes both boundaries together. Copy-L1 versus producer-L1 and grid88 versus110 preserve both L1 boundaries. Native v8 rows verify L1 input and output; historical raw reports are unchanged.",
        results=results,
        pending=pending,
        paired_helper_provenance=historical_helper,
        source_cb_inventory=dict(
            source="ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp:312",
            input0_double_buffer="M*K*2*4096 bytes",
            input1_double_buffer="K*N*2*1088 bytes",
            output_double_buffer="M*N*2*4096 bytes",
            intermediate="M*N*4096 bytes",
            observed_static_base_bytes=111616,
            original_m4_k16_n8=dict(payload_bytes=1196032, estimated_end_bytes=1307648),
            adapted_m2_k16_n8=dict(payload_bytes=737280, estimated_end_bytes=848896),
            adapted_m4_k16_n4=dict(payload_bytes=860160, estimated_end_bytes=971776),
            note="These estimates explain the first overlap and legal adaptations at that allocation state; they are not a general free-L1 guarantee.",
        ),
        interpretation=[
            "The first M4 L1 failure is a config/allocation overlap, not rejection of the entire L1 family. M2 and N4 retain K16 and change bounded geometry to fit.",
            "Matched DRAM/M2 versus copied-L1/M2 isolates the coupled input/output placement family from the M4-to-M2 geometry change. It is not evidence of an input-only benefit: unspecified output memory inherits the input memory config.",
            "The 65-token control checks three tiled M rows with an M2 block; its timings are a boundary sample, not headline performance.",
            "Copy-L1/M2 versus direct-producer-L1/M2 holds the matmul program and L1 input/output tensor specs fixed. Only the final input-normalization gamma multiply output placement changes; other sites and decode remain original.",
            "M2 copied-L1 grid88 versus110 is independent of the earlier M4 grid test. Both candidate and baseline use the same precision and L1 input spec.",
            "No absolute geometry optimum or final stage acceptance is inferred from this bounded matrix.",
        ],
    )
    (ROOT / "qkv_l1_results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Full prefill QKV L1 controls",
        "",
        report["scope"],
        "",
        "| Control | Tokens | Baseline median µs | Candidate median µs | Median paired delta µs | Faster pairs | Prefill PCC | Minimum decode PCC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        if "pairs" not in item:
            continue
        pair = item["pairs"]
        lines.append(
            f"| {item['name']} | {item['length']} | {pair['baseline_median_us']:.3f} | {pair['candidate_median_us']:.3f} | {pair['median_paired_delta_us']:+.3f} | {pair['candidate_faster_pairs']}/{pair['pairs_requested']} | {item['prefill_pcc']:.12f} | {item['minimum_decode_pcc']:.12f} |"
        )
    lines += ["", *report["interpretation"], ""]
    lines += [
        "Original M4/K16/N8 static CBs end at 1,307,648 bytes, beyond the lowest observed live L1 tensor allocation beginning at 1,114,112. The exception does not identify that allocation as input versus output. With the same 111,616-byte static base, M2/K16/N8 ends at 848,896 and M4/K16/N4 at 971,776. Raw failure and successful adapted runs are preserved separately.",
        "",
        report["output_placement_contract"],
        "",
        "The JSON links exact commands, fixture/source hashes, all paired samples, quartiles, observed tensor memory and configured programs. The median of paired deltas is reported directly; it need not equal the difference between independently computed medians.",
        "",
        "Historical paired-helper hash13f71dfb resolves to the exact preserved probe_optimized_prefill_pairs_before_format.py.txt snapshot. The later live-helper formatting is AST-equivalent; historical report hashes remain unchanged and are not rebound to the formatted helper.",
        "",
        "Pending controls: "
        + (", ".join(pending) or "none")
        + ". Selected integration validation and final native profiles remain separate.",
    ]
    (ROOT / "qkv_l1_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(completed=len(results), pending=pending)))


if __name__ == "__main__":
    main()
