# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audit four completed prefill-router pairs and retain the measured default."""

import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(artifact=str(path.relative_to(ROOT)), sha256=sha(path))


def main():
    directory = ROOT / "prefill_router_v8"
    journal_path = directory / "commands.json"
    journal = read(journal_path)
    assert len(journal) == 4
    results = []
    for command in journal:
        assert command["returncode"] == 0 and command["status"] == "passed"
        path = Path(command["output"]).resolve()
        assert sha(path) == command["report_sha256"]
        raw = read(path)
        assert raw["passed"] and raw["decode"]["passed"] and raw["decode"]["repeated_equal"]
        assert raw["program_cache_miss_guard"]
        pair = raw["paired_prefill"]
        assert pair["pairs_requested"] == 32 and len(pair["samples"]) == 64
        assert pair["program_cache_misses_forbidden"]
        assert pair["cache_entries_after_warmup"] == pair["cache_entries_after_samples"]
        values = []
        for index in range(32):
            sample = {row["candidate"]: row["whole_prefill_host_us"] for row in pair["samples"] if row["pair"] == index}
            assert len(sample) == 2
            values.append(sample[True] - sample[False])
        assert values == pair["paired_candidate_minus_baseline_us"]
        assert statistics.median(values) == pair["median_paired_delta_us"]
        metadata = raw["router_prefill_control"]["metadata"][0]
        observed = metadata["observed"]
        for value in observed.values():
            assert "FLOAT32" in value["input_dtype"]
            assert value["output_dtype"] == "float32" and value["output_memory"] == "DRAM interleaved"
        assert "BFLOAT16" in metadata["weight_dtype"]
        assert observed["baseline"]["program"] == observed["candidate"]["program"]
        assert observed["baseline"]["compute"] == metadata["baseline_compute"]
        assert observed["candidate"]["compute"] == metadata["candidate_compute"]
        if command["candidate"] == "hifi2":
            assert observed["baseline"]["input_memory"] == observed["candidate"]["input_memory"]
            assert observed["baseline"]["compute"] != observed["candidate"]["compute"]
        else:
            assert observed["baseline"]["compute"] == observed["candidate"]["compute"]
            assert "DRAM" in observed["baseline"]["input_memory"] and "L1" in observed["candidate"]["input_memory"]
        wins = sum(value < 0 for value in values)
        positive = sum(value > 0 for value in values)
        nonzero = wins + positive
        sign_p = min(1.0, 2 * sum(math.comb(nonzero, k) for k in range(min(wins, positive) + 1)) / 2**nonzero)
        results.append(
            dict(
                layer=command["layer"],
                candidate=command["candidate"],
                report=ref(path),
                runtime_sha256=raw["runtime_sha256"],
                input_fixture_sha256=raw["input_fixture_sha256"],
                command=command["command"],
                source_hashes=raw["router_prefill_control"]["source_hashes"],
                prefill_pcc=raw["pcc"],
                minimum_decode_pcc=raw["decode"]["min_pcc"],
                metadata=metadata,
                requested_fidelity_from_hashed_source=dict(
                    baseline="HiFi4", candidate="HiFi2" if command["candidate"] == "hifi2" else "HiFi4"
                ),
                paired=pair,
                paired_delta_summary=dict(
                    mean_us=statistics.mean(values),
                    standard_deviation_us=statistics.stdev(values),
                    quartiles_us=statistics.quantiles(values, n=4),
                    range_us=[min(values), max(values)],
                    first_half_median_us=statistics.median(values[:16]),
                    second_half_median_us=statistics.median(values[16:]),
                    median_pct=100 * statistics.median(values) / pair["baseline_median_us"],
                    exploratory_two_sided_sign_p=sign_p,
                ),
                disposition="Retain current HiFi4 and DRAM producer: no resolved whole-prefill gain in this 32-pair control"
                if command["candidate"] != "hifi2" or command["layer"] != 5
                else "Retain current HiFi4: candidate is slower in22/32 pairs",
            )
        )
    assert {(item["layer"], item["candidate"]) for item in results} == {
        (layer, candidate) for layer in (0, 5) for candidate in ("hifi2", "producer_l1")
    }
    report = dict(
        generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status="complete; retain current router-prefill policy for both kinds",
        runtime_sha256=results[0]["runtime_sha256"],
        journal=ref(journal_path),
        plan=ref(directory / "plan.json"),
        scope="Four separate phase-matched controls, exact real4096/128 inputs, 32 alternating same-process pairs each. Whole-prefill host timing includes final synchronization; selection, setup, warmup and output deallocation are excluded. Device-native metrics remain separate.",
        source_contract=dict(
            prefill_branch="tt/fused_decoder.py:412-421",
            actual_precision="FP32 scaled input, BF16 projection weight, FP32 output, HiFi4, fp32 destination, approximate math disabled",
            locked_program="11x10, K8, perM4/perN1, sub4x1, explicit DRAM output; actual1024x2816x128 native geometry",
            producer="Existing scaled-input multiply retains its MUL_UNARY_SFPU scalar-root activation; only its output memory changes to L1. No copy is added. HiFi2 control retains the original DRAM producer. Decode hooks delegate unchanged.",
        ),
        results=results,
        conclusion="Retain HiFi4 and DRAM prefill-router input for both kinds. Apparent savings are small relative to overlapping paired variation; full HiFi2 trends slower. Both source factors were executed independently with actual HF gates and stable program caches. No current-runtime edit follows from these trials.",
        statistical_scope="Quartiles, pair counts and all samples are primary descriptive evidence. The exploratory sign statistic assumes independent pair signs and is not a proof of equivalence or a global optimum.",
        metadata_limit="The probe serialized compute configs with str(), which yields opaque Python object identities. Requested fidelities are established by its hash-bound constructor, baseline HiFi4 guard and selected object identity; those strings are not native config dumps. Current selected native v8 rows independently prove HiFi4/FP32 destination. Historical reports are preserved unchanged.",
        remaining=[],
    )
    (ROOT / "prefill_router_results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Prefill router advice controls",
        "",
        report["scope"],
        "",
        "| Layer | Change | Baseline µs | Candidate µs | Median paired delta µs (%) | Faster pairs | Paired IQR µs | Prefill PCC | Min decode PCC |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for item in results:
        p = item["paired"]
        s = item["paired_delta_summary"]
        q = s["quartiles_us"]
        lines.append(
            f"| {item['layer']} | {item['candidate']} | {p['baseline_median_us']:.3f} | {p['candidate_median_us']:.3f} | {p['median_paired_delta_us']:+.3f} ({s['median_pct']:+.4f}%) | {p['candidate_faster_pairs']}/32 | [{q[0]:+.3f}, {q[2]:+.3f}] | {item['prefill_pcc']:.12f} | {item['minimum_decode_pcc']:.12f} |"
        )
    lines += [
        "",
        report["conclusion"],
        "",
        "Every control passes prefill and all128 decode PCC checks, exact repeated trace outputs and program-cache guards. The original top-k/softmax/scatter selection tail and per-expert scaling remain unchanged. The explicit native program is identical on both sides, preventing input placement from changing automatic geometry. Both variants keep FP32 outputs in DRAM.",
        "",
        "HiFi2 changes only fidelity. Producer-L1 changes only the existing scaled-input multiply output placement and retains HiFi4, including the fused scalar-root activation. The report’s generic BFP8 fidelity label is inaccurate for this BF16 weight; the independent candidate was nevertheless measured.",
        "",
        report["statistical_scope"],
        "",
        "The JSON retains exact commands, runtime/helper/probe hashes, fixture hashes, all128 pairs, compute/program metadata, full timing samples and before/after memory observations. These runs bind the formatted paired helper; prior QKV runs resolve to its separately preserved historical snapshot.",
        "",
        report["metadata_limit"],
        "",
        "Remaining applicable prefill-router advice: none. Integrated default validation and native profiling use the unchanged selected runtime.",
    ]
    (ROOT / "prefill_router_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(controls=len(results), remaining=[])))


if __name__ == "__main__":
    main()
