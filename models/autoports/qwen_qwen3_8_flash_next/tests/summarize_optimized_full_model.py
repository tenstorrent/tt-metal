"""Reproduce compact stage-7 metrics from runner and reduced-profiler artifacts."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/optimized_full_model"


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def main():
    before = read(OUT / "before.json")
    after = read(OUT / "selected_full.json")
    assert after["phase"] == "complete"
    old = [r["perf"] for r in before["primary_128_128"]]
    new = after["primary_128_128"]
    median = statistics.median
    metrics = {}
    for name, rows in [("before", old), ("after", new)]:
        metrics[name] = dict(
            ttft_ms=1000 * median([r["ttft_s"] for r in rows]),
            decode_ms_per_token=1000 / median([r["decode_tps"] for r in rows]),
            decode_tps=median([r["decode_tps"] for r in rows]),
        )
    metrics["workload"] = dict(prompt_len=128, gen_len=128, batch=1)
    metrics["teacher_forcing"] = after["teacher_forcing"]
    metrics["prefill_accuracy"] = after["prefill"]
    metrics["logits_only_tps"] = dict(
        before=median([r["tps"] for r in before["traced_logits_only"]]),
        after=median([r["tps"] for r in after["traced_logits_only"]]),
    )
    metrics["ttft_improvement_percent"] = 100 * (1 - metrics["after"]["ttft_ms"] / metrics["before"]["ttft_ms"])
    metrics["prefill_comparison"] = [
        dict(
            prompt_len=r["prompt_len"],
            gen_len=r["gen_len"],
            ttft_ms={k: 1000 * median([v["ttft_s"] for v in item["samples"]]) for k, item in r["variants"].items()},
        )
        for r in after["prefill_comparison"]
    ]
    write("metrics.json", metrics)
    memory = read(OUT / "memory_capacity_plan.json")
    memory["actual_prepared"] = after["prepared_memory"]
    memory["actual_trace"] = after["trace_memory"]
    ordinary, trace = after["prepared_memory"], after["trace_memory"]
    memory["actual_total_including_reserved_trace_bytes"] = (
        ordinary["num_banks"] * ordinary["total_bytes_allocated_per_bank"]
        + trace["num_banks"] * trace["total_bytes_per_bank"]
    )
    memory["actual_free_ordinary_dram_bytes"] = ordinary["num_banks"] * ordinary["total_bytes_free_per_bank"]
    memory["actual_trace_payload_bytes"] = trace["num_banks"] * trace["total_bytes_allocated_per_bank"]
    memory["measurement_artifact"] = "selected_full.json"
    write("memory_capacity_plan.json", memory)
    contract_path = ROOT / "doc/context_contract.json"
    contract = read(contract_path)
    validation = contract["optimized_full_model_validation"]
    validation.update(
        accuracy={key: after[key] for key in ("prefill", "teacher_forcing")},
        long_context=dict(artifact="optimized_full_model/selected_full.json", **after["long_context"]),
        allocated_bytes_including_reserved_trace=memory["actual_total_including_reserved_trace_bytes"],
        free_ordinary_dram_bytes=memory["actual_free_ordinary_dram_bytes"],
        trace_payload_bytes=memory["actual_trace_payload_bytes"],
        batch32_tested_context=2048,
        batch32_evidence="optimized_full_model/batch32_full_selected.json",
        runtime_source_sha256=after["source_sha256"],
    )
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    lower = read(ROOT / "doc/full_model/layer_stack_lower_bound.json")
    lower.pop("full_model_measurement", None)
    lower["inherited_stack_decode_ms"] = lower["stack_decode_ms"]
    for row in lower["rows"]:
        layer = row["representative_layer"]
        path = OUT / f"repaired_layer_{layer}/timing.json"
        current = read(path)
        row["artifact"] = str(path.relative_to(OUT))
        row["prefill_ms"] = current["median_prefill_host_ms"]
        row["decode_ms"] = current["median_traced_decode_ms"]
    lower["stack_decode_ms"] = sum(row["count"] * row["decode_ms"] for row in lower["rows"])
    lower["stack_prefill_ms"] = sum(row["count"] * row["prefill_ms"] for row in lower["rows"])
    lower["full_model_measurement"] = dict(artifact="selected_full.json", **metrics["after"])
    lower["layer_stack_tps"] = 1000 / lower["stack_decode_ms"]
    lower["raw_full_model_minus_representative_stack_ms"] = (
        metrics["after"]["decode_ms_per_token"] - lower["stack_decode_ms"]
    )
    terminal_timing = read(OUT / "terminal_timing.json")
    sampler_ms = median(terminal_timing["trace_ms"]["greedy"])
    representative_three_ms = sum(row["decode_ms"] for row in lower["rows"])
    terminal_estimate_ms = median(terminal_timing["trace_ms"]["decode"]) - representative_three_ms + sampler_ms
    lower["terminal_estimate_ms"] = terminal_estimate_ms
    lower["stack_plus_terminal_estimate_ms"] = lower["stack_decode_ms"] + terminal_estimate_ms
    lower["token_out_over_budget_percent"] = 100 * (
        metrics["after"]["decode_ms_per_token"] / lower["stack_plus_terminal_estimate_ms"] - 1
    )
    lower["terminal_estimate_scope"] = (
        "Uninstrumented reduced model-to-logits median minus one representative layer of each kind, "
        "plus separately timed split greedy. Different activations/routes make this budgeting, "
        "not an exact full-stack terminal measurement. See raw profiler for measured head work."
    )
    write("layer_stack_lower_bound.json", lower)
    profiles = OUT / "tracy/selected_profile"
    if not (profiles / "provenance.json").exists():
        print(json.dumps(metrics, indent=2))
        return
    prov = read(profiles / "provenance.json")
    timing = read(OUT / "selected_profile.json")
    inherited = read(ROOT / "doc/optimized_multichip_decoder/s5_final4_performance_accounting.json")
    payloads = {r["layer"]: sum(r["logical_payload_bytes_per_device"].values()) for r in inherited["runs"]}
    terminal = dict(
        lm_head=2560 * 62080 * 2,
        terminal_mix_weights=2 * 2560 * 320 * 2,
        norm_gamma=4 * 640 * 2,
        token_embedding=640 * 2,
        local_logits_read=62080 * 2,
    )
    floor_reduced = (sum(payloads.values()) + sum(terminal.values())) / 512e6
    floor_full = (35 * payloads[0] + payloads[1] + 12 * payloads[3] + sum(terminal.values())) / 512e6
    reduced_device = max(r["firmware_span_ms"] for r in prov["windows"]["token_out"]["rank_windows"].values())
    reduced_host = median(timing["trace_ms"]["token_out"])
    assert 0 < reduced_device < reduced_host * 1.25, (reduced_device, reduced_host)
    accounting = dict(
        scope="Same reduced real-layer0/1/3 run, full physical cache and terminal path; not all48-layer device timing",
        artifact="tracy/selected_profile/provenance.json",
        roofline_ms=floor_reduced,
        device_firmware_span_ms=reduced_device,
        host_window_ms=reduced_host,
        dispatch_and_host_gap_ms=reduced_host - reduced_device,
        terminal_payload_bytes_per_device=terminal,
        decoder_payload_bytes_per_device=payloads,
        bandwidth_GB_s_per_chip=512,
        mesh_chips=4,
        roofline_assumptions=inherited["floor_scope"],
        reduced_window_positions=timing["window_positions"],
        caveat="Ideal balanced active-expert lower bound and stored-dtype payload; not measured DRAM bandwidth. QSA payload follows the unchanged stage5 full-capacity index-cache and selected-window accounting. Per-rank firmware spans are measured directly; mixed-rank table sums are separate.",
    )
    write("reduced_performance_accounting.json", accounting)
    summary = dict(
        workload=dict(profile="single_user_decode", prompt_len=128, gen_len=128, batch=1),
        ttft_ms=metrics["after"]["ttft_ms"],
        decode_ms_per_token_e2e=metrics["after"]["decode_ms_per_token"],
        decode_ms_per_token_device=None,
        roofline_ms_per_token_estimate=floor_full,
        layer_stack_ms=lower["stack_decode_ms"],
        stack_plus_terminal_estimate_ms=lower["stack_plus_terminal_estimate_ms"],
        token_out_over_budget_percent=lower["token_out_over_budget_percent"],
        reduced_same_run_accounting=accounting,
        named_limitations=[
            "All-layer device profiling intentionally prohibited; direct device accounting is the reduced same-run measurement, not an invented full-stack number.",
            "Representative layer-stack estimate uses different activations/expert routes than full autoregressive generation.",
            "Small-op and collective latency dominates the ideal balanced DRAM floor; preserve selected decoder policy and its measured rejection ledger.",
        ],
    )
    write("perf_summary.json", summary)
    mats = []
    with (profiles / "decode_perf_report.csv").open() as f:
        for row in csv.DictReader(f):
            if "Matmul" in row["OP Code"] or "Expert" in row["OP Code"]:
                mats.append(
                    {
                        k: row[k]
                        for k in [
                            "OP Code",
                            "Device Time",
                            "Math Fidelity",
                            "Input 0 Datatype",
                            "Input 1 Datatype",
                            "Input 0 Memory",
                            "Inner Dim Block Size",
                            "Output Subblock H",
                            "Output Subblock W",
                            "Advice",
                        ]
                    }
                )
    write("measured_matmul_policy.json", dict(source="tracy/selected_profile/decode_perf_report.csv", rows=mats))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
