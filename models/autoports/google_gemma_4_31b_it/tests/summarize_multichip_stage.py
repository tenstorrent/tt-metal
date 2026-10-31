"""Build compact stage summaries from measured artifacts; never infer missing metrics."""

import json
from pathlib import Path

base = Path("models/autoports/google_gemma_4_31b_it")
root = base / "doc/multichip_decoder"
review = root / "stage_review.md"
clean_review = review.exists() and "Verdict: clean-pass" in review.read_text()
status = "validated" if clean_review else "validation_in_progress"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


rows = []
for kind in ["sliding_attention", "full_attention"]:
    for batch in [1, 32, 16, 8]:
        selected = root / "tracy_final" / kind / f"batch{batch}" / "performance.json"
        baseline = (
            base
            / (
                "doc/optimized_decoder/tracy_final_k32"
                if kind == "full_attention"
                else "doc/optimized_decoder/tracy_final"
            )
            / kind
            / f"batch{batch}"
            / "performance.json"
        )
        if not selected.exists():
            continue
        d, b = read(selected), read(baseline)
        row = dict(
            layer_type=kind,
            batch=batch,
            prefill_device_us=d["prefill_device_us"],
            decode_device_us=d["decode_device_us"],
            baseline_prefill_device_us=b["prefill_device_us"],
            baseline_decode_device_us=b["decode_device_us"],
            prefill_speedup=b["prefill_device_us"] / d["prefill_device_us"],
            decode_speedup=b["decode_device_us"] / d["decode_device_us"],
            prefill_efficiency=b["prefill_device_us"] / d["prefill_device_us"] / 4,
            decode_efficiency=b["decode_device_us"] / d["decode_device_us"] / 4,
            prefill_flops_pct=d["prefill_flops_pct"],
            decode_dram_pct=d["decode_dram_pct"],
            source=str(selected),
            baseline_source=str(baseline),
        )
        rows.append(row)
write(root / "performance_summary.json", rows)
text = "| Layer | Batch | Baseline prefill / decode (device us) | TP4 prefill / decode (device us) | Speedup pre / de | Efficiency pre / de |\n|---|---:|---:|---:|---:|---:|\n"
for r in rows:
    text += f"| {r['layer_type']} | {r['batch']} | {r['baseline_prefill_device_us']:.1f} / {r['baseline_decode_device_us']:.1f} | {r['prefill_device_us']:.1f} / {r['decode_device_us']:.1f} | {r['prefill_speedup']:.2f} / {r['decode_speedup']:.2f} | {r['prefill_efficiency']:.1%} / {r['decode_efficiency']:.1%} |\n"
(root / "performance.md").write_text(text)
checks = {}
for path in sorted(root.glob("final_layer*.json")):
    if ".manifest." in path.name or ".exit." in path.name:
        continue
    d = read(path)
    exit_record = read(path.with_name(path.stem + ".exit.json"))
    assert exit_record["exit_code"] == 0, (path, exit_record)
    checks[path.name] = {
        k: d[k]
        for k in [
            "layer_type",
            "batch",
            "seq",
            "prefill_pcc",
            "decode_pcc",
            "cache_pcc",
            "unused_pages_unchanged",
            "stress_min_pcc",
            "runtime_fallback_audit",
        ]
        if k in d
    }
    assert d["prefill_pcc"] >= 0.995 and d["decode_pcc"] >= 0.995
write(root / "acceptance_summary.json", checks)
plan = read(root / "memory_plan.json")
plan["allocator_reconciliation"] = {}
for layer, kind in [(0, "sliding_attention"), (5, "full_attention")]:
    path = root / f"final_layer{layer}_batch1.json"
    if not path.exists():
        continue
    d = read(path)["4"]
    w = d["weight_memory"]
    k = d["cache_memory"]
    measured = (w["total_bytes_allocated_per_bank"] - d["empty_allocated_per_bank"]) * w["num_banks"]
    expected = plan["layer_types"][kind]["weight_bytes_per_device_layer"]
    plan["allocator_reconciliation"][kind] = dict(
        measured_setup_bytes_per_device=measured,
        estimated_weight_tile_payload_bytes=expected,
        setup_overhead_bytes=measured - expected,
        cache_allocation_bytes=(k["total_bytes_allocated_per_bank"] - w["total_bytes_allocated_per_bank"])
        * w["num_banks"],
        available_dram_bytes_per_device=w["total_bytes_per_bank"] * w["num_banks"],
        source=str(path),
    )
plan["decoder_setup_overhead_per_device"] = sum(
    plan["layer_types"][k]["layers"] * v["setup_overhead_bytes"] for k, v in plan["allocator_reconciliation"].items()
)
fixed = (
    plan["decoder_weights_per_device"]
    + plan["decoder_setup_overhead_per_device"]
    + plan["embedding_head_reserve_per_device"]
    + plan["trace_activation_reserve_per_device"]
)
plan["windowed_total_per_device"] = fixed + plan["windowed_kv_per_device"]
plan["total_estimate_per_device"] = fixed + plan["full_context_kv_per_device"]
for row in plan["batch_context_capacity"]:
    row["estimated_bytes"] = fixed + row["batch"] * sum(
        v["layers"] * v["kv_bytes_per_token_device"] * (2048 if k == "sliding_attention" else row["context"])
        for k, v in plan["layer_types"].items()
    )
plan[
    "setup_overhead_basis"
] = "Measured setup allocation less weight tile payload; includes persistent CCL buffers for B1/8/16/32, setup indices and allocator rounding. Scaled by layer counts, separate from activation/trace reserve."
plan["b1_full_prompt_8GiB_reserve_total"] = plan["windowed_total_per_device"] + 6 * 1024**3
write(root / "memory_plan.json", plan)
contract = read(base / "doc/context_contract.json")
contract["multichip_plan"].update(
    status=status,
    topology="Ring",
    sliding_physical_tokens=2048,
    decoder_weight_bytes_per_device_estimate=plan["decoder_weights_per_device"],
    decoder_setup_overhead_bytes_per_device=plan["decoder_setup_overhead_per_device"],
    full_stack_windowed_total_bytes_estimate=plan["windowed_total_per_device"],
    full_stack_unbounded_kv_total_bytes_estimate=plan["total_estimate_per_device"],
    evidence="multichip_decoder/memory_plan.json",
)
contract["multichip_plan"]["validation"] = {}
for layer, kind in [(0, "sliding_attention"), (5, "full_attention")]:
    contract["multichip_plan"]["validation"][kind] = []
    for seq in [262143, 262144]:
        path = root / f"final_layer{layer}_context{seq}.json"
        if path.exists():
            d = read(path)
            contract["multichip_plan"]["validation"][kind].append(
                dict(
                    prefill_tokens=seq,
                    decode_position=262143,
                    prefill_pcc=d["prefill_pcc"],
                    decode_pcc=d["decode_pcc"],
                    comparison="last32 prefill rows and decode, after complete prefill execution",
                    evidence=str(path),
                )
            )
write(base / "doc/context_contract.json", contract)
agentic = read(base / "doc/agentic_serving.json")
agentic.update(
    stage="04-multichip-decoder",
    status=status,
    multichip_measurements=rows,
    multichip_memory_plan="multichip_decoder/memory_plan.json",
    multichip_cache_contract="multichip_decoder/README.md",
)
write(base / "doc/agentic_serving.json", agentic)
packet_path = Path(
    "bringup/artifacts/multigoal-runs/20261030T163027Z/continuation-02-20261030T182350Z/telemetry/packets/9630fa21-6741-4688-bae0-7db3f63850d7.json"
)
packet = read(packet_path.with_name(packet_path.stem + ".template.json"))
packet["available_hardware"] = read(root / "available_hardware.json")
packet["selected_mesh"] = {"shape": [1, 4], "devices": 4, "hardware": "Blackhole P300_X2", "topology": "Ring"}
keys = [
    "prefill_device_us",
    "decode_device_us",
    "prefill_flops_pct",
    "decode_dram_pct",
    "prefill_useful_flops",
    "decode_dram_bytes",
    "peak_flops_per_s",
    "peak_dram_bytes_per_s",
    "peak_basis",
    "source",
]
for layer, kind in [(0, "sliding_attention"), (5, "full_attention")]:
    path = root / "tracy_final" / kind / "batch1/performance.json"
    if path.exists():
        d = read(path)
        packet["layer_types"][kind] = {k: d[k] for k in keys}
        packet["layer_types"][kind]["evidence"] = str(path)
    else:
        packet["missing_reasons"][kind] = "Final4096/128/B1 device profile pending."
    for suffix, scope in [("batch1", "full"), ("context262144", "subset"), ("stress32", "subset")]:
        path = root / f"final_layer{layer}_{suffix}.json"
        if not path.exists():
            continue
        d = read(path)
        for key, phase in [("prefill_pcc", "prefill"), ("decode_pcc", "decode"), ("stress_min_pcc", "decode_stress")]:
            if key not in d:
                continue
            packet["accuracy"].append(
                dict(
                    name="TTNN optimized-baseline " + suffix,
                    value=d[key],
                    unit="PCC",
                    phase=phase,
                    layer_type=kind,
                    completed_samples=d["batch"],
                    dataset_samples=None,
                    scope=scope,
                    evidence=path.name,
                )
            )
packet["notes"] = [
    "Dense single-layer teacher-forced workload. Whole-layer device windows include all operations and internal gaps across participating devices. No end-to-end token generation or serving in this stage.",
    (
        "Independent stage review clean-pass; see stage_review.md and local work_log.md."
        if clean_review
        else "Stage validation/review in progress; see local work_log.md."
    ),
    "Worker-core watcher passes with Ethernet instrumentation disabled: ACTIVE_ETH instrumentation exceeds the build configuration buffer by 2640 bytes. See watcher_configuration.json.",
]
write(packet_path, packet)
