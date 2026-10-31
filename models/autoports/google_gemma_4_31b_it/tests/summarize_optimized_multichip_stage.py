"""Summarize stage05 measured artifacts and its compact local evidence packet."""
import json
import math
import re
from pathlib import Path

base = Path("models/autoports/google_gemma_4_31b_it")
root = base / "doc/optimized_multichip_decoder"
previous = base / "doc/multichip_decoder"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


rows = []
for kind in ("sliding_attention", "full_attention"):
    for batch in (1, 32, 16, 8):
        path = root / "tracy_final" / kind / f"batch{batch}/performance.json"
        if not path.exists():
            continue
        d = read(path)
        b = read(previous / "tracy_final" / kind / f"batch{batch}/performance.json")
        rows.append(
            dict(
                layer_type=kind,
                batch=batch,
                baseline_prefill_device_us=b["prefill_device_us"],
                baseline_decode_device_us=b["decode_device_us"],
                prefill_device_us=d["prefill_device_us"],
                decode_device_us=d["decode_device_us"],
                prefill_speedup=b["prefill_device_us"] / d["prefill_device_us"],
                decode_speedup=b["decode_device_us"] / d["decode_device_us"],
                evidence=str(path),
            )
        )
write(root / "performance_summary.json", rows)
text = "| Layer | Batch | Before prefill / decode device us | Final prefill / decode device us | Speedup pre / de |\n|---|---:|---:|---:|---:|\n"
for r in rows:
    text += f"| {r['layer_type']} | {r['batch']} | {r['baseline_prefill_device_us']:.2f} / {r['baseline_decode_device_us']:.2f} | {r['prefill_device_us']:.2f} / {r['decode_device_us']:.2f} | {r['prefill_speedup']:.3f} / {r['decode_speedup']:.3f} |\n"
(root / "performance.md").write_text(text)
candidates = []
checks = {}
for path in sorted(root.glob("*.json")):
    d = read(path)
    if not isinstance(d, dict) or "4" not in d:
        continue
    row = {
        k: d[k]
        for k in (
            "layer_type",
            "batch",
            "seq",
            "prefill_pcc",
            "decode_pcc",
            "stress_min_pcc",
            "runtime_fallback_audit",
            "cache_pcc",
            "unused_pages_unchanged",
            "candidate",
        )
        if k in d
    }
    row.update(
        prefill_host_us=d["4"]["prefill_warm_host_us"],
        decode_traced_host_us=d["4"]["decode_trace_host_us"],
        evidence=path.name,
    )
    candidates.append(row)
    if path.name.startswith("final_layer"):
        assert read(path.with_suffix(".exit.json"))["exit_code"] == 0
        assert min(d["prefill_pcc"], d["decode_pcc"]) >= 0.995
        checks[path.name] = row
write(root / "candidate_summary.json", candidates)
table = [
    "# Whole-layer host timing controls",
    "",
    "Host elapsed includes synchronization; these values are not device-time rooflines.",
    "",
    "| Artifact | Layer | Batch | Sequence | Prefill us | Traced decode us | Prefill PCC | Decode PCC |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in candidates:
    table.append(
        f'| {r["evidence"]} | {r["layer_type"]} | {r["batch"]} | {r["seq"]} | {r["prefill_host_us"]:.2f} | {r["decode_traced_host_us"]:.2f} | {r["prefill_pcc"]:.8f} | {r["decode_pcc"]:.8f} |'
    )
(root / "candidate_timings.md").write_text("\n".join(table) + "\n")
tails = []
for layer in (0, 5):
    for seq in (33, 1024, 1025, 2049):
        name = f"final_layer{layer}_length{seq}.json"
        if not (previous / name).exists() or not (root / name).exists():
            continue
        before, after = read(previous / name), read(root / name)
        tails.append(
            dict(
                layer_type=after["layer_type"],
                seq=seq,
                batch=1,
                before_prefill_host_us=before["4"]["prefill_warm_host_us"],
                after_prefill_host_us=after["4"]["prefill_warm_host_us"],
                before_decode_host_us=before["4"]["decode_trace_host_us"],
                after_decode_host_us=after["4"]["decode_trace_host_us"],
                evidence=[str(previous / name), str(root / name)],
            )
        )
write(root / "short_tail_comparison.json", tails)
search = [
    "# Dominant decode matmul search",
    "",
    "All rows use real TP4 shapes and BFP4/LoFi. Percentages are diagnostic matmul contributions to the final whole-layer device window, averaged across devices; gate/up is their sum. Trial latency is whole-layer traced host time, so it is not compared numerically to a device subtotal. Full candidate layouts and source hashes are in the named JSON/manifests.",
    "",
    "| Kind | Role (% of layer) | Sender cores / K block tried | Layer trace us | Prefill / decode PCC | Decision / evidence |",
    "|---|---|---|---:|---|---|",
]
for layer, kind in ((0, "sliding_attention"), (5, "full_attention")):
    profile = root / "tracy_final" / kind / "batch1/performance.json"
    if not profile.exists():
        continue
    runtime = read(profile)["matmul_runtime_rows"]
    for role, shape_role, trials, selected in (
        (
            "QKV" if layer == 0 else "QK",
            "qkv_local" if layer == 0 else "qk_local",
            ("q12", "q24"),
            "28 / 6" if layer == 0 else "8 / 21",
        ),
        ("O", "o_local", ("o16",), "8 / 8" if layer == 0 else "8 / 16"),
        ("gate/up", "gate_or_up_local", ("gate12", "gate24"), "28 / 6"),
        ("down", "down_local", ("down22",), "8 / 11"),
    ):
        fraction = sum(r["fraction_of_whole_layer_pct"] for r in runtime if r["role"] == shape_role) / 4
        final = read(root / f"final_layer{layer}_batch1.json")
        search.append(
            f'| {kind} | {role} ({fraction:.2f}%) | {selected}, 2 readers | {final["4"]["decode_trace_host_us"]:.2f} | {final["prefill_pcc"]:.8f} / {final["decode_pcc"]:.8f} | Kept final default; final_layer{layer}_batch1.json |'
        )
        for trial in trials:
            path = root / f"{trial}_layer{layer}_batch1.json"
            d = read(path)
            geometry = next(iter(d["candidate"]["geometry"].values()))
            search.append(
                f'| {kind} | {role} ({fraction:.2f}%) | {geometry["cores"]} / {geometry["block"]}, {geometry["readers"]} readers | {d["4"]["decode_trace_host_us"]:.2f} | {d["prefill_pcc"]:.8f} / {d["decode_pcc"]:.8f} | Rejected: slower than final cumulative family; {path.name} |'
            )
search.append(
    "\nFull QK7/K24 additionally crossed the selected norm/pool/L1-slice family in norm8poolq24_layer5_batch1.json:478.77us vs475.04us, passing PCC; rejected. Reader counts have separate alternating-order microprofiles and stage04 whole-layer controls. Prefill M4/K8/N16/subblock1×8, larger-K/grid/padding trials and exact K21 L1 failure are recorded in optimization_audit.md and the stage04 artifacts."
)
(root / "dominant_matmul_search.md").write_text("\n".join(search) + "\n")
write(root / "acceptance_summary.json", checks)
failures = []
for path in sorted(root.glob("input_l1*.exit.json")):
    d = read(path)
    if d["exit_code"] == 0:
        continue
    log = path.with_name(path.name.replace(".exit.json", ".log"))
    content = log.read_text()
    matches = re.findall(r"L1 buffer allocated at (\d+) and static dataflow buffer region ends at (\d+)", content)
    failures.append(
        dict(
            artifact=log.name,
            exit_code=d["exit_code"],
            overlap=dict(live_buffer_start=int(matches[-1][0]), static_buffer_end=int(matches[-1][1]))
            if matches
            else None,
            capacity_lines=[line for line in content.splitlines() if line.startswith("INPUT_L1_CAPACITY")],
            allocation_errors=list(
                dict.fromkeys(line for line in content.splitlines() if line.startswith("Out of Memory:"))
            ),
        )
    )
write(root / "input_l1_failures.json", failures)
capacity_rows = []
for batch in (8, 16, 32):
    for role, width, tile_bytes in (
        ("Q/MLP input", 5376, 2048),
        ("sliding O input", 2048, 2048),
        ("full O input", 4096, 2048),
        ("down input", 5632, 1088),
    ):
        tiles = batch * 1024 * width // 1024
        per_bank = math.ceil(tiles / 110) * tile_bytes
        capacity_rows.append(
            dict(
                batch=batch,
                role=role,
                bytes_per_device=tiles * tile_bytes,
                tile_rounded_bytes_per_bank=per_bank,
                exceeds_total_bank_capacity=per_bank > 1461120,
            )
        )
write(
    root / "input_l1_capacity.json",
    dict(
        observed_banks=110,
        observed_total_bytes_per_bank=1461120,
        observed_initial_largest_contiguous_free_per_bank=1372800,
        total_capacity_bytes=110 * 1461120,
        basis="INPUT_L1_CAPACITY in input_l1_q_layer0_batch8.log. Tile headers included; input-only lower bounds, before matmul CB or other live allocations. Below total capacity is not evidence of runtime fit.",
        rows=capacity_rows,
    ),
)
memory = read(previous / "memory_plan.json")
previous_setup_overhead = memory["decoder_setup_overhead_per_device"]
memory["shared_decode_l1_pool"] = dict(
    bytes_per_device=688128,
    bytes_per_core=86016,
    shard_cores=8,
    backing_tensors=2,
    shape=[1, 1, 32, 5376],
    dtype="BF16",
    ownership="One dictionary shared by all sequential layers and batch signatures; live for all traces.",
    evidence="shared_workspace.json and tt/multichip_decoder.py",
)
for layer, kind in ((0, "sliding_attention"), (5, "full_attention")):
    path = root / f"final_layer{layer}_batch1.json"
    if not path.exists():
        continue
    d = read(path)["4"]
    w, c = d["weight_memory"], d["cache_memory"]
    allocated = w["num_banks"] * w["total_bytes_allocated_per_bank"]
    payload = memory["layer_types"][kind]["weight_bytes_per_device_layer"]
    memory["allocator_reconciliation"][kind] = dict(
        measured_setup_bytes_per_device=allocated,
        estimated_weight_tile_payload_bytes=payload,
        setup_overhead_bytes=allocated - payload,
        cache_allocation_bytes=c["num_banks"] * c["total_bytes_allocated_per_bank"] - allocated,
        available_dram_bytes_per_device=w["num_banks"] * w["total_bytes_per_bank"],
        source=str(path),
    )
memory["decoder_setup_overhead_per_device"] = sum(
    memory["layer_types"][kind]["layers"] * value["setup_overhead_bytes"]
    for kind, value in memory["allocator_reconciliation"].items()
)
memory["b1_full_prompt_8GiB_reserve_total"] += memory["decoder_setup_overhead_per_device"] - previous_setup_overhead
memory[
    "stage05_delta_basis"
] = "Weights/cache unchanged. AG output moves from per-layer/batch DRAM to two shared L1 backings. Existing 2GiB activation/trace reserve and 8GiB long-prefill reserve remain conservative; full-stack memory validation belongs to full-model construction."
write(root / "memory_plan.json", memory)
review = root / "stage_review.md"
clean = review.exists() and "Verdict: clean-pass" in review.read_text()
context = read(base / "doc/context_contract.json")
contract = context["optimized_multichip_contract"]
contract["status"] = "validated" if clean else "validation_in_progress"
contract["decoder_setup_overhead_bytes_per_device"] = memory["decoder_setup_overhead_per_device"]
contract["memory_evidence"] = "optimized_multichip_decoder/memory_plan.json"
contract["validation"] = {}
for layer, kind in ((0, "sliding_attention"), (5, "full_attention")):
    for seq in (262143, 262144):
        path = root / f"final_layer{layer}_context{seq}.json"
        if path.exists():
            d = read(path)
            contract["validation"].setdefault(kind, []).append(
                dict(
                    prefill_tokens=seq,
                    decode_position=min(seq, 262143),
                    prefill_pcc=d["prefill_pcc"],
                    decode_pcc=d["decode_pcc"],
                    comparison="Last32 prefill rows and decode after complete prefill execution",
                    evidence=str(path),
                )
            )
write(base / "doc/context_contract.json", context)
agentic = read(base / "doc/agentic_serving.json")
agentic.update(
    stage="05-optimized-multichip-decoder",
    status="validated" if clean else "validation_in_progress",
    optimized_multichip_measurements=rows,
)
write(base / "doc/agentic_serving.json", agentic)
packet_path = Path(
    "bringup/artifacts/multigoal-runs/20261030T163027Z/continuation-02-20261030T182350Z/telemetry/packets/a40cd227-5628-4d84-b89b-edd36b6c8bb0.json"
)
packet = read(packet_path.with_name(packet_path.stem + ".template.json"))
packet["selected_mesh"] = {"shape": [1, 4], "devices": 4, "hardware": "Blackhole p300c", "topology": "Ring"}
keys = (
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
)
for layer, kind in ((0, "sliding_attention"), (5, "full_attention")):
    path = root / "tracy_final" / kind / "batch1/performance.json"
    if path.exists():
        d = read(path)
        packet["layer_types"][kind] = {k: d[k] for k in keys if k in d}
        packet["layer_types"][kind]["evidence"] = str(path)
    else:
        packet["missing_reasons"][
            kind
        ] = "Final default4096/128/B1 whole-layer device measurement unavailable; work remains."
    for suffix in ("batch1", "context262144", "wrap2047"):
        path = root / f"final_layer{layer}_{suffix}.json"
        if not path.exists():
            continue
        d = read(path)
        for k, phase in (("prefill_pcc", "prefill"), ("decode_pcc", "decode"), ("stress_min_pcc", "decode_stress")):
            if k not in d:
                continue
            packet["accuracy"].append(
                dict(
                    name="real-weight TTNN baseline " + suffix,
                    value=d[k],
                    unit="PCC",
                    phase=phase,
                    layer_type=kind,
                    completed_samples=d["batch"],
                    dataset_samples=None,
                    scope="subset",
                    evidence=str(path),
                )
            )
packet["notes"] = [
    "Single-layer teacher-forced workload, not token-out generation. Whole-layer device windows include all operations and internal gaps.",
    "Dense BFP4/LoFi projections, BFP8 KV, BF16 residual/norm. Useful FLOPs count projections and causal attention; DRAM estimates count physical ordinary operands plus active KV/cache updates, including BFP headers and padding. Full basis is in the referenced performance JSON.",
    "Independent review clean-pass." if clean else "Stage05 validation/review in progress; no completion claim.",
    "Worker watcher and profiler run separately. Scoped Ethernet watcher limitation inherited from instrumented ACTIVE_ETH config-buffer capacity.",
]
write(packet_path, packet)
