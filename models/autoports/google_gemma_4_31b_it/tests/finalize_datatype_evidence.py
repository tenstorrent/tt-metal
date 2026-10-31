"""Reconcile selected-policy allocator evidence and write the stage report."""
import copy
import json
from pathlib import Path

from .summarize_datatype_sweep import collect, plots

DOC = Path("models/autoports/google_gemma_4_31b_it/doc")
ROOT = DOC / "datatype_sweep"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    policy = read(ROOT / "selected_precision_config.json")
    rows = collect()
    assert (
        max((r for r in rows if r["status"] == "pass"), key=lambda r: r["decode_tps"])["config_id"]
        == policy["config_id"]
    )
    plots(rows, policy["config_id"])
    ledger = copy.deepcopy(read(DOC / "optimized_full_model/memory_plan.json"))
    old_head = ledger["weights"][1]["representations"][1]["bytes_per_device"]
    perf_rows = []
    for capacity in ledger["capacity_rows"]:
        batch = capacity["batch"]
        folder = "selected_default" if batch == 1 else f"selected_b{batch}"
        memory = read(ROOT / folder / "memory.json")
        inv = memory["inventory"]
        prepared = memory["prepared"]
        banks = prepared["num_banks"]
        capacity.update(
            inventory=inv,
            source=f"datatype_sweep/{folder}/memory.json",
            constructed_bytes_per_device=inv["allocator_bytes"],
            prepared_bytes_per_device=banks * prepared["total_bytes_allocated_per_bank"],
            usable_dram_bytes_per_device=banks * prepared["total_bytes_per_bank"],
            largest_contiguous_bytes_free_per_bank=prepared["largest_contiguous_bytes_free_per_bank"],
        )
        capacity["free_bytes_after_prepare"] = (
            capacity["usable_dram_bytes_per_device"] - capacity["prepared_bytes_per_device"]
        )
        capacity["headroom_after_transient_budget"] = (
            capacity["free_bytes_after_prepare"] - ledger["reserves"]["budget_bytes_per_device"]
        )
        assert capacity["headroom_after_transient_budget"] > 0
        assert capacity["kv_bytes_per_device"] == inv["kv_cache"]
        perf = read(ROOT / folder / f"performance_b{batch}.json")
        assert perf["input_tokens"] == 4096 and perf["generated_tokens"] == 128 and perf["streaming_exact"]
        perf.pop("samples", None)
        perf["evidence"] = f"datatype_sweep/{folder}/performance_b{batch}.json"
        perf_rows.append(perf)
    new_head = ledger["capacity_rows"][0]["inventory"]["lm_head"]
    ledger["weights"][1]["representations"][1].update(
        bytes_per_device=new_head, dtype="BFP4 tiled transposed vocabulary shard"
    )
    ledger["weights"][1]["bytes_per_device"] = ledger["weights"][1]["representations"][0]["bytes_per_device"] + new_head
    ledger["weights"][1][
        "copy_reason"
    ] = "BF16 row-major hidden-sharded lookup and BFP4 tiled transposed vocabulary-sharded head cannot alias. Two construction-time representations avoid runtime repacking; selected token-out performance and allocator inventory are measured."
    ledger["weights"][1][
        "evidence"
    ] = "datatype_sweep/selected_default/memory.json; datatype_sweep/selected_default/performance_b1.json"
    ledger["model_weights_rope_ccl_bytes_per_device"] += new_head - old_head
    ledger["selected_precision_config"] = "datatype_sweep/selected_precision_config.json"
    ledger["kv_cache_dtype"] = policy["kv_cache_dtype"]
    ledger[
        "status"
    ] = "Selected full-stack persistent allocations measured at batches1/8/16/32; target32-agent prefix checked"
    ledger["reserves"][
        "stage08_basis"
    ] = "Retain stage07 conservative transient bound: decoder/activation/KV/output shapes and dtypes are unchanged; only head weights shrink from BFP8 to BFP4. Head output and sampling storage remain BF16. This is an analytical bound, not a new full-stack transient peak."
    long = read(ROOT / "selected_b32/long_prefix.json")
    assert long["all_slots_retrieve_label"] and long["batch"] == 32
    ledger["target_workload_evidence"] = "datatype_sweep/selected_b32/long_prefix.json"
    write(ROOT / "memory_plan.json", ledger)
    context = read(DOC / "context_contract.json")
    context["validated_stage"] = "08-datatype-sweep"
    context[
        "reference_scope"
    ] = "Full60-layer selected precision execution validates262144 context, including nonaligned262143. Historical layer PCC fields retain earlier HF comparisons; datatype_sweep/selected_default/readiness_accuracy.json is short full-model agreement, not a new long-context HF score."
    plan = context["full_model_plan"]
    plan.update(
        status=ledger["status"],
        capacity_rows=ledger["capacity_rows"],
        evidence="datatype_sweep/memory_plan.json",
        memory_ledger="datatype_sweep/memory_plan.json",
        model_weights_rope_ccl_bytes_per_device=ledger["model_weights_rope_ccl_bytes_per_device"],
        kv_cache_dtype=policy["kv_cache_dtype"],
        selected_precision_config="datatype_sweep/selected_precision_config.json",
    )
    plan["validation"]["evidence"] = "datatype_sweep/selected_default/full_context_post_barrier.json"
    plan["reserves"] = ledger["reserves"]
    write(DOC / "context_contract.json", context)
    agentic = read(DOC / "agentic_serving.json")
    agentic.update(
        stage="08-datatype-sweep",
        status="validation complete; independent review pending",
        ledger=ledger,
        datatype_sweep_measurements=perf_rows,
        stage_review="datatype_sweep/stage_review.md",
    )
    agentic["datatype_sweep_long_context"] = {
        "input_tokens": long["prompt_tokens"],
        "batch": 32,
        "all_slots_retrieve_label": True,
        "evidence": "datatype_sweep/selected_b32/long_prefix.json",
        "scope": "synthetic multi-turn retention/free-running reply; not a long-context HF score",
    }
    write(DOC / "agentic_serving.json", agentic)
    selected = next(r for r in rows if r["config_id"] == policy["config_id"])
    perf = perf_rows[0]
    repeats = read(ROOT / "selected_default/readiness_accuracy.json")["teacher_forcing_repeats"]
    rates = [repeat[0]["decode_t/s/u"] for repeat in repeats]
    lines = [
        "# Datatype sweep",
        "",
        f"Selected **{policy['config_id']}**: BFP4/LoFi decoder projections and LM head; BFP8 KV and prefill MLP/CCL values; BF16 residuals, decode CCL, logits and sampling values. No layer exceptions.",
        "",
        f"Default-path full-model agreement: top1 **{selected['top1']:.0%}**, top5 **{selected['top5']:.0%}**, top100 **{selected['top100']:.0%}**. Required gates: **90% / 98% / 100%**. Accuracy covers100 continuation positions from one AIME24 chat-template prompt, not the full dataset.",
        "",
        f"Traced teacher-forcing (161-token prompt,100 continuation positions,B1): TTFT **{selected['ttft_ms']:.2f}ms**, decode **{selected['decode_tps']:.3f}tokens/s/user**. Post-selection warmed token-out (**4096 input /128 output /batch1 /one request**): TTFT **{perf['ttft_ms']:.2f}ms**, decode **{perf['decode_tps']:.3f}tokens/s/user**. Later serving comparisons must use this token-out number. It includes first-token and final-history reads, with no per-token readback in the decode loop.",
        "",
        "## Evaluated full-model policies",
        "",
        "| Config | Minimum top1 | top5 | top100 | Traced TF t/s/u | Token-out t/s/u | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config_id']} | {r['top1']:.0%} | {r['top5']:.0%} | {r['top100']:.0%} | {r['decode_tps']:.3f} | {r['token_out']['decode_tps']:.3f} | {r['status']} |"
        )
    lines += [
        "",
        f"Top-k table/plot values conservatively use the minimum of prefill and decode agreement. All policies meet accuracy; BF16 KV fails the target working-set memory check. Other policies are rejected for speed. The selected default-path confirmation replaces its search point in the final plots; the original38.035TF measurement remains in head_bfp4/readiness_accuracy.json. The final confirmation repeats span{min(rates):.3f}–{max(rates):.3f}t/s/u; token-out measures{perf['decode_tps']:.3f}t/s/u. No eager timing ranks a policy.",
        "",
        "![Top1 Pareto](top1_perf_pareto.png)",
        "",
        "![Top5 Pareto](top5_perf_pareto.png)",
        "",
        "The top1 frontier trades the selected93% agreement for the slower99% BFP8/LoFi decoder control. At top5=100% the selected policy is fastest. Red marks selection; dotted lines mark the required gate. BFP4 LoFi/HiFi2 controls cover attention, gate/up, down and head. BFP8 LoFi/HiFi2 controls cover the same projection groups. Combined BFP8 decode-CCL plus the selected head also loses in both timing regimes.",
        "",
        "## Capability, memory and quality",
        "",
        "Advertised and supported context remains262144. selected_default/full_context_post_barrier.json proves262143/262144 prefill and decode at position262143. Non-aligned prompts and repeated sampled/greedy replay pass. Full target32-agent execution retains the requested label through32,083-token chat prefixes in every slot. The estimated workload remains32 agents ×32768 live tokens, typical context4096; the allocated pool is1048576 tokens. memory_plan.json reconciles selected allocator snapshots at all configured batches and preserves the conservative prior-stage transient reserve with its scope.",
        "",
        "BF16 KV initially failed an SDPA L1 admission check. AUTODEBUG.md derives the exact2048-byte overlap; AUTOFIX.md records the Q64/K128 repair, tracked non-aligned smoke and subsequent full-model result. It is rejected for target-pool capacity, not the initial error. No advertised capability was reduced.",
        "",
        "qualitative_verdict.md reviews all six prompt-correct shared outputs against pinned HF controls. The long-prefix JSON fence is reproduced by baseline and higher-precision controls; it is not strict tool-call conformance. Fixed128-token suite caps also truncate HF controls; no complete code/functionality claim is made from partial snippets.",
        "",
        "## Default construction and reproduction",
        "",
        "selected_precision_config.json is required by Gemma4Model and build_generator default construction. Missing artifacts fail instead of silently changing precision. configs/baseline_bfp4_lofi.json is the explicit safe-baseline override. runtime_policy.json files serialize actual weight dtypes and kernel attributes; runner assertions verify them. Residual, norm, decode-CCL backing, activation, KV, head, logits/sampling and context fields feed construction. Layer exceptions use the same dtype/fidelity aliases.",
        "",
        "activation_dtype denotes prefill gate/up outputs and their intermediates; remaining activation paths are BF16. Norms use HiFi4, attention HiFi2, head FP32 accumulation, decoder projection FP32 accumulation only for the BF16/HiFi4 alias; approximation is off and packer accumulation on. Sampling_dtype denotes BF16 values; token IDs/history remain UINT32. No MoE/router exists in this target.",
        "",
        "Run from this checkout after the installed model-bringup environment.py setup:",
        "",
        "```bash",
        "TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_DEVICE_PROFILER=0 python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_datatype_candidate --id selected_default --acceptance",
        "```",
        "",
        "Per-candidate commands, branch/commit, hardware, mesh, policies, traces and environment are in sweep_results.json / sweep_results.csv. All runs use local TP4 Blackhole p300c, mesh1×4. Performance batches are below; diagnostics-enabled state checks are separate. Source manifests and work_log.md preserve provenance. selected_acceptance_snapshot/source_manifest.json scopes the earlier full-context and qualitative acceptance artifacts; selected_default/source_manifest.json scopes the final accuracy/performance refresh. Selected branch equivalence is documented in work_log.md. No full-model device-time/roofline numbers are inferred from host times, and no new profiling was needed.",
        "",
        "| Batch / concurrent requests | Token-out TTFT ms | Token-out t/s/user |",
        "|---:|---:|---:|",
    ]
    for p in perf_rows:
        lines.append(f"| {p['batch_size']} | {p['ttft_ms']:.2f} | {p['decode_tps']:.3f} |")
    lines += [
        "",
        "VLLM integration is explicitly outside this stage. propagation_check.json proves the required default model/generator artifact; it makes no claim about an adapter that has not yet been implemented. Future serving must inherit this default and verify registration in stage09.",
        "",
        "## Acceptance status",
        "",
        "Validation complete; independent stage review pending. No push. Local checkpoint SHAs will be recorded in work_log.md after clean-pass.",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
