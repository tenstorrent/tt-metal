"""Preserve an incomplete accuracy attempt and its timed serving continuation."""
import datetime
import json
import shutil
import time
from pathlib import Path

from benchmark_stage.report import write_report


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "doc/benchmark/run"
    continuation = out / "performance"
    original = json.loads((out / "accuracy_attempt_summary.json").read_text())
    perf = json.loads((continuation / "summary.json").read_text())
    config = json.loads((out / "run_config.json").read_text())
    timeline = json.loads((out / "stage_timeline.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    for source in continuation.iterdir():
        if source.is_symlink() or source.name in ("summary.json", "REPORT.md", "run_config.json", "manifest.json"):
            continue
        if source.is_file():
            target = out / source.name
            if target.exists() and target.read_bytes() != source.read_bytes():
                prior = out / "accuracy_attempt" / source.name
                if not prior.exists():
                    prior.parent.mkdir(exist_ok=True)
                    shutil.copyfile(target, prior)
            shutil.copyfile(source, target)
    summary = dict(original)
    summary.update(
        status="failed",
        error="Incomplete accuracy: original shared pass stopped within the original one-hour budget; no complete selected-set upstream scores are available.",
        performance=perf.get("performance", {}),
        performance_continuation_status=perf["status"],
        performance_continuation_error=perf.get("error"),
        accuracy_partial={},
    )
    for task in config["tasks"]:
        transcript = out / task / "responses.jsonl"
        responses = [json.loads(line) for line in transcript.read_text().splitlines()] if transcript.exists() else []
        reasons = {}
        for response in responses:
            reason = response["choices"][0].get("finish_reason", "missing")
            reasons[reason] = reasons.get(reason, 0) + 1
        group = manifest["groups"][task]
        summary["accuracy_partial"][task] = dict(
            responses=len(responses),
            selected=group["sample_count"],
            population=group["population"],
            finish_reasons=reasons,
            missing=group["sample_count"] - len(responses),
        )
    write_report(out, config, summary)
    report = (out / "REPORT.md").read_text()
    rows = []
    for task, item in summary["accuracy_partial"].items():
        ref = "80.80" if task.startswith("gpqa") else "Unavailable"
        source = (
            "[model card](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)"
            if task.startswith("gpqa")
            else "Unavailable"
        )
        metric = (
            "exact_match,flexible-extract" if task.startswith("gpqa") else "Upstream strict/loose prompt/instruction"
        )
        rows.append(
            f"| {task} | {item['responses']} returned; {item['selected']} selected / {item['population']} full | {metric} | Incomplete | {ref} | Unavailable | {source} |"
        )
    anchor = "|---|---:|---|---:|---:|---:|---|\n"
    report = report.replace(anchor, anchor + "\n".join(rows) + "\n", 1)
    report = report.replace(
        "Status: failed. Accuracy concurrency: 32.",
        "Status: incomplete (failed completion gate). Accuracy concurrency: 32.",
    )
    report = report.replace("Prefill FLOP roofline % (est.)", "Prefill FLOP roofline % (useful work est.)")
    report = report.replace("Decode DRAM roofline % (est.)", "Decode DRAM roofline % (minimum work est.)")
    report = report.replace(
        "| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |\n|---|---:|---:|---:|---:|\n\nAccuracy tasks share one request pool; their wall times refer to the same interval.\n\n",
        "",
    )
    report += "\n## Incomplete accuracy and protocol\n\n"
    report += (
        "All 384 frozen questions remain in the selected denominator and in [benchmark_inputs.jsonl](benchmark_inputs.jsonl). "
        "The shared request pool was interrupted at 17:54:01 UTC to preserve time for both required serving profiles and reporting. "
        "Unfinished requests were cancelled; requests still queued in the client were not sent. Missing answers are not scored as model errors. "
        "Completed-only upstream diagnostics, when present, are explicitly partial and are not a benchmark score or a published-score delta.\n\n"
        "| Task | Returned | Missing | Stop | Token limit |\n|---|---:|---:|---:|---:|\n"
    )
    for task, item in summary["accuracy_partial"].items():
        report += f"| {task} | {item['responses']} | {item['missing']} | {item['finish_reasons'].get('stop', 0)} | {item['finish_reasons'].get('length', 0)} |\n"
    partial_path = out / "partial_scores_summary.json"
    if partial_path.exists():
        partial = json.loads(partial_path.read_text())
        report += "\nCompleted-only diagnostics (completion-biased; no comparison delta):\n\n"
        for task, entry in partial["tasks"].items():
            for metric, score in entry["completed_only_diagnostic_scores"].items():
                report += f"- {task}, {metric}: {100 * score:.2f}% across {entry['completed_responses']} returned responses.\n"
        report += "\n[Upstream scorer settings and diagnostic summary](partial_scores_summary.json); [all questions, responses and per-question scores/nulls](partial_scores.jsonl).\n"
    report += (
        "\nBoth upstream variants use zero-shot structured chat, native pinned HF template applied exactly once, high reasoning effort, "
        "temperature 1.0, top_p 0.95, sampling, 32768 generated-token cap, native EOS IDs [1, 250019], and no textual stops. "
        "[Native tokenization checks](native_tokenization_checks.json) compare server tokens against one offline template application. "
        "GPQA uses seed-0 choice shuffling and upstream strict/flexible extraction; IFEval uses upstream strict/loose prompt and instruction scoring. "
        "The lm-eval version is 0.4.13. Document IDs, document hashes, full-population hashes and task variants are in [manifest.json](manifest.json); "
        "the manifest hash is `1d9756014d50f3ccf6aa9b3d1447c49000f495395b56d5ce75bb336b279f64d4`. "
        "Raw responses and request links remain in each task directory. Accuracy acceptance belongs to the bringup owner; this report makes no accuracy verdict.\n\n"
        "## Measurement identity and accounting limits\n\n"
        "Both serving profiles preserve all 48 layers, model/tokenizer revision `e9ce717642b2074082875be233c02c729ba47960`, "
        "`head4_ccl4_norm4_lofi` precision, P300x2 (four Blackhole ASICs, TP/EP 1x4), and 524288-token context capacity. "
        "Actual imported autoport path, constructed layers/tensor precision, command lines and source commits are recorded in "
        "[identity.json](../identity.json), configuration snapshots and runtime manifests. The one-slot profile uses the validated Stage 10 settings; "
        "the 32-slot profile changes capacity to 32. Both disable prefix caching, use distinct random prompts, greedy output and ignore EOS. "
        "Warmup requests are excluded from measured performance and included in stage wall time. Every performance request must complete with actual 4096 input and 128 output tokens.\n\n"
        "Phase time covers ordered asynchronous completion boundaries, host preparation/sampling, communication, all chunks and intervening gaps. "
        "Each gap is attributed to the next completed event's phase. The envelope starts at the first model-runner execute call and ends at final asynchronous completion/host sampling; "
        "HTTP admission/tokenization before that call and final network delivery are represented in client end-to-end latency, outside the phase envelope. "
        "No device synchronization or live device profiler was added. Completion polling is nominally 0.5 ms and host scheduling can add observation delay; "
        "the event-query timestamp is taken just before querying, within that observation uncertainty. "
        "Prefill counts useful active-expert FLOPs and uses a fidelity-weighted nominal hardware ceiling. "
        "Decode uses modeled minimum unique DRAM transfers, with actual padded tensor dtypes, positions and batches. "
        "It is not measured physical bandwidth: unknown expert unions and repeated transfers make physical traffic uncertain. "
        "Upper expert-union scenarios and component accounting are retained in each roofline accounting JSON. "
        "See [work accounting](../work_accounting.md).\n\n"
        "## Timing, checks and ownership\n\n"
        "Stage 10 had stopped its server at 17:09:31 UTC; no live server was supplied. The first 32-slot launch, load, compilation and warmup are charged "
        "to this stage. Setup before invocation is listed separately in [RUN_NOTES.md](../RUN_NOTES.md). "
        "The performance continuation reuses the owned 32-slot server and does not reset the stage clock. "
        "The original failure summary/report and continuation summary/report remain archived. "
        "The evidence gate must fail because the selected accuracy subsets are incomplete; performance-only validation and the unchanged full-context check are recorded separately. "
        "Only stage-owned processes are stopped; no device reset or external reservation release is performed.\n"
    )
    report += "\n| Server slots | Launch through readiness seconds, charged to stage |\n|---:|---:|\n"
    for capacity in (32, 1):
        snapshot = json.loads((out / f"configuration-b{capacity}.json").read_text())
        report += f"| {capacity} | {snapshot['startup_seconds']:.2f} |\n"
    report += (
        "\nIndependent [performance validation](performance_validation.json) verifies complete requests, exact token counts, full phase coverage, "
        "runtime/precision identity and stable hashes after owned-server shutdown. [Distinct prompt proof](performance_prompt_validation.json) "
        "uses the exact benchmark client and pinned tokenizer. The client's one-second-bucket overlap field is not instantaneous concurrency; "
        "exact request timestamp overlap is 32 and 1 respectively. "
        "The [context check](context_check_final.log) passes. The [stage evidence check](evidence_check.log) returns its generic incomplete-stage/budget error "
        "because status is failed from missing accuracy; the cumulative wall time itself remains below one hour. "
        "[Dependency and checkpoint hashes](binary_dependency_provenance.json) supplement the launch records.\n\n"
        "Raw evidence is retained locally and committed byte-for-byte in [archive chunks and hash index](archive/index.json). "
        "After a fresh checkout, run `python models/autoports/ifm_k2_horizon_mova_36b_a4b/tests/benchmark_restore.py` to expand and verify it. "
        "The local implementation checkpoint is `3657e65a10eb6b706ee5a71a8c9f8ca77b031952`; no changes were pushed.\n"
    )
    if perf["status"] != "completed":
        report += f"\nPerformance continuation also failed: {perf.get('error')}.\n"
    report_path = out / "REPORT.md"
    report_path.write_text(report)
    # The elapsed clock includes report writing, not just inference/subrun clocks.
    finished = time.time()
    summary["elapsed_seconds"] = finished - timeline["original_invocation_start_unix_conservative"]
    timeline["report_finished_utc"] = datetime.datetime.fromtimestamp(finished, datetime.timezone.utc).isoformat()
    timeline["cumulative_wall_seconds_through_report"] = summary["elapsed_seconds"]
    timeline["under_original_budget"] = finished < timeline["original_deadline_unix"]
    with report_path.open("a") as stream:
        stream.write(
            f"\nCumulative client-stage wall time through reporting: **{summary['elapsed_seconds']:.1f} seconds**, "
            f"from {timeline['original_invocation_start_utc']} to {timeline['report_finished_utc']}. "
            "This includes both runner invocations, the gap between them, all launches/reloads, compilation, warmups, verification and collection. "
            "The conservative start bound precedes the original runner configuration write by one second. "
            "[Clock evidence](stage_timeline.json).\n"
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "stage_timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")


if __name__ == "__main__":
    main()
