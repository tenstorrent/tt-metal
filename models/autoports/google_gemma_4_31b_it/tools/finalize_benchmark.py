#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Append the stage contract and coverage to the measured upstream report."""

import gzip
import json
import shutil
import time
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
BASE = MODEL / "doc/benchmark"
RUN = BASE / "run"


def main():
    summary = json.loads((RUN / "summary.json").read_text())
    timing = json.loads((BASE / "stage_timing.json").read_text())
    protocol = json.loads((BASE / "protocol.json").read_text())
    report = RUN / "REPORT.md"
    upstream = RUN / "runner-REPORT.md"
    if not upstream.exists():
        shutil.copy2(report, upstream)
    text = upstream.read_text()
    header = "|---|---:|---|---:|---:|---:|---|"
    coverage_rows = """
| GPQA Diamond (zero-shot CoT planned) | 0 / 198; frozen subset 128 | Not covered: gated dataset credentials unavailable, operator decision 5 | — | 84.3 | — | [Exact-model card](https://ai.google.dev/gemma/docs/core/model_card_4) |
| SWE-bench Verified | 0 / 500 | Not covered: tool-use harness, Stage 12 | — | Unavailable | — | — |
| Terminal-Bench 2.1 | 0 / 89 | Not covered: terminal harness, Stage 12 | — | Unavailable | — | No substitution of 2.0 scores |"""
    text = text.replace(header, header + coverage_rows, 1)
    status_line = (
        f"Status: {summary['status']}. Accuracy concurrency: {summary.get('accuracy_concurrency', 'unrecorded')}."
    )
    text = text.replace(status_line + "\n\n", "")
    text = text.replace(
        "Scores use fixed subsets; published figures cover the full dataset.",
        "No local accuracy score was produced. The frozen subset and population remain preserved; published figures are references only.",
    )
    text = text.replace(
        "No local accuracy score was produced.",
        f'**Measurement status: {summary["status"]}.** No local accuracy score was produced.',
    )
    text += """

Target workload beside these profiles: 32 concurrent multi-turn agents, 32768 live tokens per agent and typical context 4096 (Stage 10 estimate pending Stage 12). The measured profiles use distinct single-turn 4096-token inputs and exactly 128 generated tokens. They do not cover long-history prefix reuse, tool execution, agent waiting or the growing live working set. B1/C1 is the single-user regression and telemetry headline.

Protocol: upstream vLLM 0.26.0 benchmark client, serial use of all four Blackhole ASICs, P300x2 TP4 mesh1x4; generated autoport with all 60 layers and selected head_bfp4 policy. Full context remains 262144. Prefix caching enabled; concurrency equals server slots. Each profile warms N requests and measures max(8,3N) complete requests with distinct seeded random prompts, temperature 0 and ignore_eos=true. The client's readiness probe and extra warmups are explicitly disabled; the separate collector probe and runner warmup are outside the measured window. Raw input_lens/output_lens establish actual4096/128 lengths for every request. Performance is plain token-input completion; it intentionally does not apply a chat template. Server defaults retain checkpoint temperature1/top_p0.95 and operator top_k32 override. The official model card recommends top_k64; the served32 and intended release GPQA20 are explicit protocol differences. No live device profiler or new device synchronization was enabled.

The frozen GPQA manifest retains all128 selected document identifiers/content hashes, full198 population and population hash, shuffle seed0 and zero few-shot examples. The dataset was not downloaded using an unofficial mirror. No accuracy questions were sent and no upstream scores or raw accuracy responses exist. Accuracy stop/truncation counts are unavailable, not inferred. The intended accuracy policy is structured messages with the native pinned HF chat template applied exactly once, enable_thinking=true, temperature1, top_p0.95, top_k20, max_gen_toks126976, EOS retained and no textual stops. The available lm-eval0.4.13 gpqa_diamond_cot_zeroshot exact-match scorer differs from the release r1_gpqa_diamond extractor; its3600s request timeout differs from release14400s. These are recorded protocol gaps, not reference parity. No accuracy acceptance verdict or score threshold is applied.

Phase accounting uses consecutive adapter-observed actual completions. Each disjoint interval is assigned to its ending phase; plugin host sampling, output construction, scheduling, communication and idle gaps after the previous return lie in the next interval. The first interval starts with the first measured client request; the final client tail belongs to the final phase. Existing async readback events are awaited by the unchanged plugin before adapter host processing; no synchronization was added. This attribution is explicitly authorized by operator decision4. Submission-to-completion spans are not summed, avoiding overlap double-counting. Raw event populations and effective decode positions are retained. Useful prefill FLOPs omit pointwise operations and padding from the numerator while retaining their time; decode DRAM is a shape/precision-based estimate including weights, KV and modeled intermediate traffic. These are host-wall estimates, not device-time utilization. See per-profile assumptions and measured sampling-mode counts in roofline evidence.

Model and tokenizer revision:842da3794eaa0b77d5f08bae87a17459d91ff475. Each configuration-bN.json preserves actual process argv, source hashes, imported module path, all layer indices, precision, devices and API context capacity. Launch identity validates frozen source hashes and HEAD. Plugin runtime stays stock ca7a18246e4111629ed6cb4a7d1192002749fccf; local commit90f33d2e392eb8bfda6948df9dabe712a724cb1a contains authorized tests only. See setup/server_vllm_source.json for the installed server Python source comparison; build/native provenance is separately limited.

Before this invocation, setup installed clients, reviewed sources, prepared the frozen manifest, implemented and host-tested the collector, and committed its source. Setup wall time was not fully instrumented. No initial server was supplied: Stage10 cleanup and the startup ownership scan show none; every launch/startup is inside the timed stage. The earlier incomplete attempt is preserved under ../attempts/incomplete-96fb930d. The rejected PC-04 plugin patch remains unapplied historical evidence. GPQA noncoverage and model-owned accounting supersede the earlier blockers under operator decisions4–6.

The installed evidence checker has two stale assumptions: a3600s cap and nonempty accuracy. The local contract checker retains all identity/performance/roofline/context evidence requirements, explicitly validates authorized noncoverage, and uses1800s per configured batch. Its source hashes and exact corrections are retained in evidence_check.json; setup/workflow-check-contract.patch is the generic upstream correction. Installed skills remain read-only. The outer multigoal script still invokes that stale installed checker; its failure is reported separately from measurement validity.
"""
    roofline = json.loads((RUN / "roofline.json").read_text())
    counts = {}
    text += "\n| Profile | Actual prefill tokens | Active decode tokens | Host-sampled prefill steps | Device-sampled prefill steps | Device-sampled decode steps |\n|---|---:|---:|---:|---:|---:|\n"
    for batch in ("1", "8", "16", "32"):
        row = roofline[batch]
        path = RUN / row["prefill"]["evidence"]
        detail = json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())
        modes = {}
        for step in detail["steps"]:
            key = step["phase"] + ":" + step["sampling_mode"]
            modes[key] = modes.get(key, 0) + 1
        counts[batch] = modes
        text += f"| B{batch} | {detail['prefill_tokens_executed']} | {detail['decode_active_tokens_executed']} | {modes.get('prefill:host_compatibility', 0)} | {modes.get('prefill:device', 0)} | {modes.get('decode:device', 0)} |\n"
    (RUN / "sampling_modes.json").write_text(json.dumps(counts, indent=2) + "\n")
    text += "\n| Profile | Maximum active decode rows | Mean active decode rows | Full-batch decode steps / total |\n|---|---:|---:|---:|\n"
    for batch in ("1", "8", "16", "32"):
        path = RUN / roofline[batch]["decode"]["evidence"]
        detail = json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())
        active = [step["work"]["active_tokens"] for step in detail["steps"] if step["phase"] == "decode"]
        text += f"| B{batch} | {max(active)} | {sum(active)/len(active):.3f} | {active.count(int(batch))} / {len(active)} |\n"
    text += "\nScheduler ramp-up and completion tails are included; server slots are capacity, not a claim that every step is full.\n"
    text += "\nThese are measured-window step counts, not lifetime counters. Stock-plugin mixed intermediate prefill uses the operator-accepted host sampling path. Performance outputs all reach the requested128 tokens with EOS ignored; the upstream performance dump does not retain per-request finish_reason, so stop reasons are not inferred.\n"
    matrix = json.loads((BASE / "release_benchmark_matrix.json").read_text())
    text += "\nRelease matrix: adopted128/128/C1 row from same-device dense Qwen3-32B, without borrowing Gemma implementation settings. "
    release = matrix["rows"][0]
    text += f"Estimated {release['estimated_seconds']:.3f}s; actual {release.get('actual_seconds')}s for8 requests. See [release matrix](../release_benchmark_matrix.json).\n"
    text += "\n| Server slots | Startup inside budget (seconds) | Source commit |\n|---|---:|---|\n"
    for batch in ("32", "16", "8", "1"):
        server = json.loads((RUN / f"perf-b{batch}-server.json").read_text())
        text += f"| {batch} | {server['startup_seconds']:.3f} | {server['source_commits']['tt-metal']} |\n"
    text += "\nVerification: the corrected evidence gate and full-context check pass. The unchanged installed outer gate exits2 with `missing required accuracy tasks`; its narrow correction awaits the requested exception to the installed-package read-only rule. [Check results](checks.json).\n"
    text += "\nRaw log/event originals remain locally under run/. Large originals and command logs are also preserved byte-for-byte in gzip archives for Git; [raw_archives.json](raw_archives.json) maps filenames and SHA256 values. After a fresh checkout, restore originals with `python tools/archive_benchmark_raw.py doc/benchmark/run --restore` from the model directory before replaying evidence checks.\n"
    elapsed = time.monotonic() - timing["started_monotonic"]
    text += f'\nClient stage through this final report: {elapsed:.3f}s / {timing["budget_seconds"]}s, including launch/reload, warmup, verification, inference, collection and reporting. Upstream runner elapsed: {summary["elapsed_seconds"]:.3f}s.\n'
    report.write_text(text)
    (RUN / "final_timing.json").write_text(
        json.dumps(
            dict(timing, elapsed_seconds_through_report=elapsed, within_budget=elapsed < timing["budget_seconds"]),
            indent=2,
        )
        + "\n"
    )
    shutil.copy2(RUN / "perf-b32-server.json", BASE / "identity.json")
    shutil.copy2(BASE / "attempts/incomplete-96fb930d/run/tokenizer_protocol.json", RUN / "tokenizer_protocol.json")
    packet_dir = MODEL.parents[2] / "bringup/artifacts/multigoal-runs/v3-20261105T172350Z/telemetry/packets"
    name = "da3df348-3260-43bc-b199-d27bb7dcfaca"
    packet = json.loads((packet_dir / (name + ".template.json")).read_text())
    row = summary.get("performance", {}).get("1")
    packet["selected_mesh"] = {"shape": [1, 4], "devices": [0, 1, 2, 3]}
    packet["performance"] = {
        "source": "vllm",
        "prefill_device_us": None,
        "decode_device_us": None,
        "prefill_flops_pct": None,
        "decode_dram_pct": None,
    }
    if row:
        packet["performance"].update(
            ttft_ms=row["mean_ttft_ms"],
            decode_tps=1000 / row["mean_tpot_ms"],
            evidence="models/autoports/google_gemma_4_31b_it/doc/benchmark/run/perf-b1.json",
        )
    packet["accuracy"] = []
    packet["missing_reasons"] = {
        "device_metrics": "Live vLLM profiling prohibited; host-wall phase rooflines are reported separately and are not device-time measurements.",
        "accuracy": "GPQA credential unavailable; operator decision5 permits local noncoverage. SWE/Terminal tool suites require Stage12 harness.",
    }
    packet["notes"] = [
        "Full60layers, head_bfp4, context262144, prefix caching enabled.",
        "Four profiles use distinct4096-token inputs and128 outputs; B1/C1 is headline.",
        "Installed outer gate has documented stale budget/noncoverage assumptions; corrected evidence gate retained.",
    ]
    (packet_dir / (name + ".json")).write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "elapsed_seconds": elapsed}))


if __name__ == "__main__":
    main()
