#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report retained measurements without converting missing evidence into a pass."""
import json
import time
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
DOC = MODEL / "doc/benchmark"
RUN = DOC / "run"


def main():
    original = RUN / "runner-REPORT.md"
    if not original.exists() and (RUN / "REPORT.md").exists():
        original.write_bytes((RUN / "REPORT.md").read_bytes())
    partial = RUN / "partial-performance"
    summary = (
        json.loads((partial / "summary.json").read_text())
        if (partial / "summary.json").exists()
        else {"performance": {}}
    )
    profiles = summary["performance"]
    protocol = json.loads((DOC / "protocol.json").read_text())
    timing = json.loads((DOC / "stage_timing.json").read_text())
    elapsed = time.monotonic() - timing["started_monotonic"]
    lines = [
        "# Gemma 4 31B final benchmark attempt — incomplete",
        "",
        "| Benchmark | Completed / population | Measured | Published | Delta | Coverage |",
        "|---|---:|---:|---:|---:|---|",
        "| GPQA Diamond, upstream zero-shot CoT | 0 / 198 (planned frozen 128) | — | [84.3%](https://ai.google.dev/gemma/docs/core/model_card_4) | — | Gated dataset unavailable; accounting readiness failed before accuracy |",
        "| SWE-bench Verified | 0 / 500 | — | Not established | — | Tool-use harness required in Stage 12 |",
        "| Terminal-Bench 2.1 | 0 / 89 | — | Not established | — | Terminal harness required in Stage 12; 2.0 figures not substituted |",
        "",
        "| Decode profile | Concurrent requests | Server slots | Complete requests | TTFT mean ms | TPOT mean ms | Decode tok/s/user (1000/TPOT) | Output tok/s aggregate | Prefill FLOP % | Decode DRAM % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for b in ("1", "8", "16", "32"):
        r = profiles.get(b)
        if r:
            lines.append(
                f"| B{b}, 4096/128 | {b} | {r['server_max_num_seqs']} | {r['completed']} / {r['requests']} | {r['mean_ttft_ms']:.3f} | {r['mean_tpot_ms']:.3f} | {1000/r['mean_tpot_ms']:.3f} | {r['output_throughput']:.3f} | unavailable | unavailable |"
            )
        else:
            lines.append(
                f"| B{b}, 4096/128 | {b} target | {b} target | unmeasured | — | — | — | — | unavailable | unavailable |"
            )
    lines += [
        "",
        "**Status: incomplete.** These are latency-only measurements retained after the required runner failed its roofline readiness check. No accuracy verdict or threshold is applied. No host time is represented as device time.",
        "",
        "Target workload: 32 concurrent multi-turn agents, 32768 live tokens per agent, typical context 4096 (Stage 10 estimate pending Stage 12). These profiles measure distinct single-turn 4096-token prompts and 128 generated tokens. They do not measure long-history reuse, tools, agent waiting, or growing-context working sets.",
        "",
        f"Elapsed from the original benchmark_stage invocation through this report: {elapsed:.2f}s of 7200s. All profile launches, reloads, warmups, client verification and reporting count. Initial supplied server: none; Stage 10 had stopped it. No server startup is excluded from the timed interval. Setup before invocation includes client installs, source inspection, manifest download attempt and host checks; see RUN_NOTES.md.",
        "",
        "## Measurement details",
        "",
        "Upstream vLLM 0.26.0 empty-target benchmark client; serial use of all 4 Blackhole ASICs (P300x2, TP4 mesh1x4). Same 60-layer autoport, head_bfp4 precision policy and 262144-token context in each profile. Prefix caching enabled. Request concurrency equals server slots. Greedy temperature 0, ignore_eos=true; requests must each report4096 actual input and 128 output tokens. Seeds differ between warmups and measurements; random prompts are distinct. The client initial probe is excluded by its upstream result accounting. No live device profiler, extra synchronization, or plugin runtime patch.",
        "",
        "Each profile retains the actual process launch argv, API model capacity, imported module path, layer list, precision and source hashes in configuration-bN.json. Source commit and startup duration are in perf-bN-server.json. Model/tokenizer revision:842da3794eaa0b77d5f08bae87a17459d91ff475. Server commands preserve operator top_k 32, checkpoint temperature 1/top_p 0.95, async scheduling and chunked prefill. The server wheel identifies as vLLM 0.31.0+empty; its source commit is unavailable in installed package metadata and remains null.",
        "",
        "| Batch | Metric | Mean ms | Median ms | p95 ms | p99 ms |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for b in ("1", "8", "16", "32"):
        if b not in profiles:
            continue
        r = profiles[b]
        if b != next(x for x in ("1", "8", "16", "32") if x in profiles):
            lines += ["| Batch | Metric | Mean ms | Median ms | p95 ms | p99 ms |", "|---:|---|---:|---:|---:|---:|"]
        for m in ("ttft", "tpot", "itl", "e2el"):
            lines.append(
                f"| {b} | {m} | {r['mean_'+m+'_ms']:.3f} | {r['median_'+m+'_ms']:.3f} | {r['p95_'+m+'_ms']:.3f} | {r['p99_'+m+'_ms']:.3f} |"
            )
        lines += [
            "",
            f"B{b}: measured wall{r['duration']:.3f}s; request throughput{r['request_throughput']:.3f}/s; input tokens{r['total_input_tokens']}; output tokens{r['total_output_tokens']}. [Raw result](partial-performance/perf-b{b}.json).",
            "",
        ]
    lines += [
        "## Missing phase accounting",
        "",
        "Whole-server sampling-mode logs (including warmups) count host-compatible prefill steps B32=128, B16=64, B8=32 and B1=9; all observed decode steps use device sampling. These are lifetime counts, not an isolated measured-window attribution. See [sampling_modes.json](sampling_modes.json). No full-phase percentages can be reported. AutoDebug and an independent AutoFix hypothesis check found no lossless stock-plugin export spanning scheduler work, asynchronous completion, host sampling and final token handling with step/request identity. Adapter enqueue timers and summed request TTFT/TPOT would be invalid substitutes. The fail-closed collector records this failure before accuracy. Runtime/interface changes remain unauthorized under AGENTS.md; the unapplied PC-04 proposal is not a validated collector. See [diagnosis](../setup/AUTODEBUG.md), [hypothesis review](../setup/completion_export_review.md), and [proposal](../setup/PC-04-completion-observer.md).",
        "",
        "## Accuracy protocol and coverage",
        "",
        "The packaged ci-v1-reasoning manifest is retained verbatim with 128 selected GPQA documents from 198, per-document hashes, population hash and seed 0 choice shuffle. Zero-shot: 0 few-shot examples. Dataset content could not be downloaded or verified: HF authentication was unavailable. No questions were sent, scores computed, responses collected or stop/truncation counts measured. Those counts remain unknown rather than inferred as zero. No questions were removed or replaced.",
        "",
        f"Frozen manifest digest: `{protocol['coverage'][0]['manifest_sha256']}`. Population hash and selected document hashes remain in [manifest.json](manifest.json).",
        "",
        "Intended accuracy protocol uses structured chat messages with native pinned HF template exactly once, enable_thinking=true, temperature 1, top_p 0.95, top_k 20 and 126976 output cap with EOS retained and no textual stops. The pinned release source uses r1_gpqa_diamond and a 14400s HTTP timeout; the packaged upstream runner uses gpqa_diamond_cot_zeroshot and 3600s timeout. These differences are recorded, not described as reference parity. The published 84.3% is for this model, but no measured delta exists. See [protocol](../protocol.json).",
        "",
        "## Handoff",
        "",
        "The stage checker exits 2 because the original runner is incomplete; its generic budget message is not a timeout. All four latency/identity validations and the context check pass; see [checks.json](checks.json). Owned servers were stopped at idle and the final device-owner scan is empty. All latency results are supplemental partial evidence; the original generated runner report and failed summary are preserved. Completion requires a validated full-phase observer/collector and valid accuracy coverage or an explicit recorded coverage exclusion. GPQA needs authorized dataset access. Stop/truncation statistics and upstream scores cannot be produced without responses.",
        "",
        "Release matrix derives only from Qwen/Qwen3-32B P300X2 in pinned tt-inference-server411057222c2dc5a4268e693b0bd57bc0e2571f38: 128/128/C1, 8 requests. Its time estimate is separate from measured 4096/128 profiles; Stage 12 must measure that row. No old Gemma implementation settings were used as evidence.",
        "",
        "Context contract was preserved. Evidence/context checker outputs, ownership cleanup, commit records and setup limitations are in RUN_NOTES.md and setup/. No push, CI dispatch, issue comment or message was made.",
    ]
    (RUN / "REPORT.md").write_text("\n".join(lines) + "\n")
    result = {
        "status": "incomplete",
        "performance": profiles,
        "accuracy": [],
        "elapsed_seconds_through_report": elapsed,
        "budget_seconds": 7200,
        "within_budget": elapsed < 7200,
        "missing_reasons": protocol["coverage"],
        "roofline_missing_reason": "No authorized lossless stock completion export",
        "context_contract_sha256": protocol["context_contract_sha256"],
    }
    (RUN / "attempt_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if "1" in profiles:
        r = profiles["1"]
        root = MODEL.parents[2]
        packet = (
            root
            / "bringup/artifacts/multigoal-runs/v3-20261105T172350Z/telemetry/packets/96fb930d-2d63-4b49-a4d5-6063d2f966e8.json"
        )
        d = json.loads(packet.with_name(packet.name.replace(".json", ".template.json")).read_text())
        d["selected_mesh"] = {"shape": [1, 4], "devices": [0, 1, 2, 3]}
        d["performance"] = {
            "source": "vllm",
            "ttft_ms": r["mean_ttft_ms"],
            "decode_tps": 1000 / r["mean_tpot_ms"],
            "prefill_device_us": None,
            "decode_device_us": None,
            "prefill_flops_pct": None,
            "decode_dram_pct": None,
            "evidence": "models/autoports/google_gemma_4_31b_it/doc/benchmark/run/partial-performance/perf-b1.json",
        }
        d["missing_reasons"] = {
            "device_time_and_rooflines": "Live serving profiling prohibited; no matching device measurements. Full-phase host accounting separately unavailable: stock plugin lacks lossless completion export.",
            "accuracy": "No scored samples: GPQA dataset authentication unavailable; tool-use suites require Stage 12.",
        }
        d["notes"] = [
            "Stage11 incomplete; valid latency-only4096/128/B1/C1 evidence.",
            "Full 60 layers, head_bfp4, context 262144, prefix caching enabled.",
        ]
        packet.write_text(json.dumps(d, indent=2) + "\n")


if __name__ == "__main__":
    main()
