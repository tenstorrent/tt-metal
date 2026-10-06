# SPDX-License-Identifier: Apache-2.0
"""Build the stage report only from the final checked evidence artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/functional_decoder"


def read(name):
    return json.loads((DOC / name).read_text())


checks = read("evidence_check.json")
perf = read("performance_summary.json")
lines = [
    "# Functional decoder validation",
    "",
    "Target: Aleph-Alpha/Kolibri-1-BF16, revision `7a8f290e7858825c3cf5e4c447ba68345de9f1d3`.",
    "One p300c Blackhole chip, unit mesh `[0]`, on owner-approved base `6811c43a`.",
    "Acceptance is unchanged: PCC >=0.995. This report covers the functional decoder only.",
    "",
    "Numerical/device gates passed. Independent stage review and local checkpoint status are recorded in README/work_log.",
    "",
    "## Correctness",
    "",
    "| Layer kind | Real boundary minimum PCC | Synthetic minimum PCC | 1M sampled prefill PCC | 1M traced decode PCC | Minimum long-context PCC |",
    "|---|---:|---:|---:|---:|---:|",
]
for layer, kind in ((0, "Sliding/RoPE"), (4, "Full/RNoPE")):
    real, synthetic, context = (read(f"{stem}_{layer}.json") for stem in ("coverage", "synthetic", "context"))
    pre = next(
        r["pcc"] for r in context["rows"] if r["case"] == "full_context_prefill" and r["logical_length"] == 1048576
    )
    dec = next(
        r["pcc"] for r in context["rows"] if r["case"] == "full_context_traced_decode" and r["context"] == 1048576
    )
    lines.append(
        f"| {kind} | {min(r['pcc'] for r in real['rows']):.9f} | {min(r['pcc'] for r in synthetic['rows']):.9f} | {pre:.9f} | {dec:.9f} | {min(r['pcc'] for r in context['rows']):.9f} |"
    )
lines += [
    "",
    "Artifacts: `coverage_0/4.json`, `synthetic_0/4.json`, `context_0/4.json`, and their logs.",
    "Boundary coverage includes logical lengths 1,31,32,33,127,128,129,511,512,513,777,8193,17;",
    "continuation 129+173 followed by decode 302; unchunked 129 control; changed-input retained-trace",
    "requests including decode positions 255/256/257. Each fixture records one capture and 33 replays,",
    "bitwise identical repeated outputs, changed page maps/positions/tokens, and a clean runtime audit.",
    "",
    "| Layer / batch | Prefill PCC | Traced decode PCC | Minimum changed-input lane PCC |",
    "|---|---:|---:|---:|",
]
for layer, batch in ((0, 32), (4, 32), (0, 31), (4, 13), (0, 2), (4, 2)):
    d = read(f"batch{batch}_{layer}_final.json")
    lines.append(
        f"| {layer} / {batch} | {d['prefill_pcc']:.9f} | {d['decode_pcc']:.9f} | {min(d['changed_batch_per_row_pcc']):.9f} |"
    )
lines += [
    "",
    "## Warmed performance",
    "",
    "B1; prefill 128 tokens; traced one-token decode at position 128 with capacity 1024.",
    "Allocation tracking, watcher and runtime audit instrumentation are disabled for timing.",
    "Separate tracked, audited runs validate the same production paths. Both baseline and repaired",
    "timing runs use the same instrumentation. Three repetitions follow warmup, with synchronization at",
    "window boundaries. `Device Time` in the filtered tt-perf-report CSV is microseconds:",
    "sum / 3 / 1000 gives device kernel milliseconds per pass. Gaps and host elapsed time are separate.",
    "",
    "| Layer | Mode | Device kernels (ms) | Device gaps (ms) | Host elapsed (ms) | Numerical repair cost (ms) |",
    "|---|---|---:|---:|---:|---:|",
]
for d in perf["rows"]:
    lines.append(
        f"| {d['layer']} | {d['mode']} | {d['device_kernel_ms']:.4f} | {d['device_gap_ms']:.4f} | {d['host_elapsed_ms']:.4f} | {d['numerical_repair_cost_ms']:+.4f} |"
    )
lines += [
    "",
    "Exact commands, input CSV SHA256, repetition count and baseline comparison are in",
    "`performance_summary.json`. Human-readable tables and filtered CSVs are",
    "`tracy/layer_<0|4>/<prefill|decode>_perf_report.txt/.csv`; the copied Tracy source is",
    "`<prefill|decode>_ops.csv`. Profile commands/PCC are in `profile_<layer>.log/.json`.",
    "tt-perf-report lacks category labels for TopK/Scatter and omits modeled DRAM/FLOP",
    "utilization for runtime sparse nnz. These metadata warnings do not remove kernel times.",
    "No fixed six-expert union is assumed for multi-token groups. Optional Tracy web-viewer copy",
    "warnings concern a missing viewer-only host file; all measured device rows are present.",
    "Pandas mixed-column dtype warnings do not remove the numeric timestamp cells. UMD warns",
    "about subset MMIO and unknown motherboard tray ID; successful unit-mesh initialization",
    "and the numerical/watcher suites validate this device path. No reported latency includes",
    "the earlier failed profiler capture.",
    "Cost compares the same inputs/shapes to `before_numerical_fix/untracked_perf/tracy`; it measures the whole",
    "correctness repair, including altered expert choices and attention partition, not isolated op cost.",
    "",
    "## Capability contract",
    "",
    "| Claim | Evidence | Remaining scope or risk |",
    "|---|---|---|",
    "| Native context 262144 | Both kinds: full prefill and traced decode at 262143; tail 262127 | Long-reference comparison samples eight query rows, using all prefix K/V |",
    "| Model-card extension 1048576 | Both kinds execute every prefill token; final traced decode at 1048575; tail 1048559 | Requires max_position_embeddings=1048576 as documented by model card |",
    "| No capability reduction | doc/context_contract.json and successful full allocations/execution | B32 at 1M is not claimed; B32 correctness uses shorter contexts |",
    "| Paged ownership and positions | Shuffled physical pages/request rows, heterogeneous device positions, continuation | Caller must assign disjoint owned pages and refresh matching RoPE/positions |",
    "| Both layer kinds | Real layers 0 and 4, repeating target layer pattern; full target shapes | No full stack or generation in this stage |",
    "| Fully device-side decode trace | Allocation tracking, 33 repeated replays, exact per-lane checks | Supported programs and persistent buffers must be prepared before capture |",
    "",
    "Each layer has 3,097,908,480 checkpoint bytes. BF16 KV costs 2048 bytes/token/request,",
    "or 2,147,483,648 bytes at 1M. `capacity_arithmetic.json` excludes scratch/padding and records",
    "that actual successful allocation/execution is the capacity proof. No reduced advertised limit is used.",
    "",
    "## Numerical repair and audits",
    "",
    "Small upstream differences changed a sixth expert on near-tied router scores. Same-input",
    "CPU/TT MoE controls agree above .99998. RMSNorm now preserves BF16 rounding before gamma;",
    "decode uses accurate exponentiation, K256 and up to 16 cores/head/batch to reduce repeated",
    "BF16 online-softmax rounding. AUTODEBUG.md, AUTOFIX.md and numerical_fix_review.md record",
    "the source diagnosis, rejected candidates, measured controls and final validation.",
    "No shared-code change, oracle change, seed change, threshold relaxation or runtime host fallback was used.",
    "",
    "`watcher_0/4.json/.log` and `watcher/layer_<layer>/generated/watcher/` are separate watcher 10 runs.",
    "Runtime audits reject Torch and TTNN host-transfer calls within forward; setup and PCC readback",
    "are explicit boundaries. Allocation tracking includes program-cache allocations and uses no exemptions.",
    "`evidence_check.json/.log` verifies final decoder source hashes and gates. `binary_provenance.json`",
    "and `environment.json` identify the source-built runtime. Historical failures and pre-fix passes",
    "are retained in the baseline folders and are not substituted for final evidence.",
    "",
    "## Reference limits",
    "",
    "Stock Transformers has no Kolibri decoder. The layer-only PyTorch reference is transcribed",
    "from pinned provider architecture and checkpoint semantics, with HF-style BF16 normalization",
    "boundaries. Extracted provider routing/residual bodies pass exact checks in",
    "`provider_reference_check.json`; provider GPU attention/RMSNorm/FusedMoE kernels were not run.",
    "Long-context reference work projects every prefix K/V and compares eight complete decoder",
    "output rows at each checkpoint. Short and batched suites compare all output rows.",
    "",
    "## Reproduction",
    "",
    "```bash",
    "bash models/autoports/aleph_alpha_kolibri_1_bf16/tests/run_remaining.sh",
    "TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.run_context --layer 4",
    "TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.run_context --layer 0",
    "bash models/autoports/aleph_alpha_kolibri_1_bf16/tests/run_final_checks.sh",
    "python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.check_evidence",
    "python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.write_report",
    "```",
    "",
    "Run device commands serially in the checkout environment; watcher and profiler remain separate.",
    "See README for the public API, input shapes, chunking, page ownership and trace lifetime contract.",
]
(DOC / "functional_decoder.md").write_text("\n".join(lines) + "\n")
