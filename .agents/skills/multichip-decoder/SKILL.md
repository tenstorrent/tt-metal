---
name: multichip-decoder
description: Parallelize a functional or optimized TTNN transformer decoder layer across the current multi-chip hardware in tt-metal. Use when adding tensor parallel or 2D mesh execution to a single-chip decoder, choosing 1D TP up to 8 chips or Galaxy 4x8 strategies, preserving functional/optimized tests, comparing multi-chip output to the single-chip TTNN baseline, validating RMSNorm and paged KV-cache correctness, tracing the target decode path, profiling with tt-perf-report, and producing final golden multichip_decoder artifacts.
---

# Multichip Decoder

## Goal

Take a working single-chip TTNN decoder layer from `functional-decoder` or `optimize-decoder` and make it run correctly and efficiently on the current multi-chip hardware. The target is the hardware actually available for the run, not a sweep over smaller tensor-parallel factors.

The decoder output must have the same layout contract as the decoder input so decoder layers can be chained without extra conversion ops between layers. The implementation may use hidden-sharded, replicated, or hybrid activation streams if evidence shows that choice is fastest and correct.

## Required Reference Reads

Before implementation, read:

- `references/parallelization-knowledge.md` for GPT-OSS, `models/common`, Galaxy, MoE, RMSNorm, and CCL reference code paths.
- `references/artifact-formats.md` for exact final artifact names, JSON schemas, status values, and report headings.
- The target model's `doc/optimized_decoder/` artifacts if present and passing; otherwise `doc/functional_decoder/` artifacts.

Do not skip these reads. The skill body defines the workflow; the references define the required details that keep many model runs comparable.

## Baseline Selection

Prefer the strongest passing single-chip baseline:

1. If `models/demos/<model>/doc/optimized_decoder/manifest.json` exists with `"status": "pass"`, use optimized decoder as the baseline.
2. Otherwise use `models/demos/<model>/doc/functional_decoder/manifest.json` with `"status": "pass"`.
3. If neither exists or neither passes, stop and repair or rerun the earlier phase unless the user explicitly asked only for diagnostic planning.

Preserve the baseline's public correctness contract. Multi-chip tests may be separate tests or parameterized variants, but they must not weaken the earlier prefill, decode, paged KV-cache, determinism, stress, trace, watcher, or fallback assertions.

## Definition Of Done

- Implement multi-chip execution for the target hardware's supported mesh size. On a 1x8 mesh, target TP=8; do not spend final proof time also proving TP=2 or TP=4 unless the user asks.
- Use the current hardware's natural parallelism. For 1D hardware up to 8 chips, default to 1D tensor parallelism. On Galaxy 4x8, make an explicit plan for 2D parallelism and justify whether dense 2D TP, expert replication, or another strategy is fastest.
- Compare multi-chip TTNN outputs to the single-chip TTNN baseline, not directly to HF, with `PCC >= 0.995` for prefill and decode. Expect much higher PCC when the only intended delta is parallelization; investigate large deltas even if they pass the minimum threshold.
- Keep paged KV-cache behavior correct and preserve the baseline's supported sequence limits unless device DRAM or L1 capacity forces a documented reduction.
- Preserve any optimized-decoder guarantees if the baseline is optimized, including precision, trace, watcher, stress, and fallback guarantees.
- Run decode correctness from the warmed trace replay path at the target mesh size.
- Run watcher-enabled correctness at the target mesh size and require watcher-clean execution, or include exact false-positive evidence.
- Include repeated-run determinism checks with identical inputs and identical outputs.
- Include an optional stress mode, and run it for final success if the selected baseline requires stress.
- Report warmed prefill and decode latency for single-chip baseline and target multi-chip implementation, plus speedup and efficiency.
- Run Tracy and `tt-perf-report` for final target prefill and decode windows.
- Produce only the final golden proof artifacts under `models/demos/<model>/doc/multichip_decoder/`.

## Parallelization Workflow

1. Parse the chosen baseline artifacts and identify layer kinds, tested shapes, sequence limits, precision, latency, trace behavior, and test commands.
2. Inspect the existing single-chip implementation and closest multi-chip references listed in `parallelization-knowledge.md`.
3. Determine the target mesh from the opened hardware. Use the largest supported mesh shape for the final implementation.
4. Write a short parallelization plan covering activation layout, weight sharding, KV-cache layout, RMSNorm strategy, collectives, and expected communication boundaries.
5. Implement the smallest correct multi-chip path first, keeping setup-time torch and weight conversion out of runtime prefill/decode.
6. Compare multi-chip outputs to the single-chip TTNN baseline for component-local debugging, then for full decoder prefill and decode.
7. Tune layout and collectives. Avoid adding conversion ops between decoder layers; if a conversion is required inside a sublayer, prove why it is faster or necessary.
8. Run baseline regression checks, target-mesh PCC, KV-cache checks, determinism, stress, watcher, traced decode, and profiling.
9. Rerun the final golden proof commands and copy only the final evidence into `doc/multichip_decoder/`.

## Strategy Guidance

Treat these as defaults and examples, not hard contracts. Choose the fastest correct implementation and record the evidence.

For 1D dense decoder layers, the usual starting point matches the rest of the TP models:

- shard WQKV, W1, and W3 by output/features;
- shard WO and W2 by input/reduction dimension;
- run attention heads and KV cache locally by local Q/K/V heads;
- reduce-scatter or all-reduce after WO and W2 as needed to restore the residual layout;
- keep residual input and output in the same layout so stacked decoders do not need extra ops.

For RMSNorm, correctness is mandatory but the implementation strategy is open. If hidden activations are hidden-sharded, use distributed RMSNorm stats or another mathematically equivalent strategy. Replicating activations and using local RMSNorm is acceptable if it is faster and the layout contract plus performance evidence justify it.

For MoE on TP up to 8, default to tensor-parallel execution of each active expert selected by the gate. Do not densify all experts as the final path unless the model or hardware leaves no better option and the report explains why.

For Galaxy 4x8, make a model-specific plan. For MoE models, consider the GPT-OSS strategy of replicating expert groups so each active expert can run in parallel, but only accept it if DRAM capacity fits all layers plus the full KV cache, not just the current decoder. For dense models, evaluate 2D parallelism rather than forcing a 1D 8-chip plan; record the chosen axes and alternatives rejected.

## Tests

Write or extend pytest coverage near the model implementation.

Required behavior:

- Open the current target mesh and run at the hardware-supported target size.
- Produce a single-chip TTNN baseline output with identical synthetic inputs, weights, page tables, and positions.
- Compare target multi-chip TTNN output to that single-chip TTNN output for full decoder prefill and decode with `PCC >= 0.995`.
- Preserve the baseline's paged KV-cache tests and prove multi-chip page-table, current-position, user-slot, and local KV-head behavior.
- Preserve sequence limits or document reductions with memory evidence.
- Run repeated deterministic inputs and assert identical multi-chip outputs.
- Keep optional component PCCs for debugging, but final success requires full decoder PCC.

## Trace, Profiling, Watcher, And Fallback Audit

Trace only the target multi-chip decode path required for the current hardware. Use warmup, capture, and replay in the same style as the baseline artifacts.

Profile final warmed prefill and warmed decode separately with Tracy signposts around only the measured windows. Run `tt-perf-report` on both windows. Include communication-bound, DRAM-bound, compute-bound, and data-movement findings in the report.

Run watcher and profiling as separate passes. Watcher-clean means the pytest command exits successfully, watcher attaches and detaches cleanly, logs are present, and no watcher log or stderr contains hardware faults, asserts, sanitizer errors, NOC errors, CB/L1/stack overflows, link errors, or watcher-server errors. Record exact false-positive evidence if any message is waived.

Audit the runtime path for host fallback. No runtime torch, `ttnn.from_torch`, `ttnn.to_torch`, or host-device fallback may occur inside prefill/decode except explicit input/output test boundaries inherited from the baseline.

## Golden Artifact Layout

Final golden multi-chip evidence lives directly under:

```text
models/demos/<model>/doc/multichip_decoder/
```

Do not save every debug attempt in this doc tree. Intermediate failing or exploratory runs belong under scratch locations such as `generated/multichip_decoder/debug/`, `/tmp`, or uncommitted local artifact directories.

Required final files:

```text
manifest.json
multichip_decoder.md
commands.sh
baseline_summary.json
mesh_contract.json
parallelization_plan.json
memory_capacity_plan.json
baseline_regression_results.json
fallback_audit.md
data_movement_audit.md
tt_perf_advice.json
performance_results.json
sequence_limits.json
results/pcc_results.json
results/kv_cache_results.json
results/determinism_results.json
results/stress_results.json
watcher/watcher_summary.json
pytest/<layer_kind_id>_prefill.log
pytest/<layer_kind_id>_decode.log
pytest/<layer_kind_id>_stress.log
pytest/<layer_kind_id>_watcher.log
watcher/<layer_kind_id>/generated/watcher/watcher.log
watcher/<layer_kind_id>/generated/watcher/kernel_names.txt
watcher/<layer_kind_id>/generated/watcher/kernel_elf_paths.txt
tracy/<layer_kind_id>/prefill_ops.csv
tracy/<layer_kind_id>/prefill_perf_report.csv
tracy/<layer_kind_id>/prefill_perf_report.txt
tracy/<layer_kind_id>/decode_ops.csv
tracy/<layer_kind_id>/decode_perf_report.csv
tracy/<layer_kind_id>/decode_perf_report.txt
```

If there are multiple layer kinds, repeat layer-kind-specific pytest, watcher, and Tracy files for each `<layer_kind_id>`.

Do not store binary tensors, full model weights, TTNN program-cache directories, or giant profiler captures in the doc artifact tree. Use `references/artifact-formats.md` for exact JSON keys and report headings.

## Knowledge Base

The required references are part of this skill's contract. Re-read the relevant sections before choosing a mesh strategy, implementing RMSNorm, accepting a Galaxy MoE plan, or generating final artifacts.
