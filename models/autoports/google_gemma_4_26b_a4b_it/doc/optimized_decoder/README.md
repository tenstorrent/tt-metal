# Gemma 4 optimized decoder

The selected v8 runtime passes correctness, stress, Watcher and context gates.
Full-prefill QKV consumes normalization output directly from L1 and also produces
L1 output, with legal M2 blocking.
Native v8 profiles and whole-layer accounting are complete.
Prefill-router controls pass but show no resolved whole-layer gain, so their
HiFi4/DRAM defaults are retained. [Independent stage review](STAGE_REVIEW.md)
returned **clean-pass**, with no required work remaining.

Stage 03 for `google/gemma-4-26B-A4B-it`, pinned revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. One Blackhole ASIC (physical device3,
mesh1×1,110 available workers) on the P300c host. The delivered path is
[`OptimizedDecoder`](../../tt/optimized_decoder.py), source SHA256 `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`.
No multichip, full-model, or vLLM work is included. The work log records the local checkpoint commits; nothing is pushed.

## Before/after performance

The headline workload is **4096 input tokens,128 teacher-forced decode positions,
batch1,one request**, separately for layer0 sliding attention and layer5 full
attention. These are complete single-layer device windows, including every
operation and internal device gap. Prefill is warmed; decode is traced and the
mean covers all128 refreshed positions. No layer kinds are averaged together.

| Layer kind | Fused prefill µs | Optimized prefill µs | Speedup | Fused decode µs | Optimized decode µs | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding_attention | 2813296.579 | 221367.855 | 12.71× | 5146.563 | 881.411 | 5.84× |
| full_attention | 2825257.235 | 186562.836 | 15.14× | 5586.547 | 899.336 | 6.21× |

Exact values, native tables/CSVs and commands are in
`tracy/actual_fused_layer{0,5}` and `tracy/actual_optimized_v8_layer{0,5}`.
Each directory has `whole_layer.json`, `prefill_perf_report.txt/csv`,
`decode_perf_report.txt/csv`, `summary_commands.json`, and
`timing_reconciliation.json`. The reconciliation compares device windows to
host timing from the **same profiled run**, then separately reports profiler
overhead against the unprofiled defaults. Host measurements are not device telemetry.

| Layer kind | Unprofiled warmed prefill host µs | Unprofiled fixed-position traced decode host µs |
| --- | ---: | ---: |
| sliding_attention | 221260.293 | 796.368 |
| full_attention | 186547.611 | 832.701 |

These are final-default medians from `validated_v8_headline_layer{0,5}.json`.
They reproduce the fastest correct selected decode family; differences within
timing spread are not ranked as new wins. Full prefill HiFi2 beats its legal
HiFi4 control in all eight alternating pairs, while sliding retains HiFi4
after a real maximum-context failure. See [matched comparisons](minimal_pairs_v6_summary.md). Candidate history is in
[configuration selection](current_selection_results.md),
[compact experts/RoPE](late_topology_results.json),
[prefill projection](prefill_output_results.md), and
[norm controls](native_headnorm_control.md). Faster candidates that fail the
real-input .995 gate are rejected. Timing differences within measured spread
are not claimed as improvements.

## Correctness and capability

The unchanged acceptance threshold is **PCC≥.995**. All128 headline decode
positions pass independently, with exact repeated trace output equality.

| Layer kind | Fused prefill PCC | Optimized prefill PCC | Fused minimum decode PCC | Optimized minimum decode PCC |
| --- | ---: | ---: | ---: | ---: |
| sliding_attention | 0.9999037564 | 0.9991560943 | 0.9999465508 | 0.9952574859 |
| full_attention | 0.9999225513 | 0.9991223369 | 0.9999017276 | 0.9951056876 |

The intentional BFP8/BFP4 weight/cache and selective-fidelity policy reduces PCC
relative to fused BF16/precise arithmetic while retaining the acceptance bar.
[Precision evidence](precision_evidence.md), the actual-input candidate journals,
and the AutoFix reports retain isolated controls. They do not claim unmeasured
router rank changes or use Gaussian PCC to reject real-input wins.

[Final validation](validated_v8_validation_summary.json) separates current
v8 affected-path tests from inherited unchanged branches. Full-attention B32,
prefix, BF16-cache compatibility, allocation-tracked reuse, maximum and
near-maximum contexts are rerun on the selected precision and L1 producer layout. Both kinds have current
4096/128 headlines, Watcher and512-step HF/direct-fused stress; four pytest cases
exercise33/65 tokens. A current tight-capacity1025 case verifies K128 bounds,
and warmed full-prefill65/1023/1024/1025 timings cover short and chunk-tail work.
The [source delta](source_delta_v8.md) maps retained sliding maximum/batch/prefix
and the broader v6 bounds/lifecycle checks to unchanged code. Historical results
retain their original hashes. Together the evidence covers:

- B32 with distinct actual inputs, disjoint randomized pages and positions32–63;
- non-aligned33/65-token tests, prefix continuation, nine-request trace/cache reuse
  including1023/1024/1025/2049/2047, and BF16-cache compatibility;
- eight tight-cache cases at1025–2049 tokens, including prefill-only1152,
  native-call page-table bounds assertions and both K128/K256 programs;
- maximum262144 and non-aligned262143 contexts: complete TT prefill, independent
  PCC checks on291 selected HF query rows including boundaries/final tail,
  plus traced decode at the final positions;
- real1025-token prefill plus512 decode steps, compared both to HF and directly
  to fused output, with no excluded/shared failures;
- runtime guards against torch/from_torch/to_torch during measured execution and
  against `FunctionalDecoder._forward` fallback;
- separate Watcher runs with interval10, all features enabled, and clean logs.

The [context contract](../context_contract.json) retains262144 tokens and arbitrary
valid logical lengths. Padding/chunking stays internal. BFP8 cache payload at that
context is1,140,850,688 bytes sliding and570,425,344 bytes full.
[Memory accounting](final_memory_accounting.md) states weight aliases, packed
storage, and allocator/transient exclusions. B32 short-context correctness does
not claim32 simultaneous maximum contexts. TILE decode RoPE inputs remain supported;
`validated_v4_tile_rope_commands.json` checks both kinds after selecting setup-time
ROW_MAJOR tables to avoid whole-table conversion.

Trace reuse requires initialization of every intended prefill/decode program
signature before capture. The delivered reuse test warms its nine signatures,
then forbids program-cache misses while the trace is live; full allocation
tracking reports no retained post-capture allocations. A caller introducing an
unseen signature must warm it before capture or release/rebuild the trace.
This is a trace lifecycle requirement, not a restriction on logical sequence
length. Final sliding-prefill chunks no longer retain unused K/V-tail clones.
For tightly sized full-attention page tables, Q64/K128 replaces Q64/K256 only
when the wider rounded read would exceed logical cache capacity. See
[the review repairs](AUTOFIX_review_contracts.md) and their native-call guards.

Maximum-context accuracy is a291-row subset, not a claim of all262144 HF rows.
Recorded fixtures use real checkpoint weights and text-derived HF layer inputs,
identical BF16 transport for the oracle and TT, and pinned provenance in
[the fixture manifest](actual_text_fixture_manifest.json). There is no complete
model generation or qualitative-output claim at this decoder stage.

## Selected policy and topology decisions

[Resolved policy](final_policy.json), [selection](select_defaults.md), and
[the complete optimization checklist](optimize_checklist.md) give exact settings.
The initial topology audit and chronological trials are in [work_log.md](work_log.md);
[the current native audit](final_perf_advice.md) binds decisions to final rows.

| Current operation family | Replacement/action | Evidence |
| --- | --- | --- |
| Same-input Q/K/V matmuls | Packed direct FP32 input/output, BFP8 weights,110-core K11; faster than tuned legal separate/DRAM families | `direct_geometry_results.md` |
| Expert gate/up/down and dense128-slot movement | Indexed8-slot sparse execution, packed gate K44/down K22 on44cores, accurate fused GELU, compact matmul reduction | `compact_expert_commands.json`, `compact_geometry_commands.json`, `operator_audit_commands.json` |
| Expert precision | Sliding gate BFP8/down BFP4; full gate/down BFP4; LoFi; material11/22/44/88-core and K-block controls | `precision_evidence.md`, `expert_gate_geometry_results.md`, native audit |
| Shared gate/up/down | Packed DRAM-sharded decode; sliding BFP8 reader1, full BFP4 reader2; K11 | `direct_geometry_results.md` |
| Attention gather/reduce and casts | Native paged SDPA, BFP8 cache, BF16 attention output through projection input, FP32 residual output | `native_sdpa_boundary_results.json` |
| Router and hidden norms | Centered generalized top8 gate; direct FP32/BF16 router GEMM; sharded hidden norms for both kinds | `direct_router_results.md`, `full_all_norm_commands.json` |
| RoPE table movement | Setup-time row-major decode tables; small selected rows returned tiled by embedding | `rope_layout_commands.json`, final native audit |
| Large prefill projections/attention | Minimal QKV11×8/M4/K8/HiFi4 sliding, M2/K16/HiFi2 full with direct L1 normalization output; minimal outputK8 full only; sliding output tuned2D/K16/LoFi; full paged Q64/K256/HiFi2 with tight-capacityK128 | `minimal_selection.md`, `prefill_output_results.md`, `long_prefill_config_results.md`, `tight_cache_v6_commands.json` |
| Prefill active experts |32-token active unions, role-specific sparse configs and LoFi; sliding gate BFP8 repair shares decode weights | `AUTOFIX_sliding_prefill.md`, final native audit |

Measured rejected options include tuned separate projections, larger/smaller expert
geometries, QKV/output DRAM alternatives, weighted expert reduction, router larger
subblocks, sliding native head norms, attention BFP4, and lower precision at the
sensitive sliding expert gate. Legal adapted candidates were measured; initial
L1/API errors alone do not reject an entire family. The advice ledger records
before/after evidence and TTNN/report limitations. Completed minimal-QKV fidelity/grid and shared/output L1 alternatives have measured
results in [the advice controls](final_perf_advice_controls.md) and
[paired timing](minimal_pairs_v6_summary.md). HiFi2 has a paired prefill gain
and passes full-attention long-context acceptance; sliding retains HiFi4 because
HiFi2 fails real sampled row32 at0.994993. The11x10 QKV, redistributed110-core
shared matmuls and shared/output L1-input alternatives give no measured gain.
The full-QKV L1-placement recommendation produced a measured M2 gain. Direct
normalization output in L1 then beat the extra-copy candidate in22/32 pairs.
The selected minimal QKV and tied-K/V slice also inherit L1 output; the following
concat returns to DRAM. Current native rows and capacity tests cover this exact
path. Four phase-matched router controls test HiFi2 and direct L1 production on
each layer kind with32 alternating pairs and actual4096/128 correctness. Neither
apparent gain exceeds the observed timing spread; retain HiFi4/DRAM. No decoder
optimization is assigned to a later stage.

The repaired anomalies are documented in `AUTOFIX_reuse.md`,
`AUTODEBUG_long_prefill.md`, and `AUTOFIX_sliding_prefill.md`: centered routing
repairs a real reuse window; full paged HiFi2 repairs long attention; sliding
prefill BFP8 gate repairs four sampled-row misses. The strengthened final tests
pass these cases. Historical aggregate-only checks are not relabeled as rowwise passes.

## Whole-layer roofline estimates

| Layer kind | Prefill useful FLOPs | Prefill useful/peak | Decode estimated DRAM bytes | Decode bytes/peak |
| --- | ---: | ---: | ---: | ---: |
| sliding_attention | 882489950208 | 0.600787% | 110493448 | 24.484327% |
| full_attention | 1215408635904 | 0.981798% | 106384012 | 23.103866% |

Denominators are the complete device times in the performance table, never
matmul-only sums. Reference peaks are663.552TFLOP/s LoFi and512GB/s for the
participating Blackhole ASIC (120 theoretical cores at1.35GHz). The runtime mixes
fidelities; the common LoFi reference is explicit, not measured FPU utilization.
Useful FLOPs count8 active experts per token and exclude inactive/padded work.
DRAM estimates use native operand formats including BFP exponent storage,
indexed8-expert output metadata, and chunk-rounded native KV reads. Repeated
per-core reads, NoC traffic and profiler writes are excluded; these are estimates,
not controller counters. Percentages are not clamped.

The installed report tool cannot infer indexed expert counts. Decode reports use
`--active-experts 8` only after raw native `use_indices=true` and compact output
shape prove it. Prefill union counts vary; its reports never substitute8.

## Reproduction and stage records

Run commands from the repository root in the existing TT environment. All
hardware work is serialized; Watcher and profiler are separate. Exact argv arrays,
result paths and return codes live in:

- `validated_v8_contract_commands.json`, `v8_boundary_commands.json`,
  `validated_v8_validation_summary.json`, and retained
  `trace_alloc_v6_commands.json` / `tight_cache_v6_commands.json`;
- inherited `validated_v5_contract_commands.json`,
  `stress_validated_v8_defaults/summary.json` and `validated_v4_tile_rope_commands.json`;
- `prefill_boundary_commands.json`, [current full short/tail timing](prefill_boundary_v8_summary.md),
  `reader_layer_v6/commands.json` and
  `dram_readers_v5/reconstructed_execution_plan.json`;
- `actual_optimized_v8_profile_commands.json` and each profile's `summary_commands.json`;
- `operator_audit_commands.json` for matched one-replay geometry/precision audits.

A direct correctness invocation is:

```bash
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract run_decoder --layer 0 --length 4096 --real \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer0_4096_128.pt \
  --decode --steps 128 --timing --prefill-timing --verify-program-cache \
  --output /tmp/gemma_optimized_layer0.json
```

`create_optimized_activation_fixture.py` and `create_optimized_long_reference.py`
reproduce ignored local tensors from the pinned checkpoint. The pytest module
creates a compact real fixture if headline tensors are absent. Compact evidence
and report tables are retained; tensor payloads and raw large Tracy dumps stay local.
Python repository pre-commit and explicit pinned Black23.10.1 passed
(`precommit_final_staged.log`, `black_final_all.log`); no C++ or CMake build
is required for this Python/tests/docs-only change. The work log records review
and local checkpoint SHAs; nothing is pushed.

## Reconstructing archived evidence

[The archive manifest](evidence_archives.json) records original and gzip SHA256,
sizes and an exact reconstruction command. Some rendered reports exceed the
repository500KiB limit; some historical JSON and executed scripts must retain
original bytes for provenance. Their deterministic `.gz` companions preserve
those bytes without rewriting measurements or invalidating recorded hashes.
Original files remain available locally. In a fresh checkout, reconstruct them
before following original report paths or running historical CPU audits. Primary
summaries and runtime/test code remain directly readable; compact native report
CSVs are also archived because the repository ignores CSV files. Reconstruct
them using the manifest before opening the original CSV paths.
Raw tensors and Tracy payloads are intentionally local and can be regenerated
with the recorded fixture/profile commands.
