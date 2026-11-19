# Qwen3.6-35B-A3B fused decoder

Selected source `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87` passes the complete fused hard gates. Independent [review round2](stage_review_round2.md) returned **clean-pass**; both prior blockers are closed. This stage changes only the fused decoder and its tests/docs; no later pipeline stage has started.

The runtime is `FusedDecoder`, with no functional fallback. The current evidence covers linear attention (layers 0 and 34), full attention (layer 3), paged KV ownership, mixed batch 32, continuation, deterministic replay, fresh HF endurance, and Watcher. Advertised context remains **262,144**, with both aligned and non-aligned maximum prefills and final-position traced decode. The whole-output PCC bar remains 0.995; all delivered acceptance tests also pass their existing stricter row/contribution checks.

## Measured prefill and decode

Serial physical Blackhole chip 0, S256/context512, Watcher/profiling/allocation tracking disabled. Selected clean runs use 10 warmed prefill and 60 traced warmed decode samples. Functional rows are the preserved starting baseline; the best-correct ranking separately compares matched pure-harness runs, with exact source and version checks. Functional-equivalence checks remain in `final_0.json` and `final_3.json`; their timing setup runs an extra control and is not used to rank pure benchmarks.

| Layer | Implementation | Prefill ms | Traced decode ms | Prefill PCC | Decode PCC |
|---|---|---:|---:|---:|---:|
| 0 | [Functional baseline](baseline_0.json) | 70.047114 | 60.434721 | 0.999928249 | 0.999928290 |
| 0 | [Selected fused](final_pure_0.json) | 478.131568 | 1.447585 | 0.999993613 | 0.999996072 |
| 3 | [Functional baseline](baseline_3.json) | 69.296452 | 60.209946 | 0.999893672 | 0.999909163 |
| 3 | [Selected fused](final_pure_3.json) | 8.937745 | 0.583101 | 0.999890040 | 0.999923261 |

The selected code has faster traced decode than every qualified comparator in [the checked best-correct manifest](endurance/final_selected_best_correct_comparisons.json), including the faster 85ad linear candidate and the a478 full-attention candidate. Three rotated selection blocks independently gave linear decode medians 1.451205 ms (44f87), 1.449314 ms (85ad), and 1.447426 ms (selected d810); every selected run beat every comparator run. These are small final gains, and tuning has stopped. The comparator manifest retains its preparatory status as an immutable checker input; `hard_gates.json` and `final_manifest.json` report the completed result. [Selection results](endurance/v6_d810_selection_result.json).

Linear prefill is slower than the original functional baseline: ordered projections, exact routing/normalization, and the corrected recurrence are required to pass the stronger long-stream and layer-34 accuracy gates. Earlier faster V4/V5 graphs fail that gate and are ineligible. Against the corrected 52df baseline (2363.336 ms prefill, 2.447939 ms decode), fusion substantially improves both phases. Against the nearest 44f87 candidate (~484 ms prefill), the transpose reader reduces prefill to ~478 ms. No paired near-capacity speedup is claimed; the completed c823 long baseline is supplemental only.

## Correctness and context

| Layer / seed | Decode rows | Minimum decode PCC | Minimum contribution PCC |
|---|---:|---:|---:|
| [0 / 913](stress_endurance_0.json) | 24576 | 0.999914555 | 0.999913531 |
| [0 / 42](stress_endurance_seed42_0.json) | 24576 | 0.997974091 | 0.997281713 |
| [0 / 154](stress_endurance_seed154_0.json) | 24576 | 0.999683934 | 0.999605344 |
| [34 / 154](stress_representative_34_seed154.json) | 24576 | 0.999966639 | 0.999962902 |
| [3 / 913](stress_endurance_3.json) | 24576 | 0.997140776 | 0.996925957 |

Each endurance row above represents 8,192 changing-input decode positions repeated three times, plus 33 continuation tokens, with fresh HF references, included program-cache allocations, one capture and bit-equal repeated output. Layer 34 now has zero failing rows; this repairs review round 1 P1. Additional 128/1,024-step runs, boundary lengths 1/7/31/32/33/127/128/129/255/256/257/513/4097, batch-32 state/cache checks, page extents, padding sentinels and cross-request trace reuse pass. [Hard gates](hard_gates.json), [Watcher audit](watcher_audit.json), [context contract](../context_contract.json).

Current-source [linear](linear_long.json) and [full-attention](full_long.json) runs pass 262143/262144 prefill and deterministic decode at position 262143. Full attention additionally checks all 64 trailing rows. No context reduction or public chunk-alignment restriction was introduced. Linear internal chunk64 and full internal chunk256 are implementation details; the public chunk hint is4096.

## Implementation, profiling and choices

The selected runtime retains native paged SDPA, partial RoPE and compatible head/MoE fusion for full attention. Linear attention uses corrected increasing-K arithmetic, a canonical 64-token recurrence, a device selector matching the required CPU tie semantics, exact convolution/SiLU/carry, terminal norm scans, compact SiLU and direct decay tables. Its corrected matmul reader consumes transposed operands directly; the weighted-tail reader consumes interleaved inputs directly. These remove intermediate transfers while preserving arithmetic and persistent state.

[Current tt-perf-report tables](fused_decoder.md) cover one warmed S256 prefill and 16 warmed traced decodes for each kind; CSVs, operation sequences and raw logs are in `tracy/layer_0`, `tracy/layer_3` and the capture directories named in [the final manifest](final_manifest.json). [Completeness](profiler_completeness.json) verifies all20 replay sequences, 16 measured replays, no dropped markers, positive durations and no overlap. Kernel sums exclude host/profiler gaps and must not be substituted for clean latency. The full-attention native layout warnings have a capture-specific metadata review proving all44 final router score/index outputs have requested DRAM placement.

[All29 graph-fusing patterns and remaining movement](endurance/v6_d810_pattern_reconciliation.md) are assessed with retained/rejected evidence. [Source audit](fallback_audit.json) checks47 reachable runtime helpers,77 negative controls,80,784 planner and168 dispatch cases. Measured forward paths use device operations only; torch upload/readback is setup/validation. Remaining conversions satisfy a consumer layout, precision, state or head-layout requirement, with slower or numerically failing removal attempts recorded.

Rejected options include native router tie behavior, native substitutions at sensitive norm/projection boundaries, full SiLU tables that collide in L1, and both six-dispatch recurrence adapters. Those adapters are bit-exact but traced latency is 5.30x/5.59x native, so neither is used. [Rejection evidence](endurance/v6_recurrent_fusion_rejection_summary.md), [AutoFix](endurance/AUTOFIX_v6_dynamic_selector.md), [pattern history](pattern_assessment.json). Exhaustive conversion diagnostics retain an underflow/signed-zero limitation; no arbitrary-input CPU bit-equivalence is claimed beyond the tested contract.

## Stop rule, commands and artifacts

The owner ended tuning at the stage time/token cap and requested each remaining gate once. This explicitly supersedes the earlier eight-hour/70%-roofline requirement; neither is claimed. Untried selector/softmax fusion, recurrence alternatives, direct reduction producers and geometry/precision work are hand-off notes under **Not tried (stop rule)** in [the work log](work_log.md). No additional capacity sweep or small-gain tuning was run after the stop.

Activate `python_env/bin/activate`, source `../lane-env.sh`, set `QWEN_DECODER=fused PYTHONPATH=.`, and unset `TT_VISIBLE_DEVICES` on this host. Device jobs run serially on physical chip0. The final suite is `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/validate_fused.sh`; the interrupted run resumed only its unfinished suffix. Pure benchmark commands use `tests.benchmark_fused_snapshot --snapshot <selected snapshot> --layer 0|3 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60`. Profile commands are `bash doc/fused_decoder/endurance/v6_profile_d810_warmed_s256_plan/capture.sh 0|3` from the repo with its complete model prefix. Aggregate check: `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.check_fused_evidence --best-correct-manifest models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/final_selected_best_correct_comparisons.json`. Exact commands and environment are in adjacent provenance and [work_log.md](work_log.md).

The owner requested logs plus one current [manifest](final_manifest.json), without another per-artifact archive pass. Large raw outputs remain local; old archive manifests preserve historical runs and must not be restored over the final results. A clean checkout needs rerunning the documented gates to reproduce local raw outputs. The preserved historical evidence and `before_final_d810` remain explicitly historical. Local checkpoint `ad48340200720f4c24d49a97321bd4cd2c5a5922` predates final qualification; review round2 is clean-pass; the final local checkpoint is recorded in the work log. Never push.

The adjacent `stress_representative_34_seed154.json.tensors.pt` is the preserved **round-1 failure**, not tensor output from the current passing run. The harness writes that file only on failure and leaves it in place after a later pass. Current acceptance uses the JSON, provenance and execution logs indexed by `final_manifest.json`. Linear prefill is slightly slower than two repeated 85ad runs (by about0.056% and0.021%); all nine final **decode** comparisons pass.

Completed stage checkpoint: `476cfbbfb317210a09a5fc0928324c645fff7dc0` on local `main`; hard gates PASS and review clean-pass. See [work log](work_log.md) and [manifest](final_manifest.json) for commit receipt and artifact paths.
