# Prefill performance and kernel attribution

## Clean full-model boundary measurements

`before_prefill.json` and `final/prefill_boundaries.json` use the same boundary harness and fixture: all 40 layers, mesh 1×4, batch 1, context 4096, serial requests, one same-length warmup and one timed sample per length. The boundary includes public prefill, terminal norm/head, sampler-logits handoff, sampling and first-token delivery. The two subsequent traced decode steps are correctness checks outside that TTFT timer. Watcher, profiler and trace allocation tracking are disabled for these measurements. The harness/source revisions differ and are recorded in each original JSON; this is a comparable protocol, not identical source.

| Logical prompt tokens | Before TTFT (ms) | After TTFT (ms) | Latency change |
|---:|---:|---:|---:|
| 64 | 1474.063 | 1473.618 | -0.030% |
| 255 | 5905.173 | 5957.423 | +0.885% |
| 256 | 5350.840 | 5534.284 | +3.428% |
| 257 | 5975.767 | 6106.160 | +2.182% |
| 1025 | 21825.866 | 22067.904 | +1.109% |
| 2049 | 43157.731 | 43703.507 | +1.265% |

Exact floating-point source values and artifact paths are in `before_after_prefill.csv`. The 64-token sample is effectively unchanged; the other final samples are 0.885–3.428% slower. These single samples do not establish a statistically meaningful regression, an improvement, or a cause. Decode-kernel improvements cannot automatically be credited to prefill. There is no matched before/after prefill kernel capture here, so the final native capture can locate current work but cannot explain these deltas causally.

All six final boundaries passed. First and final sampled token IDs match the baseline at every length; this repeated AIME-token fixture is state/contract evidence, not quality accuracy. After prefill each case made two model and two sampler replays with zero token/position/page-table refreshes, zero readbacks inside the measured decode counter interval, and no trace capture. Positions reached prompt length + 2. Program-cache entries stayed fixed within each run (9602 before, 9590 after); stable census and unchanged trace IDs were checked. Non-aligned lengths 255, 257, 1025 and 2049 remain supported.

## Chunk schedules justified by source

`tt/prefill.py:26` plans linear attention in at most 64-token pieces. Full attention uses up to 256 tokens, rounded down to a multiple of 32 only at a 256-aligned absolute offset; otherwise it advances one token. For example, a fresh 255-token prompt plans one 224-token full-attention chunk plus 31 single-token steps, while 256 plans one 256-token chunk and 257 plans 256 + 1. This explains why a simple monotonic latency assumption across these boundaries is invalid; it is not proof of the observed latency difference.

`tt/prefill.py:198` uses 64-token embedding buffers with a separate valid-row count. `tt/prefill.py:213` uses up to 2048 logical MoE rows, physically rounded to 32 except a singleton. `tt/prefill.py:237` runs terminal logits for the final token unless the caller requested the separate callback path. Those are source schedules, not observations of every such logical chunk in the reduced 128-token capture.

## Final native capture: measured 128-token window

The bounded wait on `logs/final_measurement_driver.log` matched `DONE profile_driver` with exit 0. `final/profile_reduced/performance.json` reports passed capture on real layers 0 and 3, batch 1, mesh 1×4, and the pinned 128-token AIME prefix. This reduced capture is attribution evidence; it is not an all-40-layer substitute for the clean boundary measurements above.

`report_prefill_kernels.py` located `PERF_TTFT` / `PERF_TTFT_END` at native CSV lines 722674 and 735267. Within that window, 12,468 untraced and 124 traced device rows across all four ranks produce 1,564 operation/program/shape groups. No measured row lacks device duration. The profiler's host TTFT is 15761.827422014903 ms, explicitly including eager-component and terminal-handoff profiler drains/synchronizations. It must not appear as a clean TTFT headline.

The summed untraced kernel work ranges from 73.964493 to 74.084099 ms per rank; traced sampler/handoff work ranges from 0.428130 to 0.441624 ms per rank. These are per-operation work sums, not a reconstructed critical path, host-overhead estimate or additive multi-device latency. Firmware timing is retained separately in the detailed CSV.

The compact table below reports the largest physical-axis groups plus the semantically identifiable embedding, SDPA and router groups. All 40 groups, full-precision derived values and source paths are in `prefill_role_chunk_summary.csv`. The physical axis is input Y except embedding lookups, where it is the index-buffer width. A blank logical chunk remains unknown. Means are arithmetic means of four separate per-rank work sums, not means across repeated runs.

| Scope | Role | Physical input axis | Verified logical chunk | Ops/rank | Mean rank kernel work (µs) |
|---|---|---:|---:|---:|---:|
| untraced | custom_kernel_role_unresolved | 64 | unknown | 18–18 | 49434.568 |
| untraced | custom_kernel_role_unresolved | 1024 | unknown | 141–141 | 11401.057 |
| untraced | pointwise_or_reduction | 32 | unknown | 1558–1558 | 3410.208 |
| untraced | layout_or_cache_movement | 64 | unknown | 480–480 | 2740.632 |
| untraced | layout_or_cache_movement | 1 | unknown | 273–273 | 1818.942 |
| untraced | pointwise_or_reduction | 3008 | unknown | 4–4 | 1800.015 |
| untraced | layout_or_cache_movement | 32 | unknown | 287–287 | 745.449 |
| untraced | pointwise_or_reduction | 64 | unknown | 205–205 | 709.344 |
| traced | sampling_terminal | 32 | unknown | 4–4 | 331.895 |
| untraced | lm_head_matmul | 32 | unknown | 1–1 | 284.250 |
| untraced | matmul_role_unresolved | 128 | unknown | 8–8 | 274.457 |
| untraced | layout_or_cache_movement | 128 | unknown | 39–39 | 263.840 |
| untraced | collective | 128 | unknown | 3–3 | 175.116 |
| untraced | pointwise_or_reduction | 128 | unknown | 28–28 | 166.856 |
| untraced | expert_router_topk_shape_inference | 128 | unknown | 2–2 | 87.384 |
| untraced | attention_sdpa | 128 | 128 | 1–1 | 45.016 |
| untraced | token_embedding | 64 | unknown | 2–2 | 8.005 |
| untraced | rope_lookup_shape_inference | 128 | unknown | 2–2 | 5.476 |

The actual shape evidence supports these distinctions:

- Two token-embedding calls per rank use index width 64, vocabulary table `[1,1,248320,512]`, and output `[1,1,64,512]`. With the known 128-token request and the source loop, these are the two complete 64-token embedding pieces. Generic embedding buffers alone would not reveal a valid tail length.
- Two other embedding calls have table `[1,1,1024,64]` and index width 128. They are the cosine/sine RoPE lookups from `tt/prefill.py:110`, not 128-token vocabulary-embedding chunks.
- The non-decode SDPA query is `[1,4,128,256]`: one 128-token logical full-attention chunk per rank. There is no measured 256-token SDPA or non-aligned tail in this window.
- The untraced generic `TopKDeviceOperation` consumes `[1,1,128,256]` and emits eight logical expert choices. This is the two layers’ expert routing, identified from expert count and source, not the greedy vocabulary sampler. The traced sampler uses the separate 62080-wide partition TopK operations and `SamplingDeviceOperation`.
- The 18 custom operations per rank with physical input Y=64 have eight-head `[1,8,64,128]` or `[1,8,64,64]` shapes. These match the two `chunk64_state` calls selected by `tt/multichip_decoder.py:545` and the 64-token source planner. Their 49.435 ms mean rank work is the largest observed group. This is a source/shape inference about the selected recurrence family; the native metadata inventory does not identify these individual inline programs, so the CSV keeps their kernel roles unresolved rather than claiming a verified one-to-one mapping.
- Of the physical-Y=1024 custom group, 128 calls per rank consume weight `[2,1,1024,3136]` and one activation row, exactly matching the per-token packed-input projection scan loop in `tt/multichip_decoder.py:677`. The 1024 dimension is reduction K, not a 1024-token prompt chunk. The other 13 calls in this group have different shapes and remain separate in the detailed table. A broader “1024-token prefill kernel” label would be incorrect.

The large chunk64/scan groups preserve the selected arithmetic policy. This capture suggests where a future policy-preserving prefill optimization should concentrate; it does not authorize a rejected native-matmul or precision policy, establish an improvement over a matched baseline capture, or prove all inline-kernel precision/configuration choices. Only 668 of the 12,468 measured eager device rows have entries in the decode-oriented native program metadata inventory. Missing entries and scalar-CSV fidelity limitations are explicit in the detailed report.

## Preparation and warmup remain a separate scope

Before `PERF_TTFT`, the report found 717,948 timed untraced and 3,804 timed traced device rows. Another 920 setup rows lacked durations and are counted separately, not converted to zero. These rows mix weight setup, physical-bucket preparation, capture and warmup. They cannot be presented as warmed prefill measurements or folded into the 128-token headline.

The setup-only non-decode SDPA sequence axes are 32, 64, 96, 128, 160, 192, 224 and 256. Setup RoPE index widths include those sizes plus 1; token-embedding widths are 1 and 64. These are observed prepared/warmed physical variants, not proof that all those logical request lengths were timed. `setup_physical_buckets.csv` retains every observed role/axis/rank group without inventing logical chunk boundaries for generic matrix or recurrence dimensions.

## Artifacts and limits

- `before_after_prefill.csv`: exact clean full-model paired boundary values and scope.
- `prefill_role_chunk_summary.csv`: compact 40-row measured role/physical-axis table.
- `final/profile_reduced/prefill_kernels/prefill_kernel_groups.csv`: 967 KB detailed measured kernel table, retained at its persistent path rather than archived or rewritten to meet commit size limits.
- `final/profile_reduced/prefill_kernels/setup_physical_buckets.csv` and `prefill_kernel_summary.json`: separate setup attribution, missing-duration counts, window lines and source paths.
- `final/profile_reduced/ops.csv`, `performance.json`, `native_program_metadata.json`, and existing native component/perf-report exports remain the original evidence. This task added no capture archive or per-artifact hash mechanism.

This report closes the scoped CPU attribution task, not stage acceptance. Non-aligned all-40-layer boundary behavior is evidenced separately from the reduced 128-token kernel window. Kernel costs for every non-aligned logical chunk are not individually measured; a new labeled capture would be required for that finer attribution.
