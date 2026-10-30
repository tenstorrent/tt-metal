# Stage Review

Verdict: clean-pass

Final independent rereview, 2026-10-30, stage 02 fused decoder for text-only
`google/gemma-4-31B-it`, HF revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. This supersedes the interim
more-work-needed verdict. Reviewed live stage-owned changes on
`mvasiljevic/gemma4-31b-fresh-pipeline-v0811`, based on HEAD
`b682b80841d17efa285be1fc3196f2273dcbd393` before the required local checkpoint.
Final production source SHA256:
`cc1b358f90a2fadab24cbb3dd9826ffe32015fab46ccefac25c0c6c11c658149`.

## Required Work

- None. The batched-fill finding, affected boundary coverage, allocation-placement
  anomaly and pending acceptance evidence are resolved.
- Local checkpoint commits and their work-log SHAs follow this review under the
  stage-review workflow. This review does not claim those commits already exist.

## Other Concerns

- No additional correctness or stage-contract finding remains.
- The final graph reproduces the best correct decode candidates within measured
  run variation. README correctly avoids claiming a strict win over every noisy
  historical sample. Whole-layer device decode improves against the functional
  baseline in every configured row.
- DRAM utilization and useful-FLOP percentages are estimates, with assumptions
  disclosed; they are not hardware memory-controller counters.

## Hard-Check Gaps

- No unresolved gap for this stage. Six affected profiles match the final source;
  the two B1 profiles and B1 maximum-context results retain an unchanged runtime
  graph. Inspection confirmed later batching and relocation branches do not run
  at B1; the added setup row-index payload is 440 bytes. This evidence reuse is
  explicit in README and provenance, rather than silently presented as a rerun.
- Full-attention B36/B37 capacity evidence is explicitly limited to S33/page-table
  geometry. No universal batch cap or reduction in advertised context is inferred.

## Anomaly Ledger

- Observed anomaly: available batched paged fill was missing from the initial
  graph assessment.
  Evidence: original per-row K/V slice/fill loop; generic
  `paged_cache_nanobind.cpp:120-126` and batched paged-fill unit tests;
  `batched_fill/*controlled.json` and final `tt/fused_decoder.py`.
  Affected path: multi-request prefill cache fill.
  Control or comparison: six controlled ABBA comparisons at B8/16/32, four samples
  per path, sharing weights/inputs/cache; all medians favor batched fill.
  Likely subsystem: unnecessary per-row data movement and dispatch.
  Investigation performed: API/factory/writer inspection, real-weight candidate
  checks, watched B32 tails/continuation/replay, direct functional output/cache
  equivalence, and B110/B111 prefill-only boundary checks for both layer kinds.
  Resolution: fixed; final path prepares row identities during setup and uses two
  batched fills per chunk, with tested scalar/fallback branches.

- Observed anomaly: full-attention fused RoPE leaves K too low in L1 for larger
  legal batches, while still larger batches exceed an inherited cache-op limit.
  Evidence: `before_relocation_*`, `full_capacity_36_*.json`,
  `full_capacity_37_*.log`, `functional_batch55_control.log`, `AUTOFIX.md`.
  Affected path: full-attention decode cache update at B34+.
  Control or comparison: functional and repaired fused B36/S33 pass; both fail
  B37/S33 for physical L1 capacity. B55 also fails the functional cache update.
  Likely subsystem: cache-op static buffers plus resident K allocation placement.
  Investigation performed: both factories' B*Wt output sizing; recorded addresses;
  original-shard release before resharing; B33/B34/B36 checks and watched B36 replay.
  Resolution: fixed placement; inherited capacity failure controlled. Repaired K
  moves from 1409024 to 1507328. Configured B1/8/16/32 graphs are unchanged.

- Observed anomaly: initial B32 profiler loses prefill markers.
  Evidence: `AUTODEBUG.md`, preserved initial capture, `AUTOFIX.md`, and final
  collection manifests/host metadata/ops CSVs.
  Affected path: profiling harness record capacity.
  Control or comparison: original B1 capture; repaired capacity 4096 with unchanged
  full prefill and 128 decode replays, no intra-layer drains.
  Likely subsystem: profiler DRAM capacity.
  Investigation performed: independently compared captured TT_DNN operation IDs
  and ENQUEUE counts to every replay session; checked producer warnings, report
  existence, source hashes, and recalculated device windows from FW timestamps.
  Resolution: fixed. All eight selected profiles have complete replay sets and no
  dropped markers. Six replacement profiles use the exact final production SHA.

- Observed anomaly: direct full-head concatenation, resident full Q during SDPA,
  and height-sharded inputs to RMSNorm hit explicit L1/layout constraints.
  Evidence: initial failure logs, `full_lifetime_retry2.log`,
  `full_lifetime_retry4.json`, `candidates_adapted/manifest.json`,
  `candidates_batches/*split_sharded*`, and generic RMSNorm validation.
  Affected path: attention head transforms and normalization.
  Control or comparison: adapted two-group prefill concat, DRAM full Q, and
  interleaved norm inputs pass PCC and have measured complete transfer costs.
  Likely subsystem: op memory contracts and static L1 requirements.
  Investigation performed: source validation plus failed/passing candidate pairs.
  Resolution: fixed or controlled; no rewrite is rejected solely on its first API error.

- Observed anomaly: the first batched acceptance queue requested S1025 with an
  extended harness requiring S>=2049.
  Evidence: preserved `input_error_*` and corrected command manifests.
  Affected path: test orchestration.
  Control or comparison: corrected S2049 runs pass for both kinds, including
  boundaries, continuation, ten replays and six interleaved prefills.
  Likely subsystem: queue arguments.
  Investigation performed: explicit harness assertion and clean device teardown;
  reuse restricted to already-passing commands with matching source hashes.
  Resolution: fixed; no numerical assertion was relaxed.

- Observed anomaly: optional browser-copy, mixed-column CSV and device-discovery
  notices appear in collection logs.
  Evidence: `AUTOFIX.md` classification; retained local Tracy files and raw CSVs.
  Affected path: reporting/discovery metadata.
  Control or comparison: complete strict reports and independently checked numeric
  timestamp/operation-ID fields; unchanged single-device baseline notices.
  Likely subsystem: optional output path and metadata formatting.
  Investigation performed: artifact existence/completeness checks and raw parsing.
  Resolution: controlled; no corruption, device failure or missing measurement is waived.

## Scope Inspected

- Goal/skills: AGENTS.md, run contract, agentic inputs, exact stage-02 continuation
  prompt; installed stage-review, graph-fusing, agentic-serving, tt-device-usage,
  model-bringup startup, review-core/router, model-bringup-review and trace-review.
- Production: entire standalone fused decoder and unchanged functional control;
  setup packing/index ownership, logical-tail views, both layer kinds, batch
  dispatch, cache ownership, runtime conversion boundaries and K relocation.
- Tests: correctness/reference/equivalence runners, candidate implementations and
  sweeps, acceptance/validation/boundary/capacity runners, profiler and summary
  scripts. Delivered correctness runners explicitly exercise FusedDecoder.
- Correctness artifacts: original acceptance/validation, final 17-job batched
  acceptance, all twelve supported batch-boundary checks, four B110/B111 prefill
  checks, capacity controls, both maximum-context results, direct equivalence,
  watched runs/logs and synthetic pytest. Both kinds preserve context 262144;
  minimum exact-maximum prefill PCC is 0.99819169755, above 0.995. Cache/output
  comparisons, changed page rows/positions and repeated replay pass.
- Performance artifacts: all four original candidate families and batched ABBA
  comparisons; selected B1 profiles in tracy_final and six current-source profiles
  in tracy_batched_fill; raw host/ops data, collection/report manifests, native
  tt-perf-report tables/CSV, performance_summary and profile_provenance. Source and
  copied CSV hashes match. Dominant runtime matmul rows are BF16/HiFi4.
- Independently rederived whole-layer measurements match summary JSON. Across all
  eight configured rows, prefill latency falls 26.27–28.05% and traced decode
  latency falls 0.99–3.64% versus functional. Each decode row has 128 complete
  replay windows whose operation IDs match the captured body.
- Documentation: final README tables and limitations, work-log reconciliation,
  pattern inventory/dispositions, AutoDebug/AutoFix, runtime/correctness/profile
  audits, context contract and eight agentic-serving measurements. Agentic
  measurements match their profile JSON exactly; the context ledger preserves
  functional controls and records the explicit weight-copy reason.
- Commands: read-only cat/sed/rg/nl/wc/tail, git status/rev-parse, sha256sum and
  small Python artifact-analysis scripts. No reviewer TTNN import, device access,
  tests, servers, resets or implementation edits. Only this review report was written.

## Residual Risk

- Representative layers 0/5 establish this stage, not full-model accuracy,
  autoregressive text quality, multichip behavior or serving readiness.
- Long-context HF comparisons sample final output rows with all required prefix KV;
  this inherited reference scope is disclosed and does not truncate TT execution.
- Future layout/precision/serving stages must revalidate their affected paths.
  This review approves the exact fused-stage source and evidence identified above.
