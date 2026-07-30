# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Prefill MoE still executes the canonical standalone GeGLU path
  Evidence: `FusedDecoder` overrides `_dense_mlp` and
  `_moe_decode_single_user`, but it does not override `_moe_prefill` or
  `_moe_prefill_chunk`. `FunctionalDecoder._moe_prefill_chunk` delegates to
  `models.demos.gemma4.tt.experts.prefill.prefill_forward`, whose canonical
  implementation uses `apply_geglu`. The fused-path test
  `test_fused_hot_path_has_no_host_fallback` inspects only `_dense_mlp` and
  `_moe_decode_single_user`; it cannot detect the unfused prefill expert path.
  The README's statement that the stage fused the applicable prefill graph is
  therefore incomplete, and the graph-fusing audit did not exhaust the
  stage-critical MoE prefill pattern.
  Why this matters: Stage 02 requires all applicable graph-fusing patterns to
  be exhausted and requires warmed prefill to improve. The final prefill path
  still contains the exact standalone GELU-plus-multiply pattern this stage is
  intended to remove.
  Required next step: Implement and validate a fused prefill-MoE GeGLU path (or
  earn a rejection with an adapted experiment and measured correctness/perf
  evidence), add a fused-path test that inspects/executes prefill MoE through
  `FusedDecoder`, and regenerate prefill PCC/performance/profiler evidence.

- P1: The required prefill/decode profiler matrix is missing, and the retained
  decode report is not sufficient `tt-perf-report` evidence
  Evidence: The stage retains one profiler collection, sliding-attention
  batch 1. There are no profiler tables/CSVs for full-attention prefill,
  full-attention decode, or either layer kind at serving batch 32. The claimed
  decode CSV begins with the prefill-end signpost and its displayed device
  times are zero. Its conversion log says no device architecture was present,
  defaults to Wormhole, and reports 64 workers despite the run being claimed
  as Blackhole/P300. The work log explicitly says this file was made by
  extracting trace-capture rows because trace replay itself emitted no TTNN op
  rows. This can describe capture topology, but it is not a valid decode
  performance report for traced replay.
  Why this matters: The contract explicitly requires prefill and decode
  `tt-perf-report` CSV/tables and all required layer/batch profiler cases.
  Capture topology plus host replay latency does not supply per-op decode
  performance, and the architecture mismatch makes the generated advice/table
  unreliable.
  Required next step: Produce correctly attributed profiler evidence for both
  layer kinds and required batches, separating trace-capture topology from
  traced replay latency. Ensure the `tt-perf-report` inputs carry the correct
  Blackhole architecture/worker metadata and nonzero device measurements, or
  record an evidence-backed tool limitation and provide the accepted
  equivalent device-profiler evidence required by the contract.

- P1: Most required final-path evidence is stale relative to the shipped source
  Evidence: The current `fused_decoder.py` SHA-256 is
  `519ad63db255192f9e70657d2779d5ca05f612f6b5ed299849265acf82139242`.
  Only the final full-attention batch-1 timing JSON records that hash.
  Sliding batch 1, sliding batch 32, and full batch 32 record the older hash
  `383ef398...`. The watcher log ended at 23:55:37, while the final source was
  modified at 00:05:18, so the watcher run also predates the final router-scale
  folding change. Sliding PCC, boundary, trace, and batch-2 artifacts similarly
  predate the final source; most correctness artifacts contain no provenance
  hash to establish otherwise.
  Why this matters: The stage requires correctness, determinism, stress, and a
  watcher-clean run for the delivered implementation, not an earlier candidate.
  Router constant folding changes both prefill and decode numerical paths.
  Required next step: Rerun the complete required fused correctness,
  non-aligned/paged-cache, trace/determinism, stress/repeated, and separate
  watcher suite on the final source. Add immutable provenance tying each
  artifact to the final decoder/test/build hashes and exact command.

- P1: The final speedup claim is not demonstrated across the required matrix
  Evidence: The README's main table contains stale results. For example it
  claims fused sliding batch-1 prefill/decode of 679.259/3.02647 ms, while the
  live same-named JSON contains 680.137864/3.060859 ms. The table also retains
  a full batch-1 regression (3.21499 ms versus 3.21154 ms). Later prose
  supersedes only full-attention batch 1 with a 201-replay/7-prefill comparison
  (3.21365/680.599 versus 3.21538/680.699). The other three decode cases and
  sliding prefill remain older-source, 11/3-sample, separate-process results.
  Why this matters: The contract says the final implementation must beat the
  correct traced baseline at batch 1 and serving batch 32. One higher-resolution
  full-b1 result neither supersedes the stale table unambiguously nor proves the
  final source wins for sliding b1 or either b32 case.
  Required next step: Measure final-source functional and fused paths in
  controlled, comparable runs for sliding/full prefill b1 and traced decode
  sliding/full at b1/b32. Freeze run-identified artifacts, state which rows
  supersede older rows, and generate one internally consistent final table.

- P2: Delivered tests select `FusedDecoder`, but fused-specific coverage is too
  weak to prove the delivered fused contract
  Evidence: `GEMMA4_DECODER_IMPL=fused` aliases `FusedDecoder` as
  `FunctionalDecoder` before hardware tests instantiate via
  `from_state_dict`, so the environment-selected suite genuinely constructs
  the subclass. However, `tests/test_fused_decoder.py` is host-only and covers
  only dense MLP, setup folding, and decode MoE source inspection. It has no
  assertion for the router override, no prefill MoE fusion assertion, and no
  evidence-provenance assertion that prevents stale artifacts from being
  accepted.
  Why this matters: The selection mechanism is valid, but current coverage let
  an unfused stage-critical path and stale final-source evidence pass unnoticed.
  Required next step: Extend fused-specific tests to cover every overridden or
  intended-to-be-overridden graph and verify artifact provenance for the final
  selected class.

## Other Concerns

- The operation-topology table says the router final sequence is "unchanged"
  even though the final implementation folds two router scale factors into
  setup. Update the topology and candidate ledger to reflect the delivered
  graph.
- `context_contract.json` points only to functional-decoder 262144/262143
  artifacts. Inheritance supports the unchanged attention/cache capacity
  argument, but the final subclass changes the router path. The refreshed final
  fused suite should explicitly state why inherited capacity artifacts remain
  applicable and include current-source fused checks at representative
  non-aligned and advertised-context decode positions.
- The four modified `doc/functional_decoder/*host_timings.json` files and the
  shared `test_functional_decoder.py` harness are identifiable stage-02
  baseline/harness changes, but the work log should enumerate them as
  stage-owned before checkpointing so they can be isolated from unrelated dirty
  files. The unrelated GPT-OSS and `.agents` untracked paths must not enter the
  stage commit.

## Hard-Check Gaps

- No immutable manifest ties all PCC, trace, boundary, watcher, profiler, and
  timing artifacts to one final source/test/build revision.
- No final-source watcher-clean evidence exists after router folding.
- No current final-source stress/repeated artifact beyond host timing samples
  and two trace replays is identified.
- README commands omit the exact profiler extraction/conversion commands,
  201/7-repeat performance commands in executable form, and artifact-generation
  commands for the full correctness/boundary/context suite.
- No profiler evidence exists for all required layer/batch combinations.

## Anomaly Ledger

- Observed anomaly: Prefill MoE remains on canonical `apply_geglu`.
  Evidence: `FusedDecoder` has no prefill-MoE override;
  `FunctionalDecoder._moe_prefill_chunk` delegates to canonical sparse expert
  prefill.
  Affected path: Fused prefill for both sliding and full-attention layers.
  Control or comparison: Decode MoE and dense MLP explicitly fold GELU into
  `ttnn.mul`.
  Likely subsystem: Graph-fusing coverage and fused-specific tests.
  Investigation performed: Static dispatch/source inspection of both decoder
  classes and tests.
  Resolution: more-work-needed

- Observed anomaly: README timing rows disagree with live artifacts and mix
  old and final source hashes.
  Evidence: Sliding b1 README 679.259/3.02647 ms versus JSON
  680.137864/3.060859 ms; three timing files record `383ef398...`, current
  source is `519ad63d...`.
  Affected path: Final performance selection at b1/b32.
  Control or comparison: Only full b1 has a current-source 201/7-repeat control.
  Likely subsystem: Evidence lifecycle/provenance.
  Investigation performed: Recomputed source hashes and compared JSON
  provenance, timestamps, and README rows.
  Resolution: more-work-needed

- Observed anomaly: Decode `tt-perf-report` defaults to Wormhole and reports
  zero device times.
  Evidence: Conversion log and CSV contents in
  `tracy/decode_capture_sliding_batch1.{txt,csv}`.
  Affected path: Claimed traced-decode profiler evidence.
  Control or comparison: Host trace replay JSON has nonzero wall latency but no
  op attribution.
  Likely subsystem: Tracy signpost/capture extraction and report conversion.
  Investigation performed: Read raw slice, converted CSV, and conversion logs.
  Resolution: more-work-needed

- Observed anomaly: Watcher passed before the final implementation edit.
  Evidence: Watcher log ended 2026-07-29 23:55:37; final source mtime is
  2026-07-30 00:05:18.
  Affected path: Watcher-clean final-stage gate.
  Control or comparison: None tied to current source hash.
  Likely subsystem: Evidence sequencing.
  Investigation performed: Compared filesystem timestamps and artifact hashes.
  Resolution: more-work-needed

## Scope Inspected

- Goal/skill paths:
  `/home/mvasiljevic/tt-metal/.agents/skills/stage-review/SKILL.md`,
  `/home/mvasiljevic/tt-metal/.agents/skills/graph-fusing/SKILL.md`,
  `/home/mvasiljevic/tt-metal/.agents/skills/tt-device-usage/SKILL.md`, and the
  supplied stage-02 contract.
- Artifact paths:
  `doc/fused_decoder/README.md`, `work_log.md`, `AUTODEBUG.md`, all fused JSON
  artifacts, watcher log, Tracy raw/report/converted files,
  `doc/context_contract.json`, functional baseline timing/context artifacts,
  and git status/diffs.
- Code paths:
  `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_fused_decoder.py`, `tests/test_functional_decoder.py`, and the
  imported canonical Gemma4 sparse-expert prefill/GeGLU path.
- Commands run:
  Read-only `sed`, `tail`, `find`, `grep`, `git status`, `git diff`, `jq`,
  `sha256sum`, and `stat` inspections. No server, device, reservation, or
  hardware experiment was run.

## Residual Risk

- This was an artifact/code review only, as required. Runtime remediation must
  be verified on hardware by the stage owner.
- The final full-b1 improvement is small (about 0.054% decode and 0.015%
  prefill) and comes from sequential process runs; even after the missing
  matrix is filled, a controlled paired/interleaved comparison may be needed
  to establish that the claimed win is reproducible rather than host jitter.
