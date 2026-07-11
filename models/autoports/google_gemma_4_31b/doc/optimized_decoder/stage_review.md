# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None. The second review's remaining packed gate/up and reduced-core finding
  is closed by frozen-hash, policy-bound artifacts. Both legal block-1 packed
  output variants run the complete traced layer, pass real-weight decode PCC,
  and lose to the selected separate path: BF16 output is 1.518328 ms at PCC
  0.998721 and BFP8 output is 1.440399 ms at PCC 0.998637, versus 1.185875 ms
  for the final default. The intervening block-3 variants record exact
  static/live L1 collisions, and the block-7 variants record exact L1
  capacity overruns.
- The precision-locked four-core family was adapted from its initial failure to
  QKV block 3, gate/up block 3, O block 8, and down block 12. It reaches the
  down projection and records the documented exact blocker: static circular
  buffers end at 638,976 while the live allocation begins at 454,656. The
  artifact binds all policy fields and the frozen source hashes.

## Hard-Check Gaps

- None. The bounded-capacity logs at 262,144 and 262,113 predate
  `RUN_BINDING`, but `evidence/run_manifest.json` now says so explicitly and
  the README scopes them as legacy allocation/paging/self-consistency evidence,
  not full-history HF PCC. Frozen-hash evidence independently covers exact
  context with a distinct late-token periodic-history HF oracle and covers
  non-aligned logical lengths in the standard suite.

## Anomaly Ledger

- Observed anomaly: the initial combined exact-context capacity run reached
  pytest's 300-second timeout in the full-attention case.
  Evidence: `evidence/rejected_harness/context_262144_300s_timeout.{log,xml}`;
  the unchanged per-kind capacity run later passed with a 900-second limit,
  and the frozen-hash exact-context distinct-token oracle passed both kinds.
  Affected path: full-attention 262,144-token capacity harness.
  Control or comparison: `evidence/context_262144_full_bfp8.{log,xml}` and
  `evidence/context_262144_distinct_hf_oracle.{log,xml}`.
  Likely subsystem: harness timeout budget.
  Investigation performed: compared the rejected run, superseding runs,
  completion times, and device-close tails.
  Resolution: controlled.

- Observed anomaly: early candidate and pre-tail-fix artifacts contain
  intentional failures.
  Evidence: failed candidates record sub-threshold real-weight attention-BFP4
  PCC or exact L1/static-live blockers; obsolete harness artifacts are isolated
  under `evidence/rejected_harness/`.
  Affected path: candidate ranking and evidence lifecycle, not the final
  default.
  Control or comparison: the frozen default suite, final benchmark, profiles,
  watcher run, and candidate logs all bind hashes
  `9da6bf3e...` / `a096a0bc...` / `941dd1d1...`.
  Likely subsystem: expected optimization search failures and superseded cache
  update handling.
  Investigation performed: inspected XML statuses, tracebacks, effective
  policy JSON, source hashes, and superseding acceptance runs.
  Resolution: controlled.

- Observed anomaly: watcher output includes normal NOC legend/core-state text.
  Evidence: `watcher_final/generated/watcher/watcher.log` ends with completed
  dumps and normal device detach; no fatal, assert, NOC failure, overflow,
  sanitizer, or hang signature is present. The paired XML has four passes.
  Affected path: optimized mutable traced decode for both layer kinds and
  random/window-wrap positions.
  Control or comparison: the same nodes pass in the non-watcher standard
  suite.
  Likely subsystem: none; expected watcher reporting.
  Investigation performed: searched stage-critical signatures and inspected
  the log tail.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied Stage 03 contract;
  `.agents/skills/{stage-review,optimize,tt-device-usage}/SKILL.md`.
- Artifact paths: `doc/optimized_decoder/{README.md,work_log.md,
  stage_review_initial.md,stage_review_second.md,evidence/,candidates/,tracy/,
  watcher_final/}` and `doc/context_contract.json`.
- Code paths: `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`,
  inherited `tt/fused_decoder.py`, and the inherited functional-test helpers
  invoked by optimized wrappers.
- Commands run: read-only `git status`, `git branch`, `git rev-parse`, `find`,
  `sed`, `nl`, `rg`, `stat`, `sha256sum`, `jq`, and a small read-only XML
  summary script. No TT device, profiler, watcher, reset, server, or vLLM run
  was started during review.

## Residual Risk

- Review was inspection-only, as required. Runtime evidence is nevertheless
  complete and internally consistent: the frozen suite has 21 passes and 12
  explicitly gated benchmark/profile/long tests; both meaningful layer kinds
  clear PCC 0.995; batch 2/32, mutable trace, determinism, paged BFP8 cache,
  wrap and non-aligned lengths are exercised; watcher is clean; and exact
  context late-token PCC is 0.997758/0.998387 with wrong-position negative
  controls.
- The final bound profiler proves the delivered runtime policy at the material
  ops: BFP8/LoFi packed QKV and O, BFP4/LoFi separate gate/up/down, BFP8 KV,
  dedicated SDPA, and zero host ops. Same-harness final performance beats the
  fused baseline by 1.27-1.32x for prefill and 2.20-2.21x for traced decode.
  The post-candidate final rerun reproduces the selected best correct default,
  and the roofline/device/end-to-end accounting reconciles within the recorded
  0.035-0.037 ms dispatch gap.
