# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Finish the packed gate/up and reduced-core geometry families before
  declaring the selected separate path globally best.
  Evidence: both bound packed-gate/up failures use the same eight-core
  `in0_block_w=7` program (`candidates/packed_gate_up_bf16_blocker.log` and
  `packed_gate_up_bfp8_blocker.log`). They change only output dtype and fail at
  2,516,224 B / 2,699,008 B versus 1,572,864 B L1. For K=5376 on eight logical
  shards, block widths 3 and 1 are legal, and the optimize skill explicitly
  requires reducing `in0_block_w` after L1 OOM before accepting the blocker.
  The bound four-core artifact likewise runs the default gate block 7 and
  fails with static end 1,336,320 versus live allocation 1,142,784; it does not
  contain the documented gate-3/down-12 adaptation or 638,976/454,656 down-path
  collision claimed in `work_log.md`.
  Why this matters: gate/up are two of the three largest decode rows (201/208
  us in the final sliding report), and packed same-input projections plus
  smaller-core/wider-shard geometry are mandatory material candidates. The
  existing evidence proves two configurations fail, not that the topology
  family is inexpressible or slower.
  Required next step: run the packed topology with legal reduced block widths
  (at least 3 and, if needed, 1) and coherent output/layout handling; measure
  whole-layer traced latency and real-weight correctness if one runs. Reproduce
  the claimed four-core gate-3/down-12 adaptation with a bound artifact, or
  correct the documentation and continue legal coherent core/block variants.
  If the adapted family still fails, retain the exact final blocker. Then
  freeze, rerun affected final gates, and request another independent review.

## Other Concerns

- The three capacity logs for 262,144/262,113 predate the final source freeze
  and do not print `RUN_BINDING`, despite `run_manifest.json` saying every final
  pytest log is bound. This is not the primary gate because the frozen-hash
  distinct late-token HF oracle passes at the exact context and the frozen-hash
  standard suite covers non-aligned logical lengths. The manifest wording and
  provenance of the published bounded-capacity numeric rows should nonetheless
  be made precise on the next finalization pass.

## Hard-Check Gaps

- There is no bound artifact for the work log's down block-12 timing or its
  claimed adapted four-core down-path collision.
- Candidate timing nodes intentionally do not run PCC. This is acceptable for
  dtype-identical program-only changes when tied to the frozen final
  correctness suite, but any packed topology that runs needs its own
  real-weight cache-consuming correctness result because it changes splitting
  and elementwise/layout behavior.

## Anomaly Ledger

- Observed anomaly: the initial exact-context combined run hit pytest's
  300-second timeout during full attention.
  Evidence: `evidence/rejected_harness/context_262144_300s_timeout.{log,xml}`;
  later full and non-aligned capacity runs passed with a 900-second bound.
  Affected path: full-attention advertised-context capacity harness.
  Control or comparison: the frozen-hash distinct-token exact-context oracle
  passes both layer kinds; later device-facing runs and watcher completed
  normally.
  Likely subsystem: harness timeout budget.
  Investigation performed: compared rejected and passing logs and device-close
  tails.
  Resolution: controlled.

- Observed anomaly: stale pre-tail-fix and pre-binding artifacts are retained.
  Evidence: they are isolated under `evidence/rejected_harness/`; final suite,
  timing, watcher, Tracy, and candidates use current hashes.
  Affected path: evidence provenance only.
  Control or comparison: current SHA-256 values match `run_manifest.json` and
  all authoritative `final_bound` profile console logs.
  Likely subsystem: stage evidence lifecycle.
  Investigation performed: checked hashes, mtimes, `RUN_BINDING`, and report
  paths.
  Resolution: controlled, except for the capacity-log wording noted above.

- Observed anomaly: watcher output contains normal NOC legend text but no
  watcher fatal/assert/NOC failure, overflow, sanitizer, or hang signature.
  Evidence: `watcher_final/generated/watcher/watcher.log` ends in a normal
  detach; `evidence/watcher_mutable_trace.{log,xml}` has four passing nodes.
  Affected path: optimized mutable traced decode for both layer kinds.
  Control or comparison: the same nodes pass in the non-watcher standard suite.
  Likely subsystem: none.
  Investigation performed: searched the watcher log and inspected its tail.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied Stage 03 contract;
  `.agents/skills/{stage-review,optimize,tt-device-usage}/SKILL.md`.
- Artifact paths: `doc/optimized_decoder/{README.md,work_log.md,evidence/,
  candidates/,tracy/,watcher_final/}`; `doc/context_contract.json`; preserved
  `stage_review_initial.md`.
- Code paths: `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`,
  inherited `tt/fused_decoder.py` and relevant functional test helpers.
- Commands run: read-only `git status`, `git rev-parse`, `find`, `sed`, `nl`,
  `rg`, `stat`, and `sha256sum`. No TT device, profiler, watcher, server, reset,
  or implementation/test mutation was performed.

## Residual Risk

- The frozen final default is otherwise well supported: current-hash suite 21
  passed/12 gated skips; exact-context distinct-token HF PCC is
  0.997758/0.998387; watcher is clean; profiler rows prove BFP8/LoFi attention,
  BFP4/LoFi MLP, BFP8 cache, and zero host ops; same-harness final latency beats
  fused by 1.27-1.32x prefill and 2.20-2.22x traced decode. Those results do not
  close the untried packed/reduced-block family above.
