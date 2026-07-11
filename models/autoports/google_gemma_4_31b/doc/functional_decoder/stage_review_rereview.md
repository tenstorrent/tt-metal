# Stage Review Rereview

Verdict: clean-pass

## Required Work

- None.

## Initial-Finding Closure

- The long-only attention paths now have correctness controls, not merely
  finite-value capacity smokes. The exact 262144-token sliding run compares the
  final token against a real-weight HF absolute-position 1024-token window at
  PCC 0.998786, and the exact full-attention run compares the streaming-path
  prefix through token 2049 against HF at PCC 0.999089. The corresponding
  262113-token non-aligned controls pass at 0.998854 and 0.999089. These tests
  force `_streaming_full_prefill_attention` or the long sliding branch and the
  long MLP chunking used by the advertised-context path.
- Paged-cache evidence is no longer based on identity tables. `_paged_state`
  rotates every user's physical block range by one block; even the two-block
  short cases therefore map `[0, 1]` to `[1, 0]`, and batch 32 owns 32 disjoint
  rotated physical ranges. Real-weight traced decode after populated prefill
  passes against HF at PCC 0.999655/0.999620 for batch 1 and
  0.999416/0.999480 for batch 32. The exact-limit runs additionally prefill the
  complete cache and replay traced decode at absolute position 262143 at PCC
  0.998715 sliding and 0.996611 full against the corresponding prefill token.
  The 262113 runs repeat this populated-cache control at a non-aligned absolute
  position. Lengths 63/64/65 and 1023/1024/1025 exercise tile, page, and sliding
  window edges, while the two advertised/near-advertised sliding runs exercise
  bounded-cache modulo ownership after many 1024-token wraps.
- Batch-32 decode now uses the complete traced path. The test warms, captures,
  replays, checks replay PCC, copies host contents into the already-captured
  token and both current-position device buffers, replays again without
  replacing their allocations, and requires bitwise-identical results. The
  final watcher rerun covers both batch-32 layer kinds.
- Final-revision provenance is now adequate. The manifest's runtime and test
  SHA-256 values exactly match the delivered files
  (`315990f7e060a464d935efadf0dd50a8b035e187947ec320d08e4a15479e7a85`
  and `45b5053b59e5a2ac7c1a56e1cfd1aac31989588cdfcc513b47e97a0581d09c45`).
  The repository hook changed only the import order in the runtime file, and
  the complete standard suite was rerun afterward: 17 passed with only the six
  explicitly environment-gated long/performance cases skipped. The final
  watcher and profiler runs predate that import-only formatting delta, but the
  measured runtime logic is unchanged. The oversized sliding-prefill profiler
  console was losslessly gzip-packaged; `gzip -t` passes and the decompressed
  log still contains both signposts, the selected passing node, and the final
  one-pass summary.
- The historical triage artifact is no longer empty and accurately records why
  the tool could not attach, the NOC0 signature, bounded reset/list recovery,
  and successful 1x1 mesh smoke. The runtime fallback audit now includes all
  local runtime methods and the transitive Gemma attention, decode, RMSNorm,
  and MLP helpers used by the measured path. The final watcher evidence was
  expanded from short batch 1 to traced batch 32.

## Other Concerns

- None requiring stage work. The exact-limit full-attention HF oracle is a
  reduced layer-level prefix control rather than a host evaluation of all
  262144 outputs. This is consistent with the functional-decoder skill's
  allowance for a reduced long harness: it executes the exact streaming path,
  covers multiple 1024-token chunks and a non-divisible prefix end, and is
  paired with a full-capacity populated-cache decode replay.

## Hard-Check Gaps

- Runtime fallback is established by expanded static call-graph inspection and
  by signposted profiler reports containing 0 host ops; there is no separate
  dynamic monkeypatch/interposition audit. The existing evidence is sufficient
  because the inspected runtime call graph contains none of the forbidden host
  conversion APIs and the measured windows show only device operations.
- The long, watcher, and profiler console logs do not embed their source hashes.
  The only subsequent runtime delta is independently visible as import
  reordering, its current hash matches the evidence manifest, and the complete
  post-hook standard suite ties the reordered runtime and current test source
  to a passing run, so this is not a stale-artifact blocker.

## Anomaly Ledger

- Observed anomaly: the original 196577-token full-prefill implementation
  stalled NOC0 and the triage tool could not initialize.
  Evidence: `logs/prefill_long_196577_full.log` and
  `triage/long_full_196577_tt-triage.txt`.
  Affected path: the superseded large-query full-attention path.
  Control or comparison: the replacement 1024-query streaming path passed
  262144 and 262113, with long-path PCC and populated-cache traced decode above
  threshold; reset/list and the 1x1 mesh smoke restored device health.
  Likely subsystem: excessive legacy SDPA query chunk/kernel duration followed
  by device NOC unavailability.
  Investigation performed: inspected the failure, triage-capture failure,
  recovery record, replacement source, and final exact/near-context logs.
  Resolution: controlled.

- Observed anomaly: `tt-perf-report` classifies several Gemma rotary/cache/SDPA
  operations as unclassified in its console output.
  Evidence: the four `*_perf_report.console.log` files.
  Affected path: report categorization only.
  Control or comparison: raw CSV signposts delimit each warmed window; the
  filtered reports retain the operations, report 0 host ops, and independent
  sums of `Device Time` reproduce 3532.537, 4247.842, 2578.397, and 2912.882
  microseconds exactly.
  Likely subsystem: report taxonomy metadata.
  Investigation performed: checked raw signposts, CSV rows, rendered tables,
  console provenance, and recomputed totals.
  Resolution: controlled.

- Observed anomaly: firmware bundle 19.9.0 is newer than the latest fully
  tested 19.5.0 bundle named in runtime logs.
  Evidence: standard, long, profiler, and watcher console logs.
  Affected path: test environment.
  Control or comparison: all final correctness runs pass, the separate watcher
  run detaches cleanly with no assert/NOC/overflow/sanitizer finding, and the
  recorded earlier device fault was recovered and controlled by the redesigned
  path.
  Likely subsystem: environment compatibility warning.
  Investigation performed: compared the warning across independent final runs
  and scanned the watcher output for fault signatures.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: the supplied google/gemma-4-31B functional-decoder contract;
  `.agents/skills/functional-decoder/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`; and the initial
  `doc/functional_decoder/stage_review.md`.
- Artifact paths: `doc/context_contract.json`; `README.md`; `work_log.md`;
  `evidence_manifest.json`; `real_weight_stats.json`; final standard, exact and
  near-context, watcher, recovery, and performance logs; all four raw/filtered/
  rendered Tracy report groups; and the final raw watcher log.
- Code paths: `tt/functional_decoder.py`;
  `tests/test_functional_decoder.py`; and the imported Gemma attention/decode,
  RMSNorm, MLP, and paged-cache helpers named by the fallback audit.
- Commands run: read-only `sed`, `nl`, `find`, `rg`, `stat`, `sha256sum`,
  `git status`, `git diff`, and a read-only Python CSV summation. No TT device,
  reset, server, profiler, or hardware test was started.

## Residual Risk

- The functional implementation is deliberately correctness-oriented and uses
  serial per-user prefill plus long-sequence streaming; its latency is not an
  optimized-decoder claim.
- Exact-context decode parity at the final token is TT prefill-to-traced-decode
  parity, while direct HF-vs-traced-decode parity is established at short
  batch-1 and batch-32 contexts. Combined with the exact-context long HF
  prefill oracles and populated nonidentity cache, this is sufficient for the
  layer-stage capability claim but leaves normal downstream full-model
  integration risk.
- Scope is cleanly contained to `models/autoports/google_gemma_4_31b`; no
  optimized decoder, multichip decoder, full model, generator, or vLLM code was
  introduced. Unrelated pre-existing dirty/untracked workspace entries are
  outside the stage and must remain excluded from its checkpoint commit.
