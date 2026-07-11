# Stage Review Final

Verdict: clean-pass

## Required Work

- None.

## Findings

- No P1 or P2 findings remain. The delivered implementation, tests, and
  hash-bound evidence satisfy the Stage 02 fused-decoder contract.
- The README and work log still say that final review and checkpointing are
  pending. That is expected pre-review state, not a stage defect; the stage
  owner should update those administrative fields when recording this verdict
  and the local checkpoint SHA.

## Prior-Finding Closure

- Initial-review P1, mutable traced inputs: closed. The fused-only wrapper
  executes both representative layer kinds at random/non-block-aligned and
  1023-to-1024 boundary positions. `standard_suite_final.log` and the separate
  `TT_METAL_WATCHER=10` run both use the delivered test hash
  `29b99b8c...`; the watcher run passes all four cases. The inherited test
  changes token, uint32 RoPE position, and int32 cache-update position, checks
  against the second HF result and a stale-output negative control, and
  requires bitwise-identical repeated replay.
- Initial-review P1, long-prefill GELU: closed. Real-weight M4096 F4/F2/F1
  candidates all pass equivalence but regress the current 11.196 ms gate plus
  unary path by 66.5%-551.4%. Adapted C2048/C1024/C128 candidates normalize to
  18.725/18.993/20.618 ms per 4096 rows. F4 profiling proves the activation is
  genuinely fused (the standalone unary disappears) and also shows why it is
  slower. Retaining the faster correct B0 long path is therefore earned.
- Initial-review P2, stale/ambiguous provenance: closed. The final standard
  and watcher logs bind the delivered four source/test hashes; the earlier
  long and profiler logs bind the identical runtime source and explicitly
  identified pre-wrapper test hash. The final exact-context log binds the
  delivered fused source/test hashes. Canonical Tracy raw, filtered CSV, and
  text hashes reproduce `evidence_manifest.md`, and candidate reports are
  located beside their own raw inputs.
- Initial-review concern, noise-scale crop ordering: closed. The same-process
  paired experiment retains 12 complete device-timed samples per arm and six
  ABBA cycles. Post-projection crop wins the device median by 2.512 us and all
  six paired cycle means; both paths have PCC 1.0 and deterministic output.
  The delivered default source is the selected post-projection path.
- Rereview P1, advertised-context distinct-token decode: closed. The appended
  fused-only wrapper asserts exact `FusedDecoder` and `_FusedSharedMLP`
  construction, then runs the inherited 262144-context HF gate. The
  hash-bound `exact_context_distinct_262144_final.log` passes sliding/full at
  PCC 0.999380/0.998937, consumes a distinct replay token at absolute position
  262143, beats the wrong-position RMSE control for both layer kinds, and is
  bitwise deterministic across repeated trace replay.

## Contract Coverage

- Fused path: `tt/fused_decoder.py` exists. Direct tests assert the exact fused
  class and MLP type; inherited regression wrappers monkeypatch construction
  to `FusedDecoder`. The source audit forbids `torch`, `from_torch`,
  `to_torch`, and functional-forward fallback in all measured helpers.
- Semantics and meaningful layer kinds: real cached checkpoint weights cover
  sliding layer 0 and full layer 5. Seq 32/33/128 prefill, traced decode,
  batch 2/32, rotated page tables, mutable trace inputs, and repeated replay
  all pass the PCC >= 0.995 bar and determinism requirements.
- Paged cache and non-aligned lengths: seq 1025/1057 directly checks sliding
  circular-cache K/V ownership after padding and wrap. Seq 262113 passes both
  layer kinds. No public chunk-alignment restriction was introduced.
- Context contract: exact 262144 prefill, populated-cache decode parity, and
  the genuine 262143-history distinct-token traced decode all pass. Cache
  dtype, allocation, page size, and advertised capacity did not change, so
  leaving `doc/context_contract.json` unchanged is correct.
- Performance: like-for-like final warmed seq-128 prefill is
  3426.812/4192.270 us versus 3521.305/4254.306 us functional for
  sliding/full. Final traced warmed decode is 2560.197/2880.903 us versus
  2576.819/2911.473 us. The selected decode path also beats the other correct
  retained topology candidates; its small crop-placement choice is backed by
  the repeated paired experiment above.
- Topology: canonical `tt-perf-report` CSV/text exists for all four workloads.
  The measured windows report zero host ops and no generic copy, standalone
  unary, tilize, untilize, or generic reshard row. Full decode uses one
  `PagedFusedUpdateCacheDeviceOperation`; sliding correctly retains two
  modulo-aware cache updates because the fused op cannot express its cache
  contract. Remaining I2S/S2I movement is tied to sharded RMSNorm, cache
  writer, GQA SDPA output, and decode concat contracts; the direct sharded GQA
  output adaptation was attempted and rejected by the exact TTNN contract.
- Fusion exhaustion: the dedicated-op, structural-rewrite, and op-merge
  inventory covers activation, RMSNorm, SDPA, QKV/head transforms, RoPE,
  cache updates, shared-LHS packing, slice placement, residual scalar, and all
  inapplicable convolution/reduction/TopK patterns. Material legal candidates
  were PCC-checked and measured; retained rejections include exact API/L1
  blockers or measured regressions.
- Stress and watcher: the standard suite reports 23 passed and 9 explicit
  environment-gated skips; capacity gates pass separately at 262144 and
  262113. The final watcher run passes four mutable-trace cases, attaches,
  checks, and detaches cleanly, and its device log has no fatal, assert, NoC,
  overflow, sanitizer, or exception finding.
- Scope: staged paths are confined to `tt/fused_decoder.py`,
  `tests/test_fused_decoder.py`, and `doc/fused_decoder/**`. The unrelated
  deleted requirements file and untracked `.exp_run/` and `fusion_tests/` are
  not staged. No optimized-decoder, multichip, full-model, or vLLM work is in
  the staged diff.

## Evidence Provenance

- Delivered source hashes independently reproduce the manifest:
  `941dd1d1...` fused implementation, `29b99b8c...` fused tests,
  `2f8a26cb...` functional implementation, and `2fd1278e...` functional tests.
- Required-log hashes independently reproduce the manifest, including
  `a9f5fcc2...` standard, `c265d8aa...` exact-capacity,
  `cfd04e8e...` non-aligned capacity, `7b7db74f...` watcher console,
  `007556b5...` watcher device log, and `f9b28879...` distinct-token context.
- All twelve canonical Tracy raw/filtered/text hashes independently reproduce
  `evidence_manifest.md`. Their report totals and op counts agree with the
  README: 3426.812/26, 4192.270/23, 2560.197/40, and 2880.903/39.
- `git diff --cached --check` passes. The staged scope can be isolated from the
  unrelated dirty workspace for the required local checkpoint.

## Hard-Check Gaps

- Canonical headline performance uses one final signposted interval per
  workload. This is sufficient here because every final path beats its
  like-for-like functional baseline, raw and filtered reports agree, and the
  only noise-scale selection was separately resolved by 12-sample paired
  device timing.
- The long capacity logs predate the final test-only wrapper and therefore
  bind the prior fused-test hash. They bind the identical final runtime source;
  the wrapper has no runtime implementation delta, and its own exact-context
  hardware run plus the refreshed standard and watcher runs bind the delivered
  test hash. No capacity rerun gap remains.

## Anomaly Ledger

- Observed anomaly: padded sliding prefill originally overwrote live circular
  cache slots after window wrap.
  Evidence: the work log records seq 1025 decode PCC 0.994710 and a zero live K
  slot at seq 1057 before repair.
  Affected path: non-aligned sliding prefill followed by decode.
  Control or comparison: exact logical-tail writes now pass seq 1025/1057
  decode PCC 0.997649/0.999271 and direct K/V ownership PCC >= 0.999885.
  Likely subsystem: modulo cache-fill ownership for padded tail rows.
  Investigation performed: focused fused ownership/decode regression before
  and after restoring the inherited exact-tail helper.
  Resolution: fixed.

- Observed anomaly: long M4096 prefill retains a standalone GELU.
  Evidence: source branch plus the F4/F2/F1 and C2048/C1024/C128 logs.
  Affected path: prefill MLP chunks above 128 rows.
  Control or comparison: every legal fused candidate is correct, but normalized
  fused latency is 18.725 ms or worse versus 11.196 ms for B0; F4 raw profiling
  verifies that removing the unary still makes the matmul materially slower.
  Likely subsystem: explicit fused matmul program geometry.
  Investigation performed: real-weight PCC, warmed ABBA-style sampling,
  profiler topology, block-height adaptations, and chunk ladder.
  Resolution: controlled; the faster correct composition is retained.

- Observed anomaly: the first repeated crop profiler filled device-profiler
  buffers, and two-trace candidate runs emitted the active-trace allocation
  warning.
  Evidence: `profile_console.log` and the classified AutoFix ledger.
  Affected path: candidate timing evidence only, not the final canonical run.
  Control or comparison: the incomplete tail was discarded; the complete
  rerun retained 12 complete intervals per arm without buffer-full warning,
  all six device ABBA cycles agreed, PCC was 1.0, and both arms were
  deterministic. Final single-path correctness and Tracy runs also passed.
  Likely subsystem: candidate profiler buffering/trace allocator diagnostics.
  Investigation performed: bounded complete rerun plus independent final-path
  run.
  Resolution: controlled.

- Observed anomaly: runtime warns that firmware 19.9.0 is newer than the fully
  tested 19.5.0 bundle.
  Evidence: final test/profiler logs.
  Affected path: hardware environment.
  Control or comparison: all retained correctness/performance runs pass and
  the final watcher is clean.
  Likely subsystem: environment compatibility warning.
  Investigation performed: compared final logs and scanned the watcher device
  log for fault signatures.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied Stage 02 contract;
  `.agents/skills/graph-fusing/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: fused README, work log, manifest, both prior reviews,
  AUTODEBUG/AUTOFIX, final standard/long/context/watcher logs, canonical Tracy
  raw and reports, candidate reports, functional baselines, and context
  contract.
- Code paths: fused and functional decoder implementations and tests.
- Commands run: read-only `cat`, `sed`, `rg`, `find`, `sha256sum`, `git status`,
  `git diff --cached`, and `git diff --cached --check`. No device, reset,
  profiler, server, vLLM, or other hardware command was run by this review.

## Residual Risk

- The full-layer exact-context wrong-position negative-control margin is
  modest (RMSE 0.02017 correct versus 0.02247 wrong), but the correct oracle
  wins, PCC is 0.998937, the trace consumes the distinct token, and replay is
  deterministic. Together with short/batched/full-cache coverage this is not
  required work.
- The selected crop-placement win is small and canonical one-shot latency
  varies by a few microseconds, but repeated paired device timing consistently
  selects the delivered path and the final graph remains faster than the
  functional baseline. No unsupported stage-closure claim remains.
