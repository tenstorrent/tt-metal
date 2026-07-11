# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Findings

- No P1 or P2 finding remains. The committed Stage 02 implementation, exact fused-path tests, and persisted evidence satisfy the supplied fused-decoder goal contract.
- The four delivered source/test SHA-256 values reproduce `evidence_manifest.md`. The six required final gate artifacts also reproduce their manifest hashes. The current fused and functional implementations and tests are unchanged from the fused checkpoint; later commits alter only administrative functional/fused documentation in this autoport.
- Independent CSV parsing reproduces all canonical performance claims: sliding/full warmed prefill are 3426.812/4192.270 us over 26/23 ops, and sliding/full traced warmed decode are 2560.197/2880.903 us over 40/39 ops. The like-for-like functional reports reproduce 3521.305/4254.306 us and 2576.819/2911.473 us. All four selected paths therefore beat their baselines, and final decode also beats the retained correct decode candidates.
- All twelve canonical raw/filtered/text profiler hashes reproduce. The filtered selected windows contain zero host rows and no `CopyDeviceOperation`, standalone `UnaryDeviceOperation`, tilize, untilize, or generic reshard operation. The remaining explicit interleaved/sharded transitions agree with the documented RMSNorm, cache-writer, GQA-SDPA-output, and decode-head-concat contracts.
- Correctness evidence exercises exact `FusedDecoder` construction and `_FusedSharedMLP`, representative sliding layer 0 and full layer 5, real checkpoint weights, rotated page tables, aligned and non-aligned prefill, batch 2/32, mutable traced token/position/cache-index buffers, deterministic replay, sliding-cache wrap ownership, exact 262144 context, and non-aligned 262113 context. Every reported real-weight PCC exceeds the 0.995 functional bar.
- Candidate closure is earned. The retained evidence measures or records exact blockers for dedicated head concat, actual config-level GELU fusion, packed gate/up, crop placement, sharded GQA SDPA output, fused circular-cache update, padded sliding tails, and long-prefill GELU geometries/chunk families. The source/topology inventory also classifies the inapplicable convolution, bias, TopK, reduction, pre-add norm, and incompatible RoPE patterns.
- Stage scope is isolatable and locally checkpointed at `ce88390ebcceb9e8d83af37ed8a166406e360370`, with administrative checkpoint `bea8302a6ac9219922f91f7c057bb9011d2932e7`. The live optimized-decoder files and live `context_contract.json` additions are later-stage dirty work and are not present in the committed fused-stage contract. The unrelated deleted requirements file, `.exp_run/`, and `fusion_tests/` are likewise outside Stage 02.

## Other Concerns

- None.

## Hard-Check Gaps

- Canonical headline latency is one final signposted interval per workload. This is adequate for this gate because the raw and filtered reports agree, every selected path beats the like-for-like functional baseline, and the only noise-scale topology choice was separately decided by 12 samples per arm and six complete device-timed ABBA cycles.
- The capacity and canonical profiler logs bind the final runtime source but the immediately preceding fused-test hash, before the exact-context wrapper was appended. The delivered manifest identifies that hash and delta; the wrapper is test-only, while the delivered test hash is bound by the refreshed standard, watcher, and exact-context runs. Supplemental current verification also reran the delivered complete suite successfully (23 passed, 9 skipped). This is not stale-runtime evidence.

## Anomaly Ledger

- Observed anomaly: padded sliding prefill originally overwrote live circular-cache slots after the 1024-token window wrapped.
  Evidence: the retained work log records pre-fix seq-1025 decode PCC 0.994710 and a zero live K slot at seq 1057; `standard_suite_final.log` records repaired decode PCC 0.997649/0.999271 and direct K/V ownership PCC 0.999885-0.999911.
  Affected path: non-aligned sliding prefill followed by decode.
  Control or comparison: exact logical-tail cache writes versus the rejected rounded-tail implementation.
  Likely subsystem: modulo cache-fill ownership of padded tail rows.
  Investigation performed: focused before/after decode and direct cache-slot ownership regressions at 1025 and 1057.
  Resolution: fixed.

- Observed anomaly: long M4096 prefill retains a standalone GELU rather than fused matmul activation.
  Evidence: the F4/F2/F1 real-M4096 candidates pass PCC but take 18.636/36.824/72.887 ms versus about 11.196 ms for the retained gate-plus-GELU path; C2048/C1024/C128 normalize to 18.725/18.993/20.618 ms per 4096 rows.
  Affected path: long-prefill MLP gate activation.
  Control or comparison: retained B0 composition and profiler-confirmed F4 fused activation.
  Likely subsystem: fused 1D matmul program geometry.
  Investigation performed: real-weight admission/PCC, F4/F2/F1 block-height ladder, chunk-size ladder, warmed samples, and F4 topology profiling.
  Resolution: controlled; the faster correct unfused composition is retained.

- Observed anomaly: the original crop-placement ordering was below the noise scale, and an initial repeated profiler run filled its device-profiler buffer.
  Evidence: `candidates/post_projection_slice_repeated/summary.md` discards the incomplete run and records a complete 12-sample-per-arm rerun; all six device ABBA-cycle means favor post-projection crop by 2.015-3.856 us, with PCC 1.0 and deterministic replay.
  Affected path: sliding traced decode candidate selection only.
  Control or comparison: same-process pre- versus post-projection crop traces.
  Likely subsystem: candidate timing/profiler buffering, not decoder correctness.
  Investigation performed: complete bounded rerun, paired device timing, ordinary timing, and final-path correctness/profile checks.
  Resolution: controlled; incomplete evidence is excluded and the repeated winner is the delivered default.

- Observed anomaly: runtime logs warn that firmware 19.9.0 is newer than the fully tested 19.5.0 bundle.
  Evidence: final gate/profiler logs; `watcher_final/generated/watcher/watcher.log` has normal attach/check/detach records and no fatal, assert, invalid NoC, overflow, sanitizer, or exception finding.
  Affected path: hardware environment.
  Control or comparison: all retained correctness/performance checks pass and the separate watcher run is clean.
  Likely subsystem: environment compatibility warning.
  Investigation performed: cross-log warning scan and watcher-device-log scan.
  Resolution: controlled.

- Observed anomaly: the live `context_contract.json` contains optimized-decoder BFP8-cache fields absent from the fused checkpoint.
  Evidence: `git show ce88390e:.../context_contract.json` retains the 262144 functional/fused contract without those fields; the live diff adds only optimized-stage metadata/evidence links.
  Affected path: live later-stage documentation, not Stage 02 source or evidence.
  Control or comparison: committed fused-stage context contract and `ce88390e..HEAD` source/test diff.
  Likely subsystem: concurrent optimized-decoder worktree state.
  Investigation performed: committed-blob inspection and scoped Git diffs.
  Resolution: controlled; explicitly out of fused-stage scope.

## Scope Inspected

- Goal/skill paths: supplied Stage 02 contract; `.agents/skills/graph-fusing/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/autofix/SKILL.md`.
- Artifact paths: fused README, work log, evidence manifest, prior reviews, AUTODEBUG/AUTOFIX, standard/capacity/distinct-context/watcher logs, watcher device log, all canonical Tracy raw/filtered/text reports and command logs, functional Tracy baselines, long-GELU evidence, crop A/B evidence, other retained candidates, and committed/live context contracts.
- Code paths: fused and functional decoder implementations and tests, including inherited context, mutable-trace, cache-ownership, batched, and performance regressions.
- Commit/scope paths: fused checkpoint `ce88390e`, administrative checkpoint `bea8302a`, `ce88390e..HEAD` scoped diffs, current worktree status, and stage-only commit contents.
- Commands run: read-only `sed`, `rg`, `find`, `wc`, `sha256sum`, CSV analysis, and `git status/log/show/diff/diff --check`. No TT device, reset, profiler, server, vLLM, or hardware test command was run by this review.

## Residual Risk

- The performance wins versus the functional baseline are modest (0.65%-2.68%) and canonical headline reports are single intervals. They are nevertheless consistently positive across all four required workloads, while the only candidate decision at microsecond scale has repeated paired evidence.
- The full-attention exact-context wrong-position RMSE separation is modest (0.02017 correct versus 0.02247 wrong), but the correct oracle wins, PCC is 0.998937, the replay consumes a distinct token at absolute position 262143, and repeated output is bitwise identical. Combined with the short, batched, populated-cache, and non-aligned coverage, this is not required work.
