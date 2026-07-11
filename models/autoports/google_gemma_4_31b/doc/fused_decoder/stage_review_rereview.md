# Stage Review Rereview

Verdict: more-work-needed

## Required Work

- P1: Run the genuine advertised-context distinct-token decode gate through
  `FusedDecoder` for both layer kinds.
  Evidence: the functional stage's context-defining test is
  `test_exact_context_distinct_traced_decode` at
  `tests/test_functional_decoder.py:908`. It prefills 262143 tokens, captures
  decode with a sentinel, copies a distinct final token into the captured
  allocation, replays at position 262143, compares against the periodic HF
  one-query oracle, checks a wrong-position negative control, and requires
  deterministic replay. `tests/test_fused_decoder.py` wraps the short mutable
  trace test, the finite advertised-position smoke, and the long prefill plus
  same-token prefill/decode-parity test, but it never wraps this exact-context
  distinct-token HF gate. This omission matters because
  `FusedDecoder._decode_attention` changes the cache write itself: full
  attention replaces the functional two-update sequence with
  `paged_fused_update_cache` on disjoint sharded grids, while sliding attention
  changes the layouts presented to its two modulo-aware updates.
  Why this matters: the declared inherited context contract includes genuine
  decode with 262143 populated history tokens, not only capacity and absolute
  position. The retained 262144 test decodes the already-prefilled last token
  at the already-populated position and compares TT decode with TT prefill
  output (`test_functional_decoder.py:862-900`); it can therefore miss a stale,
  no-op, or incorrectly addressed final cache update. Short mutable replay and
  exact-capacity same-token parity are strong component controls, but they do
  not prove their composition on the changed fused cache-update graph at the
  advertised limit.
  Required next step: add the same thin monkeypatch wrapper used by the other
  inherited fused tests for `test_exact_context_distinct_traced_decode`; run
  `GEMMA4_LONG_DECODE=262144` for sliding layer 0 and full layer 5 against the
  final source/test hashes; retain PCC, wrong-position RMSE, repeated
  determinism, command, and hash-bound logs. If both pass at the functional
  acceptance bar, refresh the fused README/work log/manifest and rereview.

## Other Concerns

- None requiring separate stage work. The exact-context same-token parity is
  0.998679 sliding and 0.996631 full; the latter is close to the functional
  stage's same-path 0.996611 result, so it is not an unexplained fused PCC
  regression. It simply does not replace the missing distinct-token HF gate.

## Hard-Check Gaps

- The final performance evidence is one canonical signposted interval per
  workload. This is adequate for the reported before/after result because raw
  and filtered artifacts agree, all four fused paths beat their like-for-like
  functional baselines, and the only noise-scale candidate decision was
  separately resolved with 12 samples per arm and six paired ABBA cycles.
- No dynamic host-fallback interposition test exists. The fused source audit
  covers the local and inherited runtime helpers used by the path, and all four
  final profiler windows report zero host operations, so this is not required
  work.

## Anomaly Ledger

- Observed anomaly: the exact-context fused test reuses the token and position
  already written by prefill.
  Evidence: `tests/test_functional_decoder.py:862-900`, invoked through
  `test_fused_long_nonaligned_prefill_capacity`; no fused wrapper invokes
  `test_exact_context_distinct_traced_decode` at lines 908-1020.
  Affected path: fused traced decode at the advertised 262144-token context,
  including full `paged_fused_update_cache` and sliding modulo cache update.
  Control or comparison: both fused layer kinds pass short changed-buffer
  replay with stale-output negative controls under watcher, and pass exact
  capacity same-token parity; the functional decoder separately passes the
  genuine exact-context HF gate.
  Likely subsystem: fused cache-update addressing/layout at the final physical
  page and its trace binding.
  Investigation performed: inspected the live fused and functional tests,
  context contract, decode implementation, hash-bound standard/long/watcher
  logs, and functional-stage rereview evidence.
  Resolution: more-work-needed.

- Observed anomaly: long prefill retains a standalone GELU for M=4096.
  Evidence: `_FusedSharedMLP.__call__` and
  `candidates/long_gelu/{F4,F2,F1,C2048,C1024,C128}`.
  Affected path: prefill chunks larger than 128 rows.
  Control or comparison: every admitted real-weight fused candidate passed PCC
  but normalized to 18.725-20.618 ms or worse versus the 11.196 ms current
  M4096 baseline; F4 profiling proves fusion removed the unary but increased
  gate latency to 18.480 ms.
  Likely subsystem: explicit fused matmul program geometry.
  Investigation performed: checked candidate logs, source gating, op reports,
  adapted block heights, and chunk ladder.
  Resolution: controlled; the faster correct unfused long-GELU composition is
  appropriately retained.

- Observed anomaly: firmware 19.9.0 is newer than the fully tested 19.5.0
  bundle named by runtime warnings.
  Evidence: final standard, long, watcher, and profiler logs.
  Affected path: hardware environment.
  Control or comparison: every retained correctness run passed and the final
  watcher attached, checked, and detached without a fatal/assert/NOC/overflow/
  sanitizer finding.
  Likely subsystem: environment compatibility warning.
  Investigation performed: compared warnings and scanned watcher output.
  Resolution: controlled.

## Contract Coverage

- Fused path enforcement: direct `FusedDecoder` construction, exact type
  assertions, `_FusedSharedMLP` assertion, source fallback audit, and profiler
  evidence.
- Correctness: real-weight aligned/non-aligned prefill and traced decode pass
  PCC >= 0.995 for sliding/full layers; batch 2/32, page rotations, wrapped
  sliding ownership, mutable trace buffers, and determinism are covered.
- Capacity: 262144 and non-aligned 262113 prefill plus populated-cache parity
  pass for both layer kinds without changing `context_contract.json`.
- Performance/topology: final warmed seq-128 prefill is 3426.812/4192.270 us
  versus 3521.305/4254.306 us functional; final traced decode is
  2560.197/2880.903 us versus 2576.819/2911.473 us. Final reports contain zero
  host ops and no generic copy, standalone unary, tilize, untilize, or reshard
  row. Full decode uses one fused K/V update; sliding retains two updates for
  required modulo semantics.
- Fusion search: the dedicated-op, structural-rewrite, and op-merge inventory
  is classified; admitted/rejected head concat, activation, cache update,
  packed gate/up, slice placement, residual scalar, RoPE, RMSNorm, SDPA, and
  long-GELU options have retained evidence.
- Scope: staged changes are confined to `tt/fused_decoder.py`,
  `tests/test_fused_decoder.py`, and `doc/fused_decoder/**`; unrelated dirty and
  untracked workspace state is not staged. No later pipeline stage appears in
  the diff.

## Scope Inspected

- Goal/skill paths: supplied Stage 02 contract;
  `.agents/skills/graph-fusing/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: fused `README.md`, `work_log.md`, `evidence_manifest.md`,
  initial `stage_review.md`, `AUTODEBUG.md`, `AUTOFIX.md`, standard/long/watcher
  logs, all four final Tracy raw/filtered/rendered groups, candidate reports,
  functional baselines, and `doc/context_contract.json`.
- Code paths: fused and functional decoder source and tests.
- Commands run: read-only `sed`, `nl`, `rg`, `find`, `sha256sum`, `git status`,
  `git diff --cached`, and `git diff --cached --check`. No TT device, reset,
  server, profiler, vLLM, or hardware experiment was started.

## Residual Risk

- All three findings from the initial review are convincingly closed and the
  delivered fused graph is faster than the functional baselines. Closure is
  withheld only because the fused path's changed cache-update graph lacks the
  functional stage's genuine advertised-context distinct-token HF gate.
