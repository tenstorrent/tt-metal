# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The long-context prefill implementations have capacity evidence, but no correctness evidence.
  Evidence: `tt/functional_decoder.py:131-133` selects a separate sliding path above `PREFILL_SDPA_MAX_SEQ`, and `tt/functional_decoder.py:195-258` implements a separate full-attention streaming path that performs chunked QKV projection, RoPE, paged cache fill, chunked SDPA, output projection, and concatenation. The only tests that exercise advertised and near-advertised lengths, `tests/test_functional_decoder.py:422-443`, inspect one last-token sample for finiteness. None of the long logs contains a PCC result; all PCC-bearing evidence is from sequence lengths at or below 1025. In particular, `logs/prefill_advertised_262144_full_streaming.log`, `logs/prefill_near_context_262113_full.log`, `logs/prefill_advertised_262144_sliding.log`, and `logs/prefill_near_context_262113_sliding.log` only report a passing finite-value assertion.
  Why this matters: the stage-critical long branches are mathematically and operationally different from the short path that passed HF parity. A finite output cannot detect wrong causal offsets, missing history, incorrect page slices, chunk-order errors, or padded-tail contamination. The original contract requires HF-vs-TTNN prefill correctness at PCC >= 0.995 and advertised/non-aligned context support; the evidence currently proves capacity, not correctness, for the implementations that provide that support.
  Required next step: add real-weight HF parity tests that force each long-only attention branch at a tractable length (including a non-divisible tail and a sliding-window boundary), compare the full output or well-justified multiple samples at PCC >= 0.995, and retain the existing 262144/262113 capacity runs. The forced test must execute the same streaming/chunking helpers and page-table behavior used by the advertised-context run.

- P1: Paged-cache ownership/addressing and advertised-position decode are not validated with populated, genuinely permuted pages.
  Evidence: `_paged_state` constructs a claimed permutation as `roll(flip(ids), 3)` (`tests/test_functional_decoder.py:94-114`). All HF-parity decode tests allocate context 128 (`tests/test_functional_decoder.py:181` and `:314`), which is two blocks; for two elements, flip followed by a shift of three is the identity ordering. The advertised-context test allocates a fresh zero cache and immediately decodes at position 262143 (`tests/test_functional_decoder.py:378-417`); it never prefills history, never compares with HF, and asserts only finiteness. The boundary prefill tests can use non-identity tables at larger block counts, but their attention output is computed from local Q/K/V and therefore does not prove that later decode reads the filled physical pages correctly. `doc/context_contract.json` nevertheless records `largest_decode_context_tested: 262144`.
  Why this matters: the contract explicitly calls for page-table permutations, nonzero slots, random/current positions, full supported context, and cache/addressing correctness. The current evidence can pass if page-table lookup, block ownership, full-cache reads, or sliding modulo wrapping is wrong, because the PCC decode cases use identity tables and the non-identity advertised cases read an otherwise empty cache.
  Required next step: for both layer kinds, prefill a populated cache using a provably non-identity page table, then run traced decode and compare replay output with HF at PCC >= 0.995. Cover nonzero physical block IDs, at least two distinct users or ownership rows, page-boundary positions, and for sliding attention positions on both sides of the 1024-token circular wrap. Add a populated near/full-context decode control or revise `largest_decode_context_tested` so it does not represent an empty-cache absolute-position smoke as full-context correctness.

- P1: Batch-32 decode correctness was not demonstrated under the required traced execution path.
  Evidence: `test_batch_32_paged_decode_pcc` calls `decode_forward` eagerly and converts its direct output (`tests/test_functional_decoder.py:305-375`); it never captures or executes a trace. Trace replay and repeated-input determinism are tested only for batch 1 at prompt length 32 (`tests/test_functional_decoder.py:172-250`). `logs/batch32_prefill_decode_real.log` confirms the eager batch-32 PCC results, while the trace logs contain only the two batch-1 cases.
  Why this matters: batch decode uses a distinct per-user RoPE path, batched cache update, sharding, and head-concatenation behavior. Trace capture can introduce address, stale-input, and lifecycle failures not exposed by eager execution. The original goal combines decode trace completion with batch >1/up to 32 support.
  Required next step: capture and replay the complete batch-32 decode for both layer kinds, measure PCC from replay output, replay identical input twice for determinism, and demonstrate updating stable token/current-position buffers across at least one replay without allocating replacement trace inputs.

- P2: The final source revision is not tied to a complete passing evidence run.
  Evidence: `logs/final_standard_suite.log` reports the old test names `test_batch_two_nonaligned_paged_prefill` and 13 passed/6 skipped, while the current test file parameterizes `test_batched_nonaligned_paged_prefill` for batches 2 and 32 and statically contains 17 ordinary passing cases plus six gated cases. `logs/batch32_prefill_decode_real.log` closes only the four added batch-32 cases. Filesystem timestamps place both `tt/functional_decoder.py` and `tests/test_functional_decoder.py` after the standard, profiler, watcher, and batch-32 logs, and no source hash or commit is recorded in those artifacts.
  Why this matters: the evidence set demonstrably comes from more than one source/test revision, so it does not establish that the exact delivered implementation passes the complete suite, watcher run, and profiling path. This is a stale-artifact risk, not merely a presentation preference.
  Required next step: after resolving the correctness gaps, run the complete standard suite against the final files and record the exact commit or content hashes in the work log. Regenerate any watcher/performance evidence affected by runtime changes and ensure reported test counts/names match the final test file.

## Other Concerns

- `work_log.md` says the exact 196577-token triage output is in `triage/long_full_196577_tt-triage.txt`, but that file is zero bytes. The console failure is partially preserved in `logs/prefill_long_196577_full.log`, and later advertised-context passes control the original implementation failure, but the stated triage provenance is false and should be corrected.
- The fallback audit test inspects only four `FunctionalDecoder` methods (`tests/test_functional_decoder.py:501-512`), omitting `_streaming_full_prefill_attention`, `_chunked_full_attention`, and transitive runtime helpers. A whole-file scan confirms the autoport file itself contains no `torch`, `from_torch`, or `to_torch`, but the claim that the test covers every runtime helper is overstated.
- Watcher evidence is clean for the two short batch-1 traced decode cases: the raw watcher log has normal attach/dump/stack-summary/detach records and the suspicious-term scan found no fatal/assert/NOC/overflow/sanitizer result. It does not cover prefill, long streaming, or batch-32 paths.

## Hard-Check Gaps

- No real-weight PCC exists for the long-only sliding chunked or full streaming attention branches.
- No test proves cache read-after-write correctness with non-empty, non-identity page mapping at nontrivial positions.
- No batch >1 trace replay result exists.
- No final-revision identifier ties source, test logs, watcher evidence, and profiler evidence together.
- The runtime fallback audit is narrower than the runtime call graph and does not instrument a measured pass for host conversion calls.
- The historical tt-triage artifact named by the work log is empty.

## Anomaly Ledger

- Observed anomaly: advertised and near-advertised prefill runs pass only a finite-value assertion.
  Evidence: `tests/test_functional_decoder.py:422-443` and the four final long-prefill logs contain no PCC.
  Affected path: sliding chunked prefill and full streaming prefill.
  Control or comparison: real-weight HF parity passes only on short/non-aligned/boundary paths through sequence 1025.
  Likely subsystem: chunk offsets, causal/window masking, streaming paged fill/read, and logical padding.
  Investigation performed: mapped every PCC-bearing log to its exercised sequence length and inspected the branch conditions and long test assertion.
  Resolution: more-work-needed.

- Observed anomaly: page tables described as deliberately permuted are identity in every HF-parity decode case.
  Evidence: context 128 produces two blocks; `roll(flip([0,1]), 3)` equals `[0,1]`. The batch-32 helper applies the same identity transform independently to every two-block user range.
  Affected path: paged prefill-to-decode cache addressing for batch 1 and batch 32.
  Control or comparison: larger advertised-context page tables are non-identity, but those decode tests start from an empty cache and check only finiteness.
  Likely subsystem: page-table test construction and cache ownership/address verification.
  Investigation performed: rederived the page-table transform for each tested block count and traced which attention tests actually read cache contents.
  Resolution: more-work-needed.

- Observed anomaly: `largest_decode_context_tested` is 262144 although the corresponding run has no populated 262144-token history.
  Evidence: `doc/context_contract.json`; `tests/test_functional_decoder.py:378-417`.
  Affected path: advertised full-context and sliding absolute-position decode claims.
  Control or comparison: populated-cache HF PCC exists only at prompt length 32.
  Likely subsystem: capability-evidence classification rather than a proven implementation fault.
  Investigation performed: inspected cache construction, test order, assertions, and logs.
  Resolution: more-work-needed.

- Observed anomaly: the standard-suite artifact does not match the current test inventory.
  Evidence: old test names and 13-pass count in `logs/final_standard_suite.log`; current parameterization in `tests/test_functional_decoder.py:276-306`; source/test timestamps postdate all runtime evidence.
  Affected path: final revision provenance.
  Control or comparison: the later batch-32 log passes four focused cases but is not a complete suite.
  Likely subsystem: evidence freshness.
  Investigation performed: compared collected test definitions, log node IDs/counts, and file timestamps.
  Resolution: more-work-needed.

- Observed anomaly: a historical 196577-token full-prefill run stalled NOC0, and its named triage report is empty.
  Evidence: `logs/prefill_long_196577_full.log`, zero-byte `triage/long_full_196577_tt-triage.txt`, and the recovery record in `work_log.md`.
  Affected path: legacy long full attention.
  Control or comparison: redesigned full streaming later passed 262144 and 262113 capacity smokes; device recovery is documented in prose.
  Likely subsystem: legacy long SDPA/kernel duration and evidence preservation.
  Investigation performed: read the failure log, work-log recovery account, triage artifact size, and later pass logs.
  Resolution: controlled for the implementation failure; evidence/documentation correction remains.

- Observed anomaly: profiler console logs warn that several attention/cache operations are unclassified.
  Evidence: all four `*_perf_report.console.log` files name unclassified rotary, paged-fill, QKV-head, or SDPA operations.
  Affected path: performance categorization only.
  Control or comparison: filtered CSV rows and human-readable reports retain the operations and signpost filtering; recomputed `Device Time` sums exactly match README totals (3532.537 us, 4247.842 us, 2578.397 us, and 2912.882 us).
  Likely subsystem: `tt-perf-report` categorization metadata.
  Investigation performed: checked raw signposts, filtered row ranges, report text, console provenance, and independently summed CSV values.
  Resolution: controlled; no correctness or headline-latency contradiction found.

## Scope Inspected

- Goal/skill paths: the supplied functional-decoder goal contract; `.agents/skills/functional-decoder/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/context_contract.json`; `doc/functional_decoder/README.md`; `work_log.md`; `real_weight_stats.json`; every file under `logs/`, `tracy/`, `watcher/generated/watcher/`, and `triage/` relevant to stated claims.
- Code paths: `tt/functional_decoder.py`; `tests/test_functional_decoder.py`; the called Gemma4 decoder/attention implementation under `models/demos/gemma4/tt/attention/` and `models/demos/gemma4/tt/layer.py`.
- Commands run: read-only `cat`, `sed`, `nl`, `find`, `rg`, `wc`, `stat`, `git status`, and `git diff`; a read-only Python CSV sum; and `pytest --collect-only`, which stopped at import because the checkout library path was not set and did not open a TT device. No server, TT device, reset, reservation, hardware test, or profiler run was started.

## Residual Risk

- Short-context real-weight PCC is strong for both layer kinds, including non-aligned inputs, page/window boundaries, and eager batch 32. Trace replay PCC and determinism are strong for batch 1. These results substantially reduce risk in the common short path but do not close the long-path, paged-addressing, or batched-trace gaps above.
- The performance artifacts are internally consistent and human-readable, and their reported totals rederive exactly from filtered CSVs. Their remaining risk is evidence freshness relative to the final uncommitted source, not arithmetic or signpost selection.
- Static inspection found no direct host conversion in `functional_decoder.py` and no scope leakage into optimized, multichip, full-model, generator, or vLLM implementation. Transitive runtime fallback remains less completely evidenced than claimed.
