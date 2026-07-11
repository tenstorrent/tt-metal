# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The advertised-context "populated decode" control overwrites an already-prefilled token and reports TT-vs-TT self-parity, not a real final-position HF-vs-TT decode.
  Evidence: `tests/test_functional_decoder.py:478-486` prefills all `seq_len` inputs, including `hidden[:, -1, :]`, into the paged cache. The decode token is then constructed from that same last hidden state (`:500-507`) and written again at the same already-populated absolute position `seq_len - 1` (`:508-525`). The only decode assertion is `_assert_pcc(_to_host(last_output), _to_host(decode_output))` (`:535-536`). Thus the second PCC values in the 262144 logs (0.998715 sliding, 0.996611 full) are TT prefill-output versus TT replay-output. They are not HF-vs-TT decode PCC, and a no-op or misaddressed final cache update can pass because the correct K/V for that identical token and position was already filled. The full-attention HF control checks only prefill outputs 0..2048 (`:489-493`), not the final-position decode output. The README/work log accurately say the decode matched the prefill token, but `context_contract.json` and `evidence_manifest.json` elevate this to `largest_populated_decode_context_tested: 262144`/advertised populated-decode evidence. The skill also calls for random current positions; all direct HF-vs-TT decode controls use fixed position 32, and the batch-32 trace recopies identical token/position contents before its second replay (`:407-415`) rather than proving a changed position is consumed.
  Why this matters: full-supported-context decode, paged update addressing, circular sliding-cache wrap, and trace-safe mutable current positions are stage-critical contracts. Short position-32 HF-vs-TT replay proves the ordinary decode math, and the current long test proves a populated cache can be read at the maximum absolute position, but their combination does not prove a genuine 262144th-token decode or that replay consumes changed position buffers. This is required work rather than residual risk because the original goal explicitly requires longest-context decode plus current-position handling, and the functional-decoder skill explicitly requires full-context decode, random current positions, and PCC from replay.
  Required next step: prefill exactly 262143 preceding tokens, decode a distinct final token at absolute position 262143 through captured replay, and compare the replay result directly with an HF cache/reference for both layer kinds (a justified reduced HF oracle is acceptable if it preserves the final-position history semantics). Retain a provably nonidentity page table. Add at least one traced replay test that changes stable token and current-position buffer contents to a non-boundary/random legal position and checks the changed replay against HF; include sliding positions across the 1024-token modulo wrap and a page transition. Emit the tested logical length/position in the resulting logs.

## Other Concerns

- The exact 262144 and 262113 commands are recoverable from the experiment session JSONL (`.exp_run/codex_home/sessions/2026/07/11/rollout-2026-07-11T05-06-52-019f4f92-4126-7161-a8bd-751ec2c04107.jsonl`), so the advertised prefill runs are not rejected as fabricated or mislabeled. However, the committed pytest logs themselves do not print `GEMMA4_LONG_PREFILL`, `seq_len`, the command, or a source hash; compact future evidence should print those values rather than depending on an untracked session transcript and filenames.
- The final watcher run covers traced batch-32 decode for both layer kinds and is clean, but it does not cover prefill or either long streaming branch. This is a coverage limitation, not independently required work: the goal requires a watcher-clean run, not watcher coverage of every evidence command, and no concrete watcher-sensitive anomaly remains on the delivered paths.

## Hard-Check Gaps

- No replay test changes a captured current-position buffer to a different legal value and validates the resulting output; identical values are merely copied back into the same allocations.
- No direct HF-vs-TT decode PCC exists near the advertised context. The exact-context decode PCC rows are TT-vs-TT prefill/replay comparisons.
- The committed long-run logs omit the tested `seq_len` and exact invocation; the command-to-file association is presently recoverable only from the untracked experiment session transcript.
- Performance provenance establishes passing selected nodes, signposts, raw/filtered CSVs, rendered tables, and internally reproducible `Device Time` totals, but the documentation gives placeholder profiler/report commands rather than preserving every exact CLI argument.

## Anomaly Ledger

- Observed anomaly: exact-context decode reuses the final prefill token at the same already-filled cache position.
  Evidence: `tests/test_functional_decoder.py:478-486`, `:500-536`; the second PCC in each `final_long_pcc_populated_decode_262144_*.log`.
  Affected path: maximum-position paged decode, paged cache update, trace replay, and sliding modulo addressing.
  Control or comparison: direct HF-vs-TT traced decode passes at position 32 for batch 1 and batch 32 with rotated physical page mappings; exact-context prefill has above-threshold reduced HF controls.
  Likely subsystem: evidence/test semantics rather than a proven implementation defect.
  Investigation performed: traced the input token, prefill extent, cache state, current-position tensors, trace capture/replay, and both operands to every reported PCC.
  Resolution: more-work-needed.

- Observed anomaly: the batch-32 "production trace contract" copies identical host contents into captured buffers before replay.
  Evidence: `tests/test_functional_decoder.py:368-415`; `pos_u_values`, `pos_i_values`, and `tt_token_host` are never changed before `copy_host_to_device_tensor`.
  Affected path: mutable trace inputs and current-position handling.
  Control or comparison: static code passes device position tensors to on-device embedding and paged cache ops, and unchanged repeated replay is bitwise deterministic.
  Likely subsystem: trace/current-position test coverage.
  Investigation performed: inspected captured allocations, host/device copies, replay order, and reference calculation.
  Resolution: more-work-needed.

- Observed anomaly: the old 196577-token full-prefill path stalled NOC0 and triage could not attach.
  Evidence: `logs/prefill_long_196577_full.log` and `triage/long_full_196577_tt-triage.txt`.
  Affected path: superseded non-streaming long full attention.
  Control or comparison: replacement streaming code subsequently completed 262144 and 262113 with above-threshold reduced HF prefill controls; the recovery record includes bounded reset/list and a successful 1x1 mesh smoke.
  Likely subsystem: legacy long SDPA/kernel duration followed by unavailable NOC0.
  Investigation performed: read the failure, triage, recovery record, replacement branch, and later pass logs.
  Resolution: controlled.

- Observed anomaly: `tt-perf-report` reports unclassified Gemma rotary/cache/SDPA operations.
  Evidence: all four `tracy/*/*/*_perf_report.console.log` files.
  Affected path: report categorization.
  Control or comparison: raw signposts and filtered CSVs retain the operations; CSV `Device Time` sums reproduce 3532.537, 4247.842, 2578.397, and 2912.882 microseconds, and the rendered reports show zero host operations.
  Likely subsystem: report taxonomy metadata.
  Investigation performed: checked raw/filtered/report artifacts, signpost provenance, units, hashes, and totals.
  Resolution: controlled.

- Observed anomaly: runtime logs warn firmware 19.9.0 is newer than the latest fully tested 19.5.0 bundle.
  Evidence: standard, long, profiler, and watcher console logs.
  Affected path: runtime environment.
  Control or comparison: final correctness/profiling commands pass and the separate watcher run detaches cleanly without assert, invalid NOC, overflow, stack, or sanitizer findings.
  Likely subsystem: environment compatibility warning.
  Investigation performed: compared the warning across runs and scanned the final watcher log and console.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: `.exp_run/multigoal_logs/01-01-functional-decoder.prompt.txt`; `.agents/skills/functional-decoder/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/context_contract.json`; every summary/manifest/stat/review file in `doc/functional_decoder/`; exact/near-context, standard-suite, watcher, profiler, and recovery logs; all four Tracy raw/filtered/rendered groups; both watcher evidence roots; the relevant experiment session JSONL command transcript.
- Code paths: `tt/functional_decoder.py`; `tests/test_functional_decoder.py`; imported Gemma attention/decode, paged-cache, RMSNorm, MLP, and layer helpers used by the fallback and trace call graphs.
- Commands run: read-only `cat`, `sed`, `nl`, `find`, `rg`, `wc`, `stat`, `sha256sum`, `gzip`, `git status`, `git log`, `git show`, `git diff`, `git ls-files`, and `git check-ignore`; no TT device, server, reset, profiler, pytest hardware run, or implementation/test mutation.

## Residual Risk

- Real target weights and shapes are well evidenced for both layer kinds. Short aligned/nonaligned/boundary prefill, direct HF-vs-TT traced decode, batch-32 prefill/decode, deterministic identical replay, nonidentity/disjoint physical page mappings, and exact/near-context prefill capacity all pass above PCC 0.995.
- The exact full-attention prefill HF oracle is intentionally reduced to the first 2049 outputs. That is acceptable under the skill's reduced long layer-harness allowance because it forces the exact streaming branch across multiple chunks and is paired with a full-capacity run; it does not, however, convert the final TT-vs-TT decode comparison into HF-vs-TT evidence.
- Static runtime-call-graph inspection plus the signposted device-only profiler windows support the no-host-fallback claim. A dynamic interposition audit would be stronger but is not required absent contrary evidence.
- Source/test and performance CSV hashes match `evidence_manifest.json`; the final standard suite postdates the import-only runtime formatting change and reports 17 passed/6 explicitly gated skips. HEAD `82ce4e431f29c36946f6fd96391988cfa68ad3c2` changes only review/work-log documentation relative to checkpoint `dac92a78dbca5d4b2d3e85b1007b00064b1ccc42`. Untracked fused-decoder files are later-stage scope and were not inspected as stage-01 edits or modified.
