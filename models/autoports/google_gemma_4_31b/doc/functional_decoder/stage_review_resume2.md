# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The live worktree has an unstaged `doc/context_contract.json` edit, but its
  functional-decoder fields are byte-for-byte equivalent to checkpoint
  `5fa49e9fa25` after removing three appended optimized-decoder keys. The two
  new optimized evidence paths exist. This is later-stage state, not a
  functional-decoder regression or an unisolated stage-01 change.
- The historical `prefill_near_context_262113_{sliding,full}.log` files only
  record a passing capacity node and predate the final source hash. They are
  weak standalone PCC provenance. They do not create required work because the
  final-source exact-context tests exercise an even longer non-aligned prefill
  of 262143 tokens for both layer kinds, then validate a distinct traced token
  at position 262143 directly against the validated HF oracle. Exact aligned
  262144 prefill PCC is separately recorded, and the final cache-tail fix does
  not alter the aligned fill path.

## Hard-Check Gaps

- `evidence_manifest.json` hashes the delivered runtime, test, and four
  filtered performance CSVs, but not the correctness logs, watcher log, raw
  Tracy CSVs, or human-readable reports. Direct inspection found all referenced
  paths and internally consistent pass results, signposts, tables, and totals;
  this is an evidence-hardening opportunity, not a missing contract item.
- The final ordinary suite intentionally skips eight environment-gated long and
  performance nodes. Separate final-source logs cover exact-context distinct
  decode, changed trace buffers, watcher, and all four performance nodes. The
  aligned exact-context prefill logs are older, but their affected aligned path
  is unchanged by the final sliding-tail repair.

## Anomaly Ledger

- Observed anomaly: later optimized-decoder work appended keys to the shared
  context contract while the file remains unstaged.
  Evidence: `git diff` shows only
  `optimized_decoder_kv_cache_dtype` and two optimized evidence entries;
  deleting those keys reproduces the functional checkpoint JSON exactly.
  Affected path: shared documentation only.
  Control or comparison: runtime/test hashes match checkpoint `5fa49e9fa25`;
  both optimized evidence paths exist; no functional key changed.
  Likely subsystem: later-stage metadata accumulation.
  Investigation performed: compared the current JSON with the checkpoint,
  checked referenced paths, and inspected relevant git history/status.
  Resolution: controlled.

- Observed anomaly: the resumed review had shown that the original maximum
  context decode repeated an already-prefilled token and compared TTNN with
  TTNN rather than HF.
  Evidence: `stage_review_resume.md` and `AUTODEBUG.md`.
  Affected path: advertised-context paged decode and trace input mutation.
  Control or comparison: final-source
  `autofix_exact_context_decode_262144_{sliding,full}.log` prefills 262143
  history tokens, replaces a stable captured sentinel with a distinct final
  token, replays at absolute position 262143, and compares against an HF
  one-query oracle independently validated against stock HF at PCC
  0.99999997/0.99999999. Final PCC is 0.999406/0.998875, repeated replay is
  identical, and wrong-position RMSE is worse for both kinds.
  Likely subsystem: superseded test semantics.
  Investigation performed: inspected the current test flow, HF oracle,
  nonidentity page mapping, exact-context logs, and sensitivity controls.
  Resolution: fixed.

- Observed anomaly: padded lanes in non-aligned bounded sliding prefill could
  wrap modulo 1024 and overwrite live cache entries.
  Evidence: `AUTODEBUG.md` and the pre-fix runtime diff.
  Affected path: sliding paged-cache ownership after non-aligned prefill.
  Control or comparison: the delivered helper bulk-fills only valid complete
  tiles and updates valid tail tokens individually on device. Final-source
  seq-1025/1057 controls produce distinct HF-vs-TTNN decode PCC of 0.997702 and
  0.999330 and K/V ownership PCC of at least 0.999885. The 262143-token exact
  context run also traverses this repaired tail path.
  Likely subsystem: modulo cache fill of physical padding.
  Investigation performed: inspected bulk/tail bounds, update tensors,
  page-table slicing, source diff, focused logs, and exact-context coverage.
  Resolution: fixed.

- Observed anomaly: traced decode could have baked stale token or position
  contents.
  Evidence: the superseded batch-32 determinism control recopied identical
  values.
  Affected path: token input, RoPE position, and paged-cache position during
  trace replay.
  Control or comparison: final-source random and 1023-to-1024 tests mutate the
  same stable token, uint32 RoPE-position, and int32 cache-position allocations
  for both layer kinds. All direct PCC values exceed 0.998948, stale-output
  RMSE is materially worse, and repeated changed replay is bitwise identical.
  Likely subsystem: superseded coverage, not retained runtime behavior.
  Investigation performed: followed capture/copy/replay order in the test and
  inspected focused and watcher console evidence.
  Resolution: fixed.

- Observed anomaly: watcher and profiler consoles contain alarming words or
  warnings.
  Evidence: watcher compiler flags include `exception`; Tracy reports that an
  optional host `.tracy` copy was unavailable.
  Affected path: evidence classification and optional host trace convenience.
  Control or comparison: the generated 1987-line watcher log has normal
  lifecycle records and no fatal/assert/NOC/overflow/sanitizer finding; all
  four changed-input cases pass. Every profiler node passes, raw CSVs contain
  matching start/end signposts, filtered hashes match the manifest, and the
  rendered tables report 0 host ops.
  Likely subsystem: benign build flags and optional profiler artifact copying.
  Investigation performed: scanned the generated watcher log and inspected all
  profiler consoles, raw signposts, filtered CSV hashes, and human tables.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: `.exp_run/multigoal_logs/01-01-functional-decoder.prompt.txt`;
  `.agents/skills/functional-decoder/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/context_contract.json`; `doc/functional_decoder/README.md`;
  `work_log.md`; `evidence_manifest.json`; `AUTODEBUG.md`; `AUTOFIX.md`;
  prior review files; standard/focused/boundary/batch/exact-context/watcher/
  profiler logs; generated watcher evidence; and every canonical Tracy raw,
  filtered CSV, console, and human-readable table.
- Code paths: `tt/functional_decoder.py` and
  `tests/test_functional_decoder.py`. Current optimized/fused decoder files
  were recognized as later-stage scope and were not treated as stage-01
  deliverables.
- Commands run: read-only `sed`, `rg`, `find`, `wc`, `head`, `tail`, `stat`,
  `sha256sum`, `diff`, `jq`, and `git status/log/show/diff/rev-parse`; no TT
  device access, pytest hardware run, reset, server, profiler capture, or
  implementation/test mutation was performed.

## Residual Risk

- The current runtime and test hashes exactly match both the manifest and
  functional checkpoint. No functional artifact changed between
  `5fa49e9fa25` and current HEAD; later-stage code is out of scope.
- Real target weights and shapes cover sliding and full attention, aligned and
  non-aligned prefill, page/tile/window boundaries, batch 2/32, paged decode,
  nonidentity/disjoint page mappings, mutable trace inputs, determinism, and
  advertised context. All recorded acceptance PCC values exceed 0.995.
- The full-attention maximum-context reference is a periodic one-query HF
  oracle rather than stock HF over 262144 tokens. Its stock-HF validation,
  complete logical-position evaluation, real weights, wrong-position control,
  and above-threshold final PCC make the reduced harness acceptable.
- Static call-graph inspection and signposted device-only profiler windows
  support the no-host-fallback claim. Final warmed latency is 3.521/4.254 ms
  prefill and 2.577/2.911 ms traced decode for sliding/full. No required work
  remains in stage 01.
