# Stage Review

Verdict: clean-pass

Independent review, round 1 of 3, stage 1 functional-decoder for
`ornith-ai/Ornith-1.5-9B@489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
Reviewed the live, frozen worktree on `main`, base HEAD
`9e8a45111a69fe0531f65025022ab887ca59df5a`. The reviewed implementation SHA256 is
`cbebbad51893abdc7a189f8d9abf155814f8f036ec5022bbcf5738c151de2d50`.
Checkpoint commits are an explicitly pending post-review step; this verdict is
not a claim that the entire stage, including that step, has already closed.

## Required Work

- None for the reviewed implementation and completed evidence gates. No further
  hardware run is requested by this review.

## Other Concerns

- Finish the already-planned checkpoint with the evidence actually inspected.
  The four accepted `tracy/{linear,full}/{prefill,decode}_perf_report.csv` files
  exist locally, their hashes match `performance_summary.json`, and their
  sums are correct, but none is currently staged: repository `.gitignore:8`
  ignores `*.csv`. They also have no archive entries. Likewise, the latest
  `hooks_final.log` exists and passes, but is ignored and has no archive yet.
  Include these in the checkpoint, directly or losslessly archived with
  matching documentation/provenance. This is a pending commit-packaging item,
  not a failed measurement or a request to repeat the review or device tests.
- Record the stage-owned local commit SHA(s) after checkpointing. The supplied
  run plan explicitly forbids pushing; the reviewer made no remote writes.
- The reported latencies are single warmed observations at batch 1, prefill
  1024 and decode position 1024. The documentation correctly identifies them
  as layer measurements, distinguishes kernel sums from host wall time, and
  does not claim throughput or statistical confidence.

## Hard-Check Gaps

- The long-context comparison checks sampled first/middle/terminal windows,
  not every output row. This is disclosed accurately. Inspection of
  `long_context.py` confirms that TTNN computes the entire logical sequence,
  the HF linear path processes every token, and full-attention reference
  windows attend to HF-projected K/V for every preceding token. There is no
  hidden shorter attention history or advertised-context reduction.
- Repeated replay is a determinism and refreshed-request test: it restores the
  same prefix state before each replay. It does not establish a long chain of
  successive decode-state updates without restoration. A later generator
  integration should test that trajectory explicitly; this is not a missing
  gate in the supplied layer-stage contract.
- The retained single-row trace is exercised across nine prepared fresh
  requests, including chunk tails and changed page maps/positions. Explicit
  prefix-resume and ragged-batch cases are tested separately after that trace
  is released. A future generator must prepare all of its supported resume
  and batch signatures before its own first capture and test their actual
  interleaving; this review does not certify serving-level mixed-request
  trace reuse.

## Anomaly Ledger

- Observed anomaly: older full-attention JSON names a different implementation
  hash.
  Evidence: `full_final.json`, `full_long_fixed.json`, and
  `functional_decoder_before_gdn32.txt` use
  `15cd5c5d872665eccaa2fa1dcd34d0d4acb7d45a680cb00481d43ab59e264985`.
  Affected path: apparent source/evidence freshness.
  Control or comparison: exact saved-source hash and textual diff against the
  current implementation; current-hash full watcher and profile runs also pass.
  Likely subsystem: evidence versioning.
  Investigation performed: independently hashed and diffed both sources. The
  only changes are linear GDN constant dimensions and its prefill chunk size.
  Resolution: controlled; the full-attention execution path is unchanged.

- Observed anomaly: watcher builds originally exceeded kernel text capacity.
  Evidence: `linear_watcher_inline_failed.log`,
  `linear_watcher_noinline_failed.log`, `AUTOFIX.md`.
  Affected path: linear GDN prefill with watcher instrumentation.
  Control or comparison: GDN32 probe, final linear battery, full-context rerun,
  synthetic pytest, and current watcher battery.
  Likely subsystem: internal GDN kernel configuration.
  Investigation performed: checked the configuration change and affected
  rerun source hashes; inspected the actual accepted watcher logs.
  Resolution: fixed; both ordinary and watcher runs use GDN32, no checks are
  disabled, and both accepted watcher logs contain no suspicious entries.

- Observed anomaly: a profiler source-ID collision, then an overlay compile
  failure, produced unusable earlier captures; Tracy could exit zero after
  target failure.
  Evidence: `profiler_collision_diagnosis.json`,
  `profiler_collision_resolution.json`, `AUTOFIX.md`, rejected capture
  directories, and `tests/run_profiles.sh`.
  Affected path: measurement export, not the accepted numerical results.
  Control or comparison: accepted linear/full profile logs have
  `VALIDATION_PASS`, no exception/fatal markers, and complete signpost windows.
  Likely subsystem: profiler source-location metadata and wrapper exit status.
  Investigation performed: inspected relative overlay links and unchanged
  kernel-source use; independently counted accepted raw window rows and
  reconciled their timing sums with filtered CSVs and published numbers.
  Resolution: fixed; rejected captures contribute no reported timing.

- Observed anomaly: generic unsafe-allocation warnings occur in watcher runs
  while a trace is live.
  Evidence: both accepted watcher target logs, `validate.py:158-220`, the
  separate final tracked runs, and `trace_alloc_tracking/positive_control.*`.
  Affected path: eager prefill between decode replays.
  Control or comparison: the same prepared execution with allocation tracking
  enabled and program-cache exclusions disabled passes; a deliberately live
  unrelated buffer is caught.
  Likely subsystem: generic allocator warning when tracking is off.
  Investigation performed: reviewed pre-capture inputs/snapshots, temporary
  output destruction, exact-output acknowledgment, and replay ordering.
  Resolution: controlled; no broad exemption or unsafe persistent allocation
  was found. Each acknowledged output has one live trace and is consumed
  before it can be overwritten.

- Observed anomaly: long-context execution was initially mistaken for a hang.
  Evidence: `AUTOTRIAGE.md`, `AUTODEBUG.md`, `AUTOFIX.md`,
  `triage/full_long_summary.txt`, and completed long-context logs.
  Affected path: initial full-attention compilation.
  Control or comparison: triage checks pass and both original and corrected
  jobs finish; no reset or process termination was required.
  Likely subsystem: position-dependent compilation work.
  Investigation performed: inspected the bounded fill-page slice and
  device-valued SDPA start; confirmed the corrected full-context results.
  Resolution: controlled/fixed; no unsupported speedup claim is made.

- Observed anomaly: HF fallback and UMD topology warnings.
  Evidence: target logs and the setup/anomaly entries in `work_log.md`.
  Affected path: CPU reference implementation and physical-device discovery.
  Control or comparison: official HF Torch reference, runtime host-call
  guards, successful single-device computations and normal shutdowns.
  Likely subsystem: optional HF CPU accelerators and board identification.
  Investigation performed: compared HF and TTNN source paths and accepted
  logs. The TTNN GDN callers supply all constants and recurrent state, avoiding
  its optional host-construction fallbacks.
  Resolution: controlled; these warnings do not establish TT runtime fallback
  or device failure.

## Scope Inspected

- Goal/skill paths: the supplied original stage contract; installed
  `stage-review/SKILL.md`, `functional-decoder/SKILL.md`,
  `tt-device-usage/SKILL.md`, model-bringup startup, the relevant tracing skill
  requirements, and checkout `TraceCorrectness.md`. Enabled dependency
  inventory was checked from the retained setup evidence.
- Artifact paths: `README.md`, `functional_decoder.md`, `work_log.md`,
  `../context_contract.json`; final, long, watcher and profile JSON/logs for
  both kinds; allocation positive control; synthetic pytest and hook logs;
  watcher audit; profiler summaries/provenance, human tables, raw and filtered
  CSVs; anomaly reports, saved pre-GDN32 source, and archive manifest.
- Code paths: all of `tt/functional_decoder.py`; `tests/validate.py`,
  `long_context.py`, `run_decoder.py`, `test_functional_decoder.py`, profile
  and evidence runners, report renderer, archive helper, target config;
  relevant installed HF Qwen3.5 decoder/attention/GDN code and TTNN GDN
  composite helpers.
- Commands run: read-only `rg`, `sed`, `cat`, `nl`, Git status/diff/index
  inspection, and small standard-library Python analyses for AST parsing,
  config equality, source diffs, SHA256/archive validation, log scans, PCC
  minima, and raw/filtered CSV reconciliation. No TTNN import, hardware test,
  server, reset, or implementation mutation was performed. Only this report
  was written.
- Re-derived results: pinned target config matches the cached HF config
  exactly. Final whole-output PCC minima are 0.99968177 / 0.99966115;
  continuation per-row contribution minima are 0.99890006 / 0.99961668.
  Long sampled prefill minima are 0.99866915 / 0.99820769; final-position
  traced decode PCC is 0.99865657 / 0.99977314. All exceed 0.995.
  Negative history controls fail the same per-row continuation gate.
- Re-derived performance: linear prefill/decode kernel sums are
  13.239976 / 1.654715 ms; full prefill/decode sums are
  8.194264 / 1.840723 ms. Raw signposted device-row counts are
  103/116 and 70/71 respectively. Every accepted decode row carries trace ID
  0 and replay-session ID 4. The filtered CSV source hashes match the summary.
- Integrity checks: all 91 archived entries decompress to the manifest's
  recorded byte count and SHA256 and match retained originals; accepted
  watcher hashes match their audits, with 6481/7565 log lines and no suspicious
  scan matches. Staged diff checking is clean. All staged paths are within
  this model's autoport directory.

## Residual Risk

- This is layer-level evidence for the two meaningful layer kinds, using real
  layer 0 and layer 3 weights. It does not establish full-model generation,
  layer-stack accuracy, serving behavior, or optimized throughput.
- Long-context accuracy is sampled, and one-step traced decode agreement does
  not bound error accumulation in a long autoregressive rollout.
- Callers must honor the documented allocated-cache/page-map capacity and
  state ownership contracts, validate device-valued positions before replay,
  and prepare every path that will coexist with a retained trace.
- Final local checkpoint contents and SHA recording remain the stage owner's
  explicit post-pass responsibility. No push is authorized.
