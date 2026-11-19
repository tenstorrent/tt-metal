# Stage Review

Verdict: clean-pass

Fresh independent review, round 1 of 3, for stage 3 optimized-decoder,
Qwen/Qwen3.6-27B at revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`.
Reviewed the live worktree at HEAD `4c79322a4c142187308864977f00cbe5012c0352`
and independently verified delivered source SHA256
`315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.

## Required Work

- None. I found no unresolved correctness, contract, or required-evidence defect
  in the selected decoder path. The pass is for this decoder stage and its
  documented single-device scope. The stage owner must still record this review
  and create the stage-owned local checkpoint commit. The run plan prohibits
  pushing; this review performed no commit or push.

## Other Concerns

- The GDN accuracy margin is modest: the final B32/512-step minimum per-user
  PCC is **0.9955427456116543** against the unchanged 0.995 bar. The selected
  two-term BFP8 input projection, exact normalization/recurrent arithmetic,
  and persistence rounding are part of the passing configuration. Later
  multichip and full-model work must retain this evidence and revalidate its
  changed numerical boundaries. This is a residual risk, not a failed gate.
- GDN prefill has a disclosed accuracy/performance tradeoff. The final B1/S512
  prefill is 6.213297 ms versus fused 5.577254 ms; B32/S512 is 175.079354 ms
  versus 165.624200 ms. The targeted traced decode improves substantially,
  the fused GDN path fails the new long-sequence gate, and the report does not
  claim a universal prefill speedup. The satisfied stop rule permits the
  recorded tail-SwiGLU and multi-user prefill movement work to remain handoff
  items. This is consistent with the optimize evidence checklist's allowance
  for reported tradeoffs in the other phase.
- Two recurrence comments describe one-head ring capacity although the
  implementation allocates and initializes two slots. The actual capacities,
  one-head reserve/push/pop units, and writer completion barriers agree with
  the kernel contract. This is a comment cleanup, not a dataflow defect.

## Hard-Check Gaps

- The static runtime auditor follows direct function calls but does not resolve
  function-valued locals such as `norm_function`, `recurrent_op`, and `conv_op`.
  Thus its “43 reachable functions” is not a complete runtime call graph.
  Inspection of the selected indirect helpers, their embedded kernels, and the
  runtime conversion sentinels found no host tensor fallback. Improving that
  audit would make future regressions easier to catch; no additional current
  device run is required for this observation.
- `run_decoder.py` and `run_decoder_compact.py` record per-user PCC but assert
  the aggregate PCC. I independently checked all 1,356 recorded per-user and
  selected-context contribution values in the final JSONs: none fails 0.995;
  the minimum is 0.998022662160164. The sequence runner separately asserts its
  minimum across every user and step. Explicit per-user short-test assertions
  would strengthen future checks, but there is no hidden failing row here.

## Anomaly Ledger

- Observed anomaly: the stage-1 functional GDN recurrent state is a poor oracle
  on the recorded correlated text activations.
  Evidence: `text_state_functional.json` has prefill recurrent PCC
  0.01702516259500038, while `text_state_fused.json` has 0.9999876304669332 on
  the same activation hash; `AUTODEBUG_functional_state_oracle.md` records the
  source diagnosis.
  Affected path: GDN prefill/state equivalence validation.
  Control or comparison: independent HF cache state and fused decoder, followed
  by `oracle_repaired_text.json` and the frozen final text run.
  Likely subsystem: the old functional polynomial-inverse implementation.
  Investigation performed: inspected the matched diagnostic JSONs and the
  final harness's actual independent-HF state comparisons. Final GDN text-state
  PCC is at least 0.9999900434585203 across prefill and two decode boundaries.
  Resolution: controlled; the defective functional state is not the final
  acceptance oracle, and the HF state threshold remains 0.995.

- Observed anomaly: short correctness does not establish long GDN stability.
  Evidence: `sequence_fused_0_b32.json` fails at 128 steps; the indexed
  `control37_fused.json` fails the 512-step comparison. Several faster
  experimental arithmetic/fidelity paths also failed long checks.
  Affected path: cumulative recurrent decode.
  Control or comparison: final real-weight, changing-input teacher-forced runs
  against the HF layer, including B32/512 and B1/128 for both layer kinds.
  Likely subsystem: recurrent numerical error accumulation.
  Investigation performed: read the sequence runner and independently
  recomputed the minima from all final per-step/per-user values.
  Resolution: fixed in the selected path; final B32 minima are
  0.9955427456116543 GDN and 0.9966053676872968 full attention. The failing fused
  timing is explicitly historical, not an eligible correct long-run candidate.

- Observed anomaly: GDN prefill regressions and a large one-token-tail cost.
  Evidence: `final_measurements.md`, raw final performance JSONs, and the
  S512/513/1025 prefill CSVs.
  Affected path: warmed prefill, especially the composite MLP tail.
  Control or comparison: matching legacy-layout fused baselines and repaired
  prefill configuration candidates in `work_log.md`.
  Likely subsystem: additional precision work and physical composite padding.
  Investigation performed: recomputed baseline/final timings and profile sums;
  checked that phase-only fidelity/configuration alternatives were measured and
  their accuracy or latency outcomes were recorded.
  Resolution: controlled; the tradeoff is explicit, with further work listed
  under the satisfied optimization stop rule.

- Observed anomaly: allocation-corruption messages appear in final logs.
  Evidence: 14 final correctness/pytest log occurrences of the deliberately
  unsafe-buffer positive control; corresponding result fields are true.
  Affected path: trace allocation validation.
  Control or comparison: a live post-capture buffer must reject replay, then
  releasing that buffer must permit replay.
  Likely subsystem: expected tracker enforcement, not spontaneous corruption.
  Investigation performed: read the positive-control code and final environment
  records. All 26 correctness jobs enable tracking and include program-cache
  allocations; no acknowledgment or broad allocation exemption is used.
  Resolution: controlled.

- Observed anomaly: profiler FLOPs percentages can exceed 100%; final profiling
  logs also contain optional Tracy web-copy and CSV dtype warnings.
  Evidence: all eight final profile directories and their logs.
  Affected path: reporting and interpretation, not decoder output.
  Control or comparison: stored raw operation CSV hashes, rendered CSV sums,
  explicit physical-byte accounting, and same-run host timing.
  Likely subsystem: nominal tool/core-count accounting and optional report export.
  Investigation performed: decompressed and hashed each raw CSV, recomputed both
  phase sums and decode traffic, and checked the measured dtype/fidelity rows.
  Resolution: controlled; those percentages are not used for stop-rule claims,
  and complete compact reports/raw CSVs exist despite optional web-copy warnings.

## Scope Inspected

- Goal/skill paths: `stage_review_brief.md`; installed version 0.1.23
  `stage-review`, `optimize`, `tt-device-usage`, and `tt-enable-tracing` skills;
  model-bringup startup; optimize performance-accounting/evidence-checklist
  references; repository `TraceCorrectness.md`.
- Artifact paths: README, work log, checklist, delivery manifest,
  `final_evidence_check.json`, final measurement/profile conclusions,
  `stop_rule_selection.json`, capacity/context contracts, runtime/kernel audits;
  every listed final-suite JSON and exit ledger; final pytest log; all eight
  final profiles' compressed raw CSVs, rendered CSVs, metrics and accounting;
  selected baseline, precision, geometry, oracle and exact-kernel probe artifacts.
- Code paths: selected defaults, setup and weight packing/correction,
  prefill/decode dispatch, normalization, projections/MLP, page allocation,
  SDPA/continuation, compact recurrence, QKV staging, unflatten, affine/gate and
  beta/decay helpers in `tt/optimized_decoder.py`; final launchers and job wrapper;
  `tests/evidence.py`, `audit_runtime.py`, `run_decoder.py`,
  `run_decoder_compact.py`, `run_decode_layout.py`, `run_contract.py`,
  `run_trace_reuse.py`, `run_mixed_decode.py`, `run_long_context.py`,
  `test_optimized_decoder.py`, and `account_decoder_perf.py`.
- Commands run: read-only `git status`, `rg`, `sed`, `cat`, and small standard
  library Python artifact analyses. Analyses verified final source and harness
  hashes, all 58 suite records and zero-exit ledgers, final PCC arrays, trace IDs,
  baseline/final medians, profile raw hashes/sums/dtypes/bytes, and stop-rule
  arithmetic. Embedded C++ kernels independently match the probed helpers
  byte-for-byte: 9 norm, 3 recurrence, 1 QKV, 3 affine, 3 beta, and 2 unflatten
  strings. No TTNN import, device operation, test, server, or reset was run.

Key independently checked conclusions:

- The 58 suite runs use the frozen default source with empty policy/candidate
  overrides; recorded runner hashes match the current harnesses. The four
  pytest cases explicitly select `OptimizedDecoder` and pass.
- Both kinds process all 262143/262144 prefill tokens with diagnostic overrides
  disabled. The 262143-prefix runs trace decode to context 262144. The minimum
  selected contribution PCC at maximum prefill is 0.9994892490246287 GDN and
  0.9982123418300793 full attention. High-position full-attention continuation
  also passes. No context reduction or B32 maximum-context claim is hidden.
- Cross-request reuse retains one trace across the prepared lengths and resume
  offsets, with one capture and zero request releases/invalidations. The runner
  actually prepares variants before capture and replays after each prefill;
  this conclusion does not rely solely on literal JSON counters.
- Matching legacy B1/S512 decode improves 2.132502 to 1.166630 ms GDN and
  2.041142 to 1.031517 ms full attention. B32/S33 improves 7.963243 to
  1.681194 ms and 4.922335 to 1.185567 ms. Preferred compact final B32 medians
  of 1.580467/1.079028 ms reproduce the selected candidates.
- All final decode matmul rows are BFP8. GDN attention uses HiFi4 and MLP
  HiFi2; full attention uses LoFi. The actual residual matrix width is 11520.
  The GDN B32 necessary traffic is 575,307,776 bytes, yielding 1.123648 ms at
  512 GB/s. Dividing by the same-run 1.602468080818653 ms host median gives
  **70.11983661015961%**. Full-attention B32 independently reaches 70.554854%.
  The last two kept GDN gains recompute to **0.384237% and 0.393176%**.
  The owner's roofline/gain stop rule is satisfied; neither elapsed wall time
  nor the excluded all-items-assessed shortcut is needed.

## Residual Risk

- This is a source/artifact review, not a new hardware execution or exhaustive
  proof of all embedded kernel schedules. Exact probes, watcher, changing-input
  replay, awkward batches/tails, and full-context evidence support the selected
  Blackhole implementation.
- Decoder PCC and real-text activation probes do not establish full-model
  generation quality. No generator or TT text output is expected in this scope;
  later model/precision stages must supply their own corresponding gates.
- Maximum-context validation is B1. Trace setup and stable request-state
  ownership remain caller responsibilities until a generator is implemented.
  Multichip operation and policy overrides are not certified by this review.
