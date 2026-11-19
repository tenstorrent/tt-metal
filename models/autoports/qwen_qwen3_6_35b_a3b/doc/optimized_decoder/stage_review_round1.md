# Stage Review

Verdict: clean-pass

Independent round 1 review of stage 3, optimized-decoder, for
`Qwen/Qwen3.6-35B-A3B`. Reviewed the live worktree on `main`, based on
`575fea054263db4505a340533a19a99d72b1ab95`. The reviewed production source is
`tt/optimized_decoder.py`, SHA256
`99e2a2fa1d4cb78f4dd0eb95079cc8288e8f7e0ebd92e6ffbc44ffb3be05cb00`;
the inherited fused source is `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87`.
No hardware, TTNN import, server, or new reviewer was used in this review.

## Required Work

None. Two evidence-presentation issues found during review were corrected by
the owner within this round: inaccurate full-attention chunk counts and missing
prominent disclosure of the S127 prefill regression. I inspected the corrected
summarizer and documentation. These changes did not alter production code or
require new device measurements.

The original goal explicitly makes .995 a whole-output PCC gate and per-row
PCC diagnostic. The historical AutoDebug recommendation to restore a universal
per-row .995 gate is therefore not the governing acceptance contract.

## Other Concerns

- Full-attention S127 prefill is a measured tradeoff: 41.864672 to 43.940496 ms,
  a 4.96% regression. All three optimized samples exceed all three baseline
  samples. The corrected README, performance conclusions, and work log disclose
  this, retain the selected path for its decode/headline benefits, and defer
  tail-specific tuning under the optimization stop rule. The stage does not
  establish a prefill speedup at every logical length.
- Near-tied expert routing remains a numerical sensitivity. In particular,
  continuation position257 has an inherited FP32-versus-BF16 router boundary
  difference even on identical router inputs. Its complete decoder row passes
  .995; this stage does not prove exact routing agreement or full-model token
  agreement. Preserve the diagnostic for later full-model/precision evaluation.
- Further geometry, recurrence, KV-dtype, and short-tail performance work is
  legitimately bounded by stop rule 2. The residual search independently
  supports consecutive gains of about .716% and .263%. These handoffs do not
  waive correctness, and the delivered gates supply that evidence separately.
- Local checkpoint commit and its receipt follow this passing review. They
  are not yet established by this report. The authorized run plan says never
  push from this stage; no push was performed or inferred from Git history.

## Hard-Check Gaps

- No unresolved required gate was found. I called `Audit(...).run()` directly
  in a read-only Python invocation, rather than the CLI that rewrites
  `hard_gates.json`; it returned PASS over 61 checked artifacts.
- The hard checker does not establish candidate ranking or optimization
  tradeoffs. I separately inspected candidate commands and their actual result
  files, recomputed medians, and compared the final path with the fastest
  eligible alternatives. The 0.430870/0.430929-ms screens use the selected
  decode policy; the reported final number is 0.431675 ms. The best different
  residual policy is 0.432066 ms. Differences at this scale are within the
  recorded variation, so this is reproduction of the selected policy, not
  proof of a statistically separated win over every nearby configuration.
- The old layers0/3 stress artifacts retain per-row results rather than the
  newer concatenated-stream fields. They are not a mathematical substitute
  for aggregate PCC. Aggregate coverage is supplied independently by final
  prefill/decode, context, and the current 128/128 functional-sequence artifacts;
  the old stress runs additionally establish changing-input endurance and
  determinism. The late-linear aggregate reruns are explicit and passing.
- Source provenance records more test files than each run executes. I checked
  the differences against current files for the final and late-linear stress
  runs: later changes were in the checker, candidate-only prefill helper,
  diagnostic probe, and profiler wrapper, not the executed clean timing or
  stress path. Production and inherited fused hashes match throughout the
  accepted evidence.

## Anomaly Ledger

- Observed anomaly: Layer34 synthetic continuation position276 has output PCC
  .990702112 and residual-subtracted contribution PCC .988451614.
  Evidence: `initial_stress_34.json`, `stress_34.json`,
  `diagnostic34_fused.json`, `stress34_controls.json`,
  `continuation_router_probe_fixed.json` and its saved intermediate tensors.
  Affected path: Multi-token linear-attention continuation after 128 prefill
  and 128 decode steps.
  Control or comparison: Unchanged fused output PCC .990553211 at the same
  boundary; optimized decode/continuation aggregates .999913701/.999794035.
  Natural HF-propagated inputs pass the same boundary, minimum row .999873394.
  Likely subsystem: Discontinuous top-8 routing after a small upstream
  numerical difference.
  Investigation performed: Checked rollback controls and the repaired probe's
  capture hook; independently recomputed saved-tensor input PCC .9999992645
  and MoE output PCC .9667548968. TT's recorded IDs agree with its logits;
  the eighth expert is254 versus HF's101 on HF's own input. The probe's
  same-TT-input HF control also selects254 and reports MoE PCC .999799595.
  Independently recomputed saved decode aggregates for original optimized,
  matched optimized, fused, and repaired diagnostic runs.
  Resolution: controlled. The specified aggregate gate passes, the anomaly
  predates this optimization, and there is a concrete mechanism/control.
  This is not mislabeled as an HF-versus-HF known limitation or a guarantee of
  full-model quality. The first contaminated diagnostic probe is not used.

- Observed anomaly: Position257 selects different cutoff experts even with
  identical postnorm input.
  Evidence: `continuation_router_probe_fixed.json`, row1; inspected probe
  source and inherited/optimized `_moe` paths.
  Affected path: Multi-token router precision boundary.
  Control or comparison: HF on TT input and BF16-rounded TT logits select110;
  the production FP32 route selects21. Same-input MoE PCC is .994705346;
  complete decoder output remains about .997687.
  Likely subsystem: Inherited router rounding/top-k semantics.
  Investigation performed: Separated this from position276 and from the new
  expert weight/fidelity policy; inspected its documented residual risk.
  Resolution: controlled under this decoder's acceptance contract; retain for
  subsequent model-level accuracy work.

- Observed anomaly: S127 prefill regresses, and the original latency table
  understated full-attention call counts.
  Evidence: `baseline_l3_s127.json`, `final_l3_s127.json`,
  `latency_comparison.csv`, and `prepare_prefill` in `tt/fused_decoder.py`.
  Affected path: Eager prefill with singleton tails.
  Control or comparison: Matched serial measurements; S127 executes96 plus31
  singleton calls, S129 executes128 plus1, and S4095 executes15 calls of256,
  one224, then31 singletons.
  Likely subsystem: Table used `ceil(S/256)` instead of the actual planner;
  singleton-heavy eager execution is distinct from warmed traced decode.
  Investigation performed: Compared raw samples, recorded maximum chunk
  lengths, planner branches, and the corrected summarizer. Exact latency
  attribution among kernels/dispatch was not measured and is not claimed.
  Resolution: fixed for metadata; controlled and explicitly retained as a
  measured performance tradeoff. No correctness or logical-length restriction.

- Observed anomaly: Final prefill5.876 ms differs from selected screen5.600 ms;
  final decode .431675 ms differs from screen .430929 ms.
  Evidence: `final_3.json`, `prefill_search/2d_b8_l3.json`,
  `residual_search/sharded_moe_reshard_l3.json`.
  Affected path: Candidate-to-final performance reproduction.
  Control or comparison: Same selected policy, overlapping recorded samples;
  final numbers replace screening numbers in the delivered claims.
  Likely subsystem: Run-to-run timing variation, especially eager prefill.
  Investigation performed: Recomputed medians and inspected sample arrays and
  final policy wiring.
  Resolution: controlled; no older faster measurement is substituted for final.

- Observed anomaly: Projection setup failures and a block64 L1 allocation
  failure appear in the search history.
  Evidence: `projection_search/`, `projection_search_retry/`, including
  `bfp8_readers3_block64_l3.log`.
  Affected path: Experimental DRAM-sharded projection geometry.
  Control or comparison: Adapted legal reader1/2/3 and block4/8/16/32 runs
  complete; block64 requests2,774,976 bytes versus1,572,864 available.
  Likely subsystem: Initial API use, then a physical per-core L1 limit.
  Investigation performed: Inspected failure signatures, successful retry
  result files, matched precision, and final runtime geometry.
  Resolution: fixed for the API attempt; controlled exact blocker for block64.
  The optimization family was not rejected on its first error.

- Observed anomaly: First BFP4 profiler capture fails JSON argument parsing.
  Evidence: `tracy/bfp4_candidate_quote_failure/capture.log`, repaired
  `profile_optimized_decoder.py`, and `tracy/bfp4_candidate/`.
  Affected path: Diagnostic profiler launch, before model execution.
  Control or comparison: Repaired environment-policy transport; successful
  capture with actual BFP4/LoFi sparse rows and passing PCC/determinism.
  Likely subsystem: Tracy subprocess argument quoting.
  Investigation performed: Inspected parser failure, wrapper policy default,
  successful result, and profiler dtype/fidelity checks.
  Resolution: fixed. Production remains BFP8; grouped prefill remains BFP8 in
  the BFP4 decode candidate.

- Observed anomaly: Tracker log reports a live buffer that would be corrupted;
  normal logs also contain motherboard, subset-device, clock-settling, and
  no-op `to_layout` warnings.
  Evidence: `logs/trace_reuse_watcher_{0,3}.log`, corresponding raw Watcher
  logs, `logs/baseline_3.log`, and `logs/final_{0,3}.log`.
  Affected path: Positive-control allocation and runtime diagnostics.
  Control or comparison: Deliberate hazard is rejected and deleted before
  subsequent successful replay; no-op layout warning also occurs in fused
  baseline. Selected jobs use one physical chip, not remote-chip traffic.
  Likely subsystem: Intended tracker rejection and existing diagnostics.
  Investigation performed: Inspected exact-output acknowledgment, single-live-
  trace lifetime, positive-control code, nine-request trace reuse, and raw
  Watcher dumps. Found3/2 completed dumps and no assert/corruption/fatal error.
  Resolution: controlled. No broad allocation exemption or program-cache skip
  is used; the clock warning is recorded variation, not a claim of exact1350MHz.

## Scope Inspected

- Goal/skill paths: Original
  `/home/ttuser/qwen36-35b-a3b/multigoal_logs/03-03-optimized-decoder.prompt.txt`;
  installed0.1.24 `stage-review`, `optimize`, `tt-device-usage`, relevant
  `tt-enable-tracing` requirements, model-bringup startup, and optimize method,
  evidence-checklist, precision, perf-report, and prefill/TTFT references.
  Applied the owner's revoked-cap clarification and decoder-only scope.
- Artifact paths: Stage README, work log, checklist, topology audit, hard gates,
  performance/validation summaries, candidate ranking, manifest, context
  contract; baseline/final and boundary JSON/provenance; stress0/3/34 at128/1024;
  natural-input metadata and results; functional128/128; B32 batteries;
  trace reuse, page extent, padded trace, pytest XML; AutoDebug/AutoFix and
  routing controls; search results and selected failure logs; raw Tracy
  `ops.csv`, advice-enabled reports, grouped tables, and Watcher logs.
- Code paths: Entire `tt/optimized_decoder.py`; inherited fused constructor,
  attention, routing, planner, prefill/decode/batch dispatch; explicit factory,
  real-weight pytest, run/stress/battery/trace harnesses, runtime audit,
  provenance, profiler wrapper, activation preparation, continuation probe,
  hard checker, and summary scripts. Staged changes stay inside this model's
  code/tests/docs; only `optimized_decoder.py` changes under `tt/`.
- Commands run: Read-only `rg`, `sed`, `cat`, Git status/diff inspection;
  standard-library JSON/CSV/source-hash analyses; read-only `Audit.run()`;
  CPU Torch `weights_only=True` loads and PCC/top-k calculations on existing
  small diagnostic tensors. No acceptance artifact was regenerated by this
  reviewer; the sole write is this report.
- Independent numerical checks: Clean medians reproduce1.447121→1.346515ms
  linear decode and .5830345→.431675ms full decode. Raw profiler signposts
  contain11920/104 prefill ops and2640/1152 decode ops across16 replay sessions,
  giving454.499884/5.451319ms prefill and1.2057035/.3912945ms decode kernel sums.
  Actual expert rows are BFP8/LoFi; shared/full projections are BFP8/HiFi2;
  full router remains FP32/HiFi4. Functional384 timing samples per layer give
  means1.346337/.428768ms and the documented15.09%/28.04% logical-bandwidth
  floors. B32 batteries contain32 prefill,32 eager decode,96 traced rows and
  determinism, plus64 full-attention cache checks and untouched guard pages.
  All242 manifest paths checked at initial inspection exist.

## Residual Risk

- This validates representative decoder layers 0/3/34, not the 40-layer stack,
  generation, token accuracy, serving, or an across-prompt precision frontier.
  Natural activation evidence uses one recorded prose stream and does not
  establish qualitative chat quality.
- Batch32 is a traceable per-request decoder loop with owned state. Its
  correctness is covered; this is not evidence of a fused batched serving path
  or batch32 throughput.
- Maximum262144 context is inherited with unchanged BF16 KV/cache layout and
  reduced weight payload. Current stage smoke covers32767/32768 and traced
  decode for both layer kinds, as required; this review does not claim a new
  optimized maximum-context hardware run.
- Logical-byte rooflines are estimates; kernel sums exclude dispatch gaps and
  profiled host timing includes instrumentation. Clean and profiled timings
  are correctly separated. Tiny candidate differences should not be treated
  as statistically established improvements.
- The passing verdict applies to the stated production hash and present
  evidence. Stage completion still requires the owner's local checkpoint
  receipt and work-log update after this review.
