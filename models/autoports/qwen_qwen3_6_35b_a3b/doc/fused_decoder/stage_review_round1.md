# Stage Review

Verdict: more-work-needed

Independent review, round 1 of 3, of repo-local stage 2, fused decoder for `Qwen/Qwen3.6-35B-A3B`. Reviewed the live worktree on `main`, HEAD `4c643884bf51a39fc1166e0b86b86bd373bedfee`. The runtime SHA256 is `8ac733159befba919abf705603f3aada9d105195c7b495d8cd25aa65e531769f` and matches the selected V5 snapshot and current failing run.

The delivered implementation cannot pass the original accuracy contract. The completed AutoFix experiments substantiate an unsuccessful repair outcome; they do not establish that a solution is impossible. I found no additional, already-supported model-local patch that would close this failure. The user's instruction permits stopping after AutoFix fails, with the stage explicitly incomplete. That exception does not make this a passing review or satisfy the optimization completion condition.

## Required Work

- **P1: The actual layer 34 decode path fails the unchanged HF accuracy gate.**
  Evidence: [the fresh required regression](stress_representative_34_seed154.json) contains 24,576 decode rows: 8,192 positions across three repeats. There are 888 failing rows at 296 unique positions. At position 7944, output PCC is `0.927889633267` and contribution PCC is `0.912541466536`, below `0.995`. The earliest failure is position 137. Prefill PCC is `0.999716380061`; all 33 continuation rows pass. I independently recomputed all first-repeat decode PCCs from `stress_representative_34_seed154.json.tensors.pt`: the maximum difference from the JSON was `2.01e-14`, with the same 296 failures and finite outputs. The tensor artifact SHA256 is `2b08e72dbe08fc524a19195986d8f53740c7d45467b5ff354c644f888122d936`. Provenance binds this run to current runtime and stress-harness sources, fresh HF, allocation tracking, and included program-cache allocations. The [aggregate gate](hard_gates.json) correctly reports FAIL.
  Why this matters: Layer 34 uses the same supported linear-attention implementation as layer 0, with actual checkpoint parameters. Passing layer 0/3, deterministic repeats, and capacity checks cannot waive observed incorrect outputs on this supported path. The functional and V4 controls also fail; neither is an original-HF control demonstrating expected model behavior.
  Required next step: Before any stage promotion, produce a model-local device implementation that passes this retained regression and affected existing gates with the original reference and thresholds. The concrete unresolved boundary is expert selection and normalization, together with remaining upstream numerical differences. Start any resumed investigation from the saved same-logit fixtures and actual intermediate captures, then run a focused discriminator before another long stream. Preserve the failed artifacts. A passing successor needs source-bound representative correctness, replay/watcher checks, capacity checks if residency or layout changes, and matched final timing/profile evidence. Existing unaffected evidence may be reused; there is no reason to repeat the same failed candidates unchanged.

  **Repair evidence adjudication:** `tt/fused_decoder.py:679–699` uses the generalized gate for decode. On the captured position-7944 logits, CPU selection includes experts `202,64`; the native gate includes `64,74`. The tested direct-topk and softmax/topk alternatives include `74,202`, also a different membership. I reproduced all three saved CPU tie-permutation cases from [the proof](endurance/v5_torch_topk_tie_permutation_proof.json): changing only lower logits changes the tied subset, so a fixed expert-index bias is not an evidenced fix. The installed Torch implementation uses value-only `nth_element` at K8/N256; native stable topk specifies lowest original index and is not the same contract. Selected-score discrepancies are independently present.

  The [18-control comparison](endurance/v5_layer34_repair_comparison.json) agrees with the raw artifact hashes and failure counts, including continuation failures where present. Accurate first residual ADD, ordered Q/Z and postnorm, explicit gated norm, ordered output projection, all three native routing alternatives, and complete compatible combinations were tried. The final BF16-topk composition still fails 203 positions with minimum contribution PCC `0.946055341978`; the softmax/topk composition fails 200 with minimum `0.930944200802`. Both fail position 7944 at approximately `0.986205319036`. These are actual failed model gates, not synthetic-only rejections.

  There are concrete *partial* numerical repairs within scope: accurate first residual ADD matches the captured CPU addition; explicit norm and ordered output projection repair sampled local boundaries. They are already tested, individually or in composition, and are insufficient. An on-device dynamic selector reproducing the frozen CPU tied membership remains an unimplemented investigation, not a verified patch or guaranteed complete fix. The evidence does not justify requesting a host fallback, weakening the oracle, promoting one of the partial repairs, or asserting that a shared/native edit is the only possible remedy. [The AutoFix outcome](endurance/AUTOFIX_v5_layer34.md) accurately records these limits.

- **P2: The original optimization completion condition has not been established.**
  Evidence: I recomputed the non-overlapping completed intervals in [the device ledger](device_time_ledger.json): `18,461.864 seconds = 5.128295556 hours`, below eight hours. This is session ownership, including host work during an open session, rather than active kernel time. [The current roofline report](profile_v5/roofline.json) is explicitly an optimistic bandwidth-only estimate; its E2E fractions are `19.8353%` linear and `38.0862%` full attention. It does not establish the requested 70% condition. The last-two-gains condition alone cannot satisfy that branch. The goal specifically names the 70%-plus-gains or eight-hour stop conditions; a long pattern inventory is not a substitute.
  Why this matters: An incomplete accuracy repair and an optimization Stop Rule are different outcomes. Neither current source nor a future repaired successor can be labeled a completed fused stage using the current stop evidence.
  Required next step: If work resumes and P1 is repaired, finish the applicable graph-fusion work to a condition permitted by the original goal, recording untried candidates as hand-offs when that condition fires. Do not spend idle device time merely to fill the ledger. Under the user's failed-AutoFix exception, stopping now is compatible with an explicitly incomplete stage; this finding does not demand redundant device experiments after the documented failure.

## Other Concerns

- **Historical qualification wording was partly corrected during review.** I verified the owner's README update now limits V4 to its original layer 0/3 suite and explicitly says neither V4 nor V5 satisfies the expanded accuracy gate. V4 fails 282 positions on the matched layer 34 stream. `pattern_assessment.json` still retains a “current model evaluation pending” entry for the combined ordered-Q/Z and postnorm experiment even though its completed failure is in the comparison. Update that current summary and preserve historical records as historical. This wording fix does not need another review round.
- **The source-audit state description is stale.** `tests/audit_fused_source.py` emits `BF16 TILE backing [1,3,8192]` for convolution state, while `tt/fused_decoder.py:471–472` allocates ROW_MAJOR and `doc/context_contract.json` correctly records ROW_MAJOR. Correct that emitted description on the next evidence maintenance pass; the actual source, runtime results, and context evidence already establish the layout. This is not a new device correctness failure.
- **Performance evidence is useful but remains conditional on accuracy.** At S256/context512, final V5 linear prefill/decode is `106.239562 / 1.285348 ms`, versus its manifest control `115.624242 / 1.295863 ms`; full attention is `8.985362 / 0.583571 ms`, versus `9.004255 / 0.587704 ms`. These are measured matched-harness improvements on the recorded layer 0/3 workloads. They cannot turn the known failing implementation into a best-correct stage result, and should not be carried unchanged onto a numerical repair. The original functional linear-prefill regression is disclosed and is not silently counted as a fusion speedup.
- The run plan says local checkpoint only, never push. The owner reported staging during this review, with no commit or push. A local evidence checkpoint may preserve this failed state; it must not be labeled accepted stage completion. No push state is inferred from reflog history.

## Hard-Check Gaps

- The revised checker explicitly requires the layer 34 regression and overwrites the inherited intermediate PASS with FAIL on an exception. I inspected this control flow rather than invoking the checker, because it writes evidence and the review was otherwise read-only. The prior `qualified_v5/hard_gates.json` is a preserved old-suite result, not the current aggregate result.
- `stress_fused_decoder.py` sets `repeated_bit_equal=true` even for a one-repeat diagnostic. Its final checker correctly requires at least three repeats; single-repeat repair controls are not acceptance evidence. Also, the stress script's local failure predicate does not explicitly reject NaN metrics, while the final checker does. Current failing artifacts are finite, so neither issue explains or hides this failure.
- The final performance comparator list has layer 0/3 endurance qualification. It does not establish correctness on the newly failing layer 34 case. Correctness eligibility must be reconsidered with the expanded regression when selecting a successor, rather than inferred from the old manifest's name.
- The 138 pattern-assessment entries cover the graph-fusing catalog and many adapted candidates, but the review did not independently re-execute every old candidate or all native op contracts. No additional unmeasured performance candidate is elevated to Required Work merely because it might be faster.

## Anomaly Ledger

- Observed anomaly: Large sparse output/contribution failures with bit-equal repeats.
  Evidence: `stress_representative_34_seed154.json`, its tensor sidecar and provenance; independently recomputed metrics above.
  Affected path: Actual layer 34 linear decode, including attention, residual/postnorm, routing and expert combination.
  Control or comparison: V4 and functional implementations also fail; current layer 0/3 streams pass.
  Likely subsystem: Numerical differences amplified at expert selection, with a separately proven tied-membership discrepancy. Not proven to be exclusively the router or exclusively recurrence.
  Investigation performed: Fresh required regression, 18 source controls, immediate intermediate captures, same-operand norm/output-projection probes, native router alternatives and complete compositions.
  Resolution: more-work-needed; AutoFix unsuccessful.

- Observed anomaly: Native expert membership differs on identical tied logits.
  Evidence: `endurance/v5_router_tie_device.json` and `endurance/v5_torch_topk_tie_permutation_proof.json`.
  Affected path: Decode top-8 routing at position 7944; selected scores also differ.
  Control or comparison: CPU recomputation on the actual TT logits and three lower-logit permutations; all three native alternatives.
  Likely subsystem: Native selection network versus Torch CPU value-dependent partition behavior; normalization arithmetic is a separate boundary.
  Investigation performed: Source inspection and recorded same-input device observations; reviewer reproduced CPU proof without TTNN.
  Resolution: more-work-needed. Tied scores do not waive this project's failed HF PCC gate.

- Observed anomaly: Historical convolution tilize/widen folding is not bitwise equivalent for all inputs.
  Evidence: `endurance/AUTOFIX_v4_conv_widen_boundary.md`; selected source and explicit diagnostic failure flags.
  Affected path: FP32 convolution tap operands and very small adversarial outputs.
  Control or comparison: Retained component diagnostics and target-weight layer 0/3 streams; observed differences reach approximately `1.65e-38` in constructed outputs.
  Likely subsystem: Conversion behavior near underflow.
  Investigation performed: Exhaustive conversion and constructed convolution tests; current recorded numerical tolerance was not changed.
  Resolution: controlled as a bounded component difference under the unchanged model PCC contract, not universal exactness. It does not resolve the separate layer 34 accuracy failure.

- Observed anomaly: A historical V4 profiler run aborted; cached metadata could describe an earlier shape.
  Evidence: `endurance/AUTOFIX_v4_profiler.md`, `endurance/AUTOFIX_profiler_cached_shape_metadata.md`, current `profile_v5/profile_metadata_audit.json` and raw profiles.
  Affected path: Instrumentation and interpretation, not a demonstrated model output fault.
  Control or comparison: Fresh V5 collection with `TT_METAL_PROFILER_NO_CACHE_OP_INFO=1`; separate watcher runs.
  Likely subsystem: Profiler marker processing and metadata cache.
  Investigation performed: Preserved failed capture, unchanged retry, fresh metadata collection and completeness validation. I checked all four CSV hashes and independently summed device durations; values exactly match `perf_summary.json`. Sorting decode report rows by operation ID gives 16 identical operation-code sequences of 166/68 operations. Report presentation order itself is not execution order.
  Resolution: controlled for the current profiles; no upstream native bug-fix claim.

## Scope Inspected

- Goal/skill paths: `goal_contract.md`; installed `stage-review`, `graph-fusing`, `optimize`, `tt-device-usage`, `model-bringup` startup, and `tt-autodebug/autofix` instructions under the exact package roots supplied in the review brief. No applicable ancestor `AGENTS.md` was present at the checked workspace paths.
- Artifact paths: Current README/work log/source freeze/hard gates, the fresh layer 34 regression and provenance/tensors, all 18 repair results referenced by `endurance/v5_layer34_repair_comparison.json`, AutoDebug/AutoFix and routing-tie reports, final compatible-composition decision, same-logit router device results, CPU tie proof, pattern assessment, device-time ledger, best-correct manifest and timing artifacts, `qualified_v5/preservation.json`, `profile_v5` summaries/metadata/completeness and four performance CSVs, watcher audit, context contract, maximum-context results, boundary timing rows, battery summaries, trace reuse and page/padding guards.
- Code paths: Entire `tt/fused_decoder.py`; `tests/check_fused_evidence.py`, `validate_fused.sh`, `stress_fused_decoder.py`, `audit_fused_source.py`, `runtime_audit.py`, `trace_reuse.py`, `summarize_perf.py`, `restore_fused_evidence.py`; relevant repair snapshots; installed Torch `TopKImpl.h`; native topk and generalized-gate binding contracts.
- Commands run: Read-only `cat`, `sed`, `rg`, `wc`, `git status`, `git diff HEAD --name-only`, and `git rev-parse`; small standard-library JSON/hash/AST/CSV/interval analysis scripts; CPU-only Torch analysis of saved tensors and saved tie fixtures; `python models/autoports/qwen_qwen3_6_35b_a3b/tests/restore_fused_evidence.py --verify-only`.
- Verification results: All 62 V5 preservation hashes match; all 18 repair result hashes and failure counts match; all 1,595 text archives verify with zero restores and no mismatches. No tracked differences outside the allowed model root were found. Current capacity evidence is source-bound and retains 262143/262144 prefill plus traced decode at position 262143. The nonaligned 127/128/129/257/4097 timing/correctness rows remain present. Runtime code inspection found no torch/from_torch/to_torch fallback in forward methods. Setup and test readbacks remain separate.
- Reviewer actions: No TTNN import, device operation, server, reset, hardware test, implementation edit or subagent. Only this report was written.

## Residual Risk

- The saved capture set does not prove the earliest divergence for every failed position, and matching tied membership alone is not proven sufficient. Incoming recurrent-state differences and upstream rounding remain relevant.
- The original V5 watcher/context/profile evidence is valid for the unchanged recorded source and exercised layer 0/3 paths; it is not evidence that layer 34 is correct. A future repair can invalidate parts of that evidence according to its actual changed branches and allocation rules.
- The current roofline is not a full compute roofline, and the device ledger is not an active-kernel counter. Neither supports a completed Stop Rule now.
- Binary captures/oracle tensors used for deep diagnosis remain local. The archived text metrics, provenance, CSVs and logs are reproducible from the checked restoration manifest, but do not replace the binary intermediates for future numerical forensics.
- No generation/full-model/serving claim is reviewed or implied. This verdict leaves the fused-decoder stage incomplete and does not authorize later stages.
