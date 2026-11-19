# Stage Review

Verdict: more-work-needed

Independent follow-up, round 2 of 3, for Stage 10 optimized-vLLM, Qwen/Qwen3.6-35B-A3B. This is a review of the prior findings and remediation delta, not another whole-stage acceptance review. The inspected live tree is based on `8fbe38e7ce079a19153b1baad80f270f3d0c3025` on `main`.

Prior finding disposition: **RW1 closed; RW2 open; RW3 open.** RW3's missing repair-campaign evidence is now supplied, but the performance requirement remains failed and unwaived. No new defect was found in the retained capacity-dispatch change. Both outstanding floors now have explicit, scoped AutoFix failure records; a failed repair campaign is not stage acceptance. This round cannot return `pass-with-carried`, and neither failed required gate is an optional optimization note.

Paths below use `M = models/autoports/qwen_qwen3_6_35b_a3b`, `D = M/doc/optimized_vllm`, and `R = M/readiness_vllm`.

## Required Work

- **P1 — RW2 remains open: primary single-user FUNCTIONAL floor.**
  Evidence: the unchanged raw `R/optimized_acceptance_perf/vllm_result.json` gives `mean_tpot_ms = 31.426704362196126`, or **31.8200721423 tokens/s/user**, for 128 input / 128 output / concurrency 1, capacity 32 and context 262144. Recomputing the inherited useful-work floor gives `1,107,522,496 / 512e9 / 0.10 = 0.02163129875` seconds, or **46.2293092781 tokens/s/user**. There is no new selected-path measurement or applicable owner revision. `D/AUTOFIX.md`, `D/autofix_failure_audit.md`, and the current limitations retain the thirteen-hypothesis campaign's failure and the unwaived requirement.
  Why this matters: the previous improvement does not meet the required floor. RW1 changes dispatch for other accepted capacities; the selected capacity-32 operation sequence remains unchanged, so it cannot close this performance deficit.
  Required next step: retain this as unresolved acceptance after the recorded AutoFix failure. Future passing acceptance needs a correct selected-path result meeting the unchanged floor, or an explicit applicable owner revision. This review does not request another unchanged benchmark or reopening arbitrary optional searches.

- **P1 — RW3 remains open: canonical P4 batch-32 FUNCTIONAL floor. The repair-campaign subgap is closed.**
  Evidence: `M/doc/eval_readiness/long_answer_load.json` still reports **2.36 tokens/s/user** in the first bucket for 32 users ×6000 outputs; the raw request records contain 32 error-free 6000-token completions with maximum prompt length 74. Re-summing `D/eval_batch32_necessary_floor.json` gives **10,745,186,240 useful bytes/rank/step** and the necessary minimum **4.764924390924285 tokens/s/user**. `D/stage_check.json` preserves exit 2. The new `D/p4_floor_repair/AUTOFIX.md` and `serving_disposition.json` are backed by completed component and real-serving experiments, detailed below. The temporary candidate's separate 32-user ×1000-output diagnostic also fails its first-bucket floor at **2.56 tokens/s/user**; `R/p4_repair_candidate_01/records/p4_first_bucket.execution.json` preserves exit 2. The canonical 6000-token result was not replaced. The current limitation has `original_model: "not_run"`, `fix_hours: 0.5786`, and `gate_waived: false`.
  Why this matters: the new evidence establishes a real batch-32 repair attempt and its failure to restore the contract. It does not satisfy the probe's waiver conditions: the recorded interval is below two hours and there is no matched original-model control. Component speedups, 17 exact greedy trajectories, successful HTTP completion, and clean resource release do not satisfy the performance gate. The prototype was reverted and its full selected-path acceptance was not run.
  Required next step: preserve the explicit failed-campaign disposition and the open canonical gate. Future closure needs a qualifying final selected-path result or a genuine disposition permitted by the governing contract, supported by its actual prerequisites. No waiver follows from AutoFix failure or elapsed optional-search time. An unchanged expensive rerun would not resolve this item; this review does not request one merely to reconfirm the established failure.

## Other Concerns

- **RW1 — closed, with focused affected-path evidence.** `tt/model.py:440` now selects the fused hidden mask only for rows 2, 4 and 32. Other accepted row counts reach the existing generic extraction and post-layer activity masking; routing-mask handling is retained. `tests/test_vllm_adapter_host.py:50` extracts and executes the actual `QwenModel.decode` dispatch for every prepared bucket at capacities 1–32. `D/capacity_dispatch_host.xml` records eight tests, zero failures/errors/skips. The full40 capacity-8 server used context 262144 and budget 2048 with trace allocation tracking enabled and program-cache allocations included. `R/optimized_capacity8_repair/lifecycle_summary.json` passes all 23 checks, including 95 replays of its fixed 8-row graph, all prepared decode buckets, mixed chunk continuation, page-only updates, and retained trace identities. I independently compared the raw `lifecycle_requests.json` against `optimized_acceptance_async_2048`: all 17 greedy token arrays are exactly equal, and all 18 requests contain 96 output tokens. The sampled row differs in top-k (control 40, repair 64) and is correctly excluded from matched-trajectory claims. The original wrapper failure remains preserved; `repair_acceptance.json` is a valid CPU reassessment of existing evidence, not a fabricated rerun. Shutdown records show six releases, no remaining owned PIDs, and no device holders. This meets round 1's requested representative-capacity validation. No capacity-32 rerun is needed for this branch restriction.

- **Prior concern: mixed-traffic ITL — open, advisory, unchanged.** The original three-budget result remains 25.6824 / 92.4546 / 186.7118 seconds against the 1-second target. Budget 2048 remains explicitly diagnostic. The new capacity-8 and candidate lifecycle maxima, about 52.39 and 52.86 seconds respectively, do not close that concern. The 16384-token background values remain caps; completion of the packaged chunk probe is not proof that every background request produced a full 16384-token answer. No new required experiment follows from this advisory finding.

- **Prior concern: load forecast — open, advisory, with a useful clarification.** The canonical default plan remains 15.08 hours against 14.4 hours; it is not measured eval accuracy or release-job duration. The updated docs correctly add a cache-capacity caveat: `32 × (74 + 12000) = 386368` logical tokens exceeds the observed 263168-token pool. The candidate's 13.66-hour extrapolation from only 1000 outputs is not promoted. Stage 11 still needs its actual release plan and measured answer lengths.

- **Prior concern: bounded quality evidence — open as an evidence limit, not a demonstrated TT regression.** The retained shared suite is unchanged, so this round does not re-review it. I read all four new candidate P4 sample entries; every `answer_tail` is empty. The installed `run_long_answer_load.py:312` saves only `rec["text"][-400:]`, while `served_probe.py` collects reasoning separately and counts arriving content or reasoning chunks for timing. The saved samples therefore cannot show the reasoning text or establish a completed-answer quality result. They also do not prove that no text was generated. No matched HF quality control or ten-sample blind comparison exists for this diagnostic; no TT-specific degradation claim or quality waiver is justified. The README/work log correctly keep this separate from qualitative acceptance.

- **Prior concern: untried optional optimization — open as handoff work.** Existing geometry, transport, head-to-sampler, embedding and optional batched-prefill opportunities remain bounded by the owner's optional-search stop rule. The new combined candidate is preserved as a promising, unadopted patch. It has not been shown slower, numerically wrong, or incapable of future improvement. Its complete sampling/accuracy suite, full 6000-token P4 and final selected-path acceptance remain prerequisites to any future promotion. These prototype follow-ups are distinct from waiving RW2 or RW3.

- **Documentation note, not a new blocker:** the earlier P4 paragraph in `D/README.md` still says the limitation has `fix_hours: 0`. Its later remediation section, both current limitations records, and the work log correctly report the new campaign and 0.5786 hours. The checklist's earlier E1/E2 rows also retain pending-gate/cleanup wording superseded by its final-gates and repair sections. Update those stale summaries during normal documentation maintenance. Historical pending statements in the chronological work log and the diagnostic driver's `MEASURED_PENDING_DISPOSITION` status are superseded by the explicit final disposition; they do not establish an active job or acceptance. I also inspected the final `acceptance_resume_state.json` correction: old recovery PID/session fields are now under `historical_recovery`, `active_processes` is empty, cleanup is recorded as passed, and the completed P4 campaign/retained capacity-only source delta are explicit. This bookkeeping agrees with the saved cleanup evidence and introduces no acceptance claim.

## Hard-Check Gaps

- The capacity repair has the focused dispatch, representative serving and raw output evidence requested in round 1. No new broad sweep is requested.
- The P4 prototype's component and mixed-serving checks have limited scope. They do not replace whole-model accuracy, the full sampling suite or canonical long-answer acceptance. Because the prototype is not selected, those unrun gates do not invalidate the restored implementation's unaffected evidence.
- P4's stored request summaries do not contain full raw SSE/timestamp arrays or retained reasoning. This limits independent reconstruction of exact bucket timing and content. The emitted failing result is not being relabeled as a pass; a stronger artifact format alone is not new Required Work.
- Combine raw payloads and build/server logs remain local under ignore rules, with the checked-in summary linking exact paths. I inspected the raw baseline/candidate/restored payloads here. A later reviewer of only the commit will need those persistent local artifacts to repeat the full per-case comparison.
- The installed stage check still exits 2. No new checker run is needed to demonstrate the same canonical unwaived failure, and no passing checker result is claimed.

## Anomaly Ledger

- Observed anomaly: accepted capacities 3 and 5–31 reached an unsupported fused-mask helper.
  Evidence: `tt/model.py` dispatch delta, host test/XML, capacity-8 lifecycle and raw token arrays.
  Affected path: prepared decode buckets outside 2/4/32.
  Control or comparison: actual dispatch across capacities 1–32; retained capacity-32 greedy outputs.
  Likely subsystem: model dispatch selection.
  Investigation performed: narrow guard repair plus tracked full40 representative serving.
  Resolution: **fixed; RW1 closed.**

- Observed anomaly: capacity repair wrapper recorded `AssertionError('top_k')` after successful serving.
  Evidence: `optimized_capacity8_repair/execution.json`, raw request settings, and `repair_acceptance.json`.
  Affected path: postprocessing comparison of sampled trajectories.
  Control or comparison: control top-k 40 versus repair top-k 64; all greedy settings and 17 arrays match exactly.
  Likely subsystem: evidence comparator.
  Investigation performed: exclude the unmatched sampled row from exact cross-capacity claims; preserve original failure; correct driver.
  Resolution: **controlled; no hardware or greedy correctness failure established.**

- Observed anomaly: first B32 norm probe failed on 2175 wide-input words for the finite-edge fixture.
  Evidence: `p4_batch_norm_01/acceptance.json`, corrected `p4_batch_norm_02/acceptance.json`, probe code and `NORM_PLAN.md`.
  Affected path: temporary candidate's packed input transport.
  Control or comparison: independently recorded exhaustive finite BF16 tilize mapping; singleton production oracle.
  Likely subsystem: known signed-zero/subnormal transport mapping at the added packing boundary.
  Investigation performed: explicit exact mapping check only at `wide`; unchanged exact checks for every downstream arithmetic intermediate and final physical output. Corrected eager/changed/unchanged trace checks pass on all four ranks; the positive allocation control rejects a live allocation.
  Resolution: **controlled within the component fixtures; not general whole-model equivalence or adoption.**

- Observed anomaly: P4 remains below its necessary floor despite component improvements.
  Evidence: canonical 2.36 and temporary candidate 2.56 tokens/s/user; candidate execution exit 2; `p4_floor_repair/AUTOFIX.md`.
  Affected path: full40 B32 serving decode.
  Control or comparison: identical API server argv, eval profile, capacity, context and budget; candidate diagnostic has a shorter output cap. Combine baseline/candidate/restored have 36 exactly equal case payloads across seq32/33/64. The untracked norm component median is 1.3598291 versus 0.4645922 ms, including pack/split.
  Likely subsystem: remaining batched graph costs; no sufficient low-level attribution is established.
  Investigation performed: isolated diagnosis, two component discriminators, 23 tracked mixed-serving checks, 17 exact greedy control arrays, then untracked 32×1000 diagnostic. Both servers close cleanly. Native and Python candidate changes are reverted; rebuild/install logs and 36 exact restored cases support restoration.
  Resolution: **more-work-needed for acceptance; applicable new AutoFix campaign completed and failed to close RW3.**

- Observed anomaly: four candidate answer tails are empty at the 1000-token cap.
  Evidence: candidate `answers_sample.json`, request summaries and installed collector code.
  Affected path: diagnostic evidence retention for reasoning output.
  Control or comparison: separate content/reasoning collection; only content tails persisted; no matched HF sample set.
  Likely subsystem: artifact retention plus an output cap; generated reasoning content is unavailable here.
  Investigation performed: inspect all saved samples and collector fields; preserve the unchanged shared quality evidence separately.
  Resolution: **controlled as an evidence limitation; neither quality pass nor demonstrated TT regression.**

## Scope Inspected

- Goal/skill paths: `D/stage_review_round2_brief.md`, `D/stage_review_contract.md`, `/home/ttuser/qwen36-35b-a3b/multigoal_logs/10-10-optimized-vllm.prompt.txt`; installed tt-model-bringup 0.1.24 `stage-review`, `model-bringup`, `optimize`, and relevant `vllm-integration`, `tt-enable-tracing`, `tt-device-usage` requirements; `vllm-integration/references/eval-readiness-load.md`; installed tt-autodebug 0.1.8 `autofix/SKILL.md`.
- Artifact paths: remediation sections of README/work log/checklist/manifest/limitations; final `acceptance_resume_state.json` correction; `capacity_dispatch_host.xml`; `R/optimized_capacity8_repair`; `D/p4_floor_repair`; `R/p4_batch_norm_01`, `p4_batch_norm_02`, `p4_combine_01`, and `p4_repair_candidate_01`; unchanged floor/result/check files only to confirm the prior finding's continuing status and compare raw controls.
- Code paths: active `tt/model.py` branch and `tests/test_vllm_adapter_host.py` delta; new capacity/candidate drivers and component probes; preserved `combined_candidate.patch`; installed P4 collector and checker waiver logic. `git diff HEAD` is empty for the reverted `tt/mesh_compensated_norm.py`, `tt/multichip_decoder.py`, and native combine directory. The prototype ownership header is contained in the patch, not active source; its host reproducer requires applying that patch.
- Commands run: read-only `rg`, `cat`/bounded `sed`/`tail`, `git status`, `git diff --cached`, `git diff HEAD`, and standard-library Python parsing of JSON/JSONL/XML, raw greedy-array equality, complete combine-case equality, server-argv comparison and floor arithmetic. No tests, servers, TTNN import, hardware access, signals, reset, profiler, source mutation, commit or push were performed. The only review write is this report.
- Push state: the supplied run plan authorizes local checkpointing without pushing; this review does not infer push status or authorization from repository history.

## Residual Risk

- The selected implementation still misses two required performance floors. The recorded failed AutoFix campaigns support an honest failed/incomplete handoff under the original stop condition; they do not make the stage complete.
- Capacity 8 is representative device evidence for the restored generic path. The host dispatch check covers the full accepted range, but this is not a claim that every capacity received a hardware run.
- The prototype's numerical and sampling acceptance remains incomplete and its component performance does not establish a final serving improvement. Reapplying it requires the recorded follow-up validation.
- Original whole-stage evidence outside the remediation delta was not re-reviewed. This verdict closes only the findings and changes stated above.
