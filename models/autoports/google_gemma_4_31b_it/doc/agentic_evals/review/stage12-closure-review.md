# Stage Review

Verdict: more-work-needed

Independent stage-12 closure snapshot, 2026-11-15, live `v5-extended-run` at supplied HEAD `8a68d14dbf6788609fe0356d2a8c67e2b37b08d1`. The worktree and active jobs can advance after this inspection. This verdict distinguishes unfinished required investigation and validation from the retained below-band score. A below-band result at the time limit is **not passed yet**; it is not grounds to discard the list or invent a new score.

## Required Work

- P1: Complete the current production Q32 correction's acceptance and precision disposition.
  Evidence: `setup/p19-resume-investigation/prefill_resume_after_bf16_ccl.json` has nine failing cases after the CCL-only correction. `prefill_resume_q32_diagnostic.json` passes all fourteen under an explicit post-construction geometry override. Current `tt/multichip_decoder.py` now selects Q32 for HiFi4. `setup/p19-q32-production-validation/candidate-admission.json` and command records are still prepared/not admitted, and do not supply production-path results. Historical P19 accuracy and fixed108 outputs used Q128. This is an unselected candidate: the selected head-BFP8/HiFi2 serving geometry is unchanged by the HiFi4-only Q32 branch.
  Why this matters: the production change affects fresh as well as resumed attention arithmetic. Neither the diagnostic override nor old Q128 acceptance proves current-policy accuracy, free-running behavior, or serving readiness.
  Required next step: finish the already prepared unmodified production resume check, then exact-source broad acceptance and applicable fixed108 anomaly comparison; review original thresholds, negative controls, runtime policy, nonaligned/context cases, actual qualitative replies and trace-allocation evidence. Record a final candidate disposition. If adopted as the precision repair, validate its serving/memory path, remeasure the stage-11 rows used by the eval and rerun the retained list within the allowed window. Do not transfer diagnostic teacher-forcing speed into vLLM rows. If the remaining window cannot honestly fit a scored rerun after validation and closure reserves, do not launch one or adopt an unmeasured policy: retain the unselected candidate and unfinished remediation, and report the extended result as not passed yet. Do not claim precision repair or release readiness. Historical task-duration rescheduling, rather than a B1 autoregressive/vLLM speed ratio, is the appropriate current forecast basis.

- P2: Finish the original-history Sklearn conditional control and final anomaly disposition.
  Evidence: `setup/swe85-conditional-hf/sklearn13135-full-missingtool-conditional-history-b/hf-refresh-c-after-tt-review/` contains a command/admission and active log but no terminal completion or terminal review at inspection. The separate HF22 batch is already complete and independently checked: 22 hashed raw result files, 1,370 positions, weighted top-1 agreement 0.9861313869, no hash mismatches. HF22 is no longer pending work.
  Why this matters: the original missing-tool switch was not retained by earlier suffix controls. Its admitted exact reconstructed-history comparison remains unresolved; high HF22 agreement on other forced histories cannot close this case.
  Required next step: collect the active job's actual terminal outputs, compare matching TT/HF IDs and positions, classify the missing-tool onset and terminal window, and record its limited conditional scope. A timeout or incomplete result must remain explicit, with the evidence-backed disposition rather than an inferred pass.

- P2: Reconcile final stage artifacts and commit boundary after active work terminates.
  Evidence: the review entrypoint still says Q32 diagnosis is pending and HF22 unrun; newer terminal records contradict that status. Canonical files are being refreshed. The telemetry packet has the required IDs, 4096/128 B1/C1 workload, measured selected-head serving rates and null unavailable device metrics, but still names final review and commit as pending. `git status` includes unrelated staged benchmark report/usage files and root debug artifacts.
  Why this matters: closure requires final results, learned evidence, usage provenance and an isolated stage-owned local checkpoint, not a collection of mutually stale live snapshots.
  Required next step: reconcile REPORT/results/learned/anomaly disposition and usage/telemetry from terminal records; retain all first outcomes and the unknown verifier bounds; refresh the compact inventory; submit terminal additions for independent closure review; then commit only stage-owned files locally. Never push. This item does not demand a new device run to repair wording.

## Other Concerns

- The three Requests verifier outcomes remain unknown. I independently recounted 85 rows: 23 successes, 59 zero outcomes, three nulls, with every coordinator path present. Preserve the pending operator question and denominator. Even all three passing gives 26/85 and cannot reach the reference band. Faithful unresolved outcomes may be retained as unknown; do not invent a score or treat a pending operator answer as permission.
- Phase 2's missing prelaunch forecast is disclosed. Its post-hoc replay must not be relabeled as that forecast. Phase 2 began at hour 6.57, before the twelve-hour boundary.
- The uniform reconstructed prompt-count gap and different HF/TT RNG streams remain material scope limitations. Conditional likelihood alignment establishes behavior on saved histories, not native free-running incidence or task correctness.

## Hard-Check Gaps

- A mechanical tail detector misses the directly observed Pytest caret continuation because its minimum character threshold is unmet. The raw HF output includes a traceback fragment followed by repetitive carets despite `flag: false`. The semantic classification is necessary and present; do not replace it with the boolean.
- This inspection rederived aggregate counts and HF22 result hashes/weighted agreement, sampled raw controls, and inspected the current production diff. It did not redo every trajectory audit or repeat hardware tests. Prior scoped reviews are supporting evidence, not newly executed checks.

## Anomaly Ledger

- Observed anomaly: P19 resumed-prefill divergence.
  Evidence: actual CCL-only and Q32 diagnostic resume JSONs, including per-position thresholds and explicit override metadata.
  Affected path: HiFi4 fresh/resumed attention and Ring prefill reduction.
  Control or comparison: CCL-only correction leaves nine failures; uniform Q32 diagnostic passes fourteen with original thresholds.
  Likely subsystem: partition-dependent attention arithmetic plus reduction phase; CCL alone is insufficient.
  Investigation performed: inspected actual case results and production geometry/reduction diff, prepared acceptance commands and source-pinned independent preparation review.
  Resolution: more-work-needed, production acceptance remains unfinished.

- Observed anomaly: Requests5414/Pytest7432 repeated caret tails and other full-TT conditional motifs.
  Evidence: actual `p19-native-onset-caret-controls/results` HF private outputs; HF22 raw result files and manifest.
  Affected path: reconstructed diagnostic histories with full TT and pinned BF16 HF reference.
  Control or comparison: both inspected free HF outputs visibly repeat carets; Pytest detector is false while the text is visibly repetitive. HF22 has matching hashed records and 98.61% weighted conditional top-1 agreement.
  Likely subsystem: reference continuation behavior on these conditioned inputs, with bounded numerical differences.
  Investigation performed: read actual text, checked reference revision and forced-history identity metadata, recomputed weighted agreement across all22 manifest rows and checked all raw hashes.
  Resolution: controlled in the documented conditional/reconstructed scope. No native RNG replay or equal-incidence claim.

- Observed anomaly: Sklearn13135 missing-tool switch.
  Evidence: original-history full-TT controls and active HF refresh admission/log.
  Affected path: original reconstructed long history, not the previously shortened suffix.
  Control or comparison: full-TT control collected; matching HF terminal comparison not yet available at inspection.
  Likely subsystem: unresolved pending reference comparison.
  Investigation performed: checked separate job identity and distinguished it from the completed22-window batch.
  Resolution: more-work-needed pending terminal classification.

- Observed anomaly: extended SWE below reference band.
  Evidence: `current/swe-extended85-head-bfp8/collected-final/final-score-with-adjudication.json`, retained first outcomes and task coordinator links.
  Affected path: selected head-BFP8 serving policy, frozen85-task list.
  Control or comparison: served/full precision work and candidate matrix; current P19 remediation remains unselected.
  Likely subsystem: mixed wrong patches, fetched solutions, anomalies and verifier infrastructure; not a single scalar precision diagnosis.
  Investigation performed: recounted causes (26 wrong,22 fetched-solution,11 primary anomaly,3 infrastructure,23 successes), checked null grades and timing/phase records.
  Resolution: score remains not passed yet; outstanding remediation is listed above. Completed diagnosis does not convert the score into a passing result.

## Scope Inspected

- Goal/skill paths: bringup/run_contract.json; bringup/agentic_evals/inputs.json; supplied AGENTS instructions; installed stage-review, agentic-serving, datatype-sweep and tt-device-usage skills.
- Artifact paths: active REPORT/results/learned; review entrypoint, interim review and closure index; final85-task adjudication; HF22 root/independent reviews and raw22 results; original-prompt caret HF outputs; CCL/Q32 resume JSONs; production validation package; Sklearn HF refresh directory; mandatory telemetry packet.
- Code paths: current diffs in tt/multichip_decoder.py, tt/optimized_decoder.py, tt/kv_pool_budget.py, tt/generator_vllm.py; prepared runtime commands and host geometry regression assessment.
- Commands run: local cat/sed/rg, git status/diff and short standard-library JSON/hash/arithmetic analyses. No network, model execution, TTNN imports, hardware access, reset, server, container or implementation mutation. Only this review was written.

## Residual Risk

This is not a clean-pass or release-ready verdict. Active CPU HF and planned production validation can resolve findings, but terminal evidence must be inspected before changing the verdict. The stage deadline remains 2026-11-16 01:15:25 UTC. A completed time-bounded report can truthfully retain an extended phase as not passed yet; it cannot claim extended-mode readiness or conceal unfinished correctness work. Unknown verifier grades alone do not justify discarding85 tasks or fabricating a point score.

## Independent clarification: runner-owned usage

The snapshot verdict and its correctness/validation findings remain unchanged. Final stage usage is **not a missing pre-closure gate while this stage is active**. I inspected `review/stage-usage-refresh-instructions-20261115.md`, the installed `scripts/multigoal` implementation (`StageUsage.write` and its terminal call sites), and the current `/home/mvasiljevic/v5-agentic/logs/manifest.txt`. The manifest corroborates stage thread `01a1d731-c582-76d3-b263-350c027e1a66` and has no terminal usage-write entry yet.

At terminal closure the runner computes the actual end time, main/round thread usage and device timeline, writes `doc/agentic_evals/stage_usage.json` plus the REPORT usage section, and locally commits those paths. Exact current-run provenance and this documented pending runner write suffice before agent closure. Do not invent final values, manually rewrite usage, replay guessed runner state or reuse v3 IDs. Verify the runner write and local commit afterward; an actual post-closure write failure requires recovery using current-run provenance. This explicitly supersedes any implication in Required Work item three that final usage must already exist before the agent closes. Other final artifact reconciliation, validation and review requirements remain.

## Terminal evidence update: production resume, broad acceptance and HF2

Verdict remains **more-work-needed**. This update supersedes the earlier pending status of these completed checks while preserving the original snapshot.

- Production resume is resolved: independently recounted the raw `setup/p19-q32-production-validation/resume/prefill_resume.json`:14/14 pass, all top1=1,24/24 applicable negative controls fail as expected. Its SHA matches the root record. Unmodified production geometry and tracking=1 completed in272.732s.
- Production broad acceptance is resolved for its declared scope: `acceptance/completion.json`, `root-review.json` and `independent-review-d.json` record965.306571s exit0. I verified all11 raw artifact hashes listed by root, inspected accuracy counters and actual paired qualitative outputs, and inspected the long-prefix response. AIME prefill100/100, decode98/100 on each of three repeats, natural466/512 top1 and512/512 top5 meet the configured bars. The six shared outputs are coherent bounded continuations; two naturally stop and match HF exactly, while four reach128 tokens. The32,083-token prompt retrieves ORCHID-742 inside fenced JSON, not strict raw JSON. Boundary execution at262143/262144 is distinct from semantic long-context accuracy. Tracking=1 AR18.503796tps and TTFT1167.1495ms are diagnostic, not vLLM rates or a speed comparison to tracking=0.
- Original-history Sklearn HF2 is resolved within its conditional scope: the two terminal raw HF files match the independent review's hashes and pinned revision, with63/64 top1 agreement in each window. The channel marker is forced token0 at original source190; it is not preconditioned. HF assigns0.135294 and TT0.161622, both preferring newline107. The terminal turn marker is top1 in both. The corrected independent review excludes the347-token intervening middle; no continuous replay, native incidence, task correctness or spontaneous HF missing-tool behavior is claimed. This closes Required Work item two's pending-terminal-control requirement, not the overall candidate disposition.
- Fixed108 remains active and requires terminal semantic review. I checked its admission and rehashed all130 pinned inputs/source files with zero mismatches. It uses the original case list/build kwargs and a fresh output directory. Admission and preserved identities are not panel acceptance.

Remaining required work is the terminal fixed108 comparison and candidate disposition, any still-required validation following that disposition within the time bound, and final canonical artifact/independent closure reconciliation. P19 remains unselected; selected serving policy, serving rates and task scores remain unchanged. The earlier runner-owned usage clarification continues to apply. No device/model commands were run for this update.
