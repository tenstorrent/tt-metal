# Stage6 independent follow-up brief

Round2 of3. Original goal: `stage_goal.txt`; explicit updated owner contract: `owner_decision.md`. Base `35169ad54c7acbf0016ab155c5a4c08f9b1ec016`, branch main, live worktree. Read-only inspection; only reviewer report may be written. No hardware, target imports, tests, serving, vLLM or resets.

Changed files: root and full-model README; limitations.json; acceptance_checklist.json; cap_wrap_up.json; runtime_fallback_audit.md/.json; serving_contract.json; work_log.md; declaration-only tt/serving_capabilities.py; new owner_decision.md and owner_acceptance.json, CPU validation logs. Full inference source, saved numerical results and previous review are unchanged. Inspect git diff against base.

New evidence: `logs/owner_acceptance_stage06.log` and exit_code0; `logs/owner_acceptance_precommit.log`; owner_acceptance.json re-derives7top1cases/10controls from existing raw result. Exact commands are in work_log.md; no hardware evidence was regenerated.

Check each prior finding closed/open under the explicit owner decision. Inspect only changed files for new issues. Treat docs as claims. Read the source decision and HF control, raw TT cases/controls and actual declaration. Do not demand a numerical repair or rerun that the owner explicitly excluded. Record preserved risk faithfully; no claim that PCC improved.

## Previous findings verbatim

## Required Work

- **P1: Required resumed-prefill acceptance is still failing; the corresponding serving contract remains incomplete.**

  **Evidence:** [original goal](stage_goal.txt) explicitly requires the resumed-prefill check to pass and no missing serving-contract rows. [The actual failed check](../../readiness/prefill_resume_check.failed.json) contains seven all-40-layer/context-2048 cases, of which three fail the unchanged installed criteria:

  | Length / resume start | Mean PCC | Minimum row PCC | Failing criterion |
  | --- | ---: | ---: | --- |
  | 257 / 128 | 0.9935122896 | 0.8585700580 | Minimum >= 0.98 |
  | 1025 / 260 | 0.9931221526 | 0.9686653286 | Minimum >= 0.98 |
  | 1025 / 1024 | 0.9870205088 | 0.9870205088 | Mean >= 0.99 |

  All seven top-1/final-token criteria pass; the first failing case has 64/65 top-1 agreement. All ten applicable absent-history/fresh-suffix controls fail as expected. Those useful controls do not change the three failed PCC results. No passing `readiness/prefill_resume_check.json` is supplied. [AutoFix](AUTOFIX_RESUME.md) explicitly concludes that it failed to produce a verified policy-preserving repair. [Reduced accepted-decoder parity](prefill_equivalence/resume_legacy_parity_drain.json) is exact on real layers 0/3, but compares bounded versus accepted continuation, not uninterrupted versus resumed all-40 computation.

  [The corrected serving manifest](serving_contract.json) still marks `prefill_logits_continuation`, `supports_chunked_prefill`, `chunked_prefill`, and `resumed_prefill` as `missing`. The [single installed gate log](logs/stage06_final_gate.log) and [exit code](logs/stage06_final_gate.exit_code) record exit 2. Inspection of the installed checker confirms that any missing row is critical. The later schema correction fixes representation issues; it cannot resolve these four semantic rows. This is one underlying acceptance problem, not four independently diagnosed bugs.

  **Why this matters:** Genuine continuation is an explicit stage requirement for chunking and retained-state resumes. The allowed `recurrent_state` exclusion for automatic KV-only prefix caching is correctly documented, but does not exclude the separately required retained recurrent/KV continuation path. A passing teacher-forcing or maximum-context run does not exercise the failing full-versus-split equivalence requirement.

  The [original-HF control](qualitative/hf_resume_control/result.json) fails the corresponding case-level numerical criteria too; this supports the recorded numerical-sensitivity limitation. It does not establish exact failing-row correspondence for the two multirow TT cases, whose saved evidence contains aggregates only. More fundamentally, the explicit user requirement for a passing resumed-prefill check has not been amended. The review neither invents a stronger threshold nor substitutes the general layer PCC rule for the actual installed resume criteria.

  **Required next step:** Keep the stage/checkpoint explicitly incomplete under the current cap; do not relabel this finding as carried or begin the next stage as a passed hand-off. If work is resumed, resolve the existing continuation contract and obtain passing evidence for the affected required resume check and serving rows, then review the affected changes. The already documented next diagnostic is full-depth localization of the original 1025/start-1024 case while preserving policy and genuine retained state. It is an untried hand-off, not a proven fix or an instruction to run another experiment during this capped wrap-up. No additional speculative sweep or new acceptance format is requested by this review.

## Other Concerns

- **HF continuation sensitivity is a controlled baseline limitation.** The pinned BF16 CPU HF control reports mean/minimum PCC of 0.98702738/0.49634269, 0.99161743/0.95285638, and 0.98846549/0.98846549 in the three corresponding cases. [The limitation record](../eval_readiness/limitations.json) contains `original_model: "same"` and the evidence paths. The control is one deterministic measurement per case reused under ten seed labels, explicitly not ten independent sampled trials. It supports a numerical observation, not a statistical quality waiver or an identified TT root cause. No additional qualitative-control campaign is required merely because those numerical records reused seed labels.

- **Qualitative evidence is a useful smoke with a limited claim.** I read all six actual HF/TT completions and rendered prompts under `qualitative/tt_greedy/q01` through `q06`, rather than relying only on the verdict. Both sides spend the 128-token cap on relevant reasoning. Both have the isolated q03 phrase “thinking thinking”; both reason in English toward the French translation. Neither observation establishes TT-specific mechanical looping or wrong-language degradation. First mismatches independently recompute to 64, 85, 53, 107, 19, and 74; q05 first changes Markdown formatting. These samples do not establish final-answer task accuracy or distributional quality. The descriptive wording in [the paired review](qualitative/tt_greedy/paired_review.md) is appropriate. No content failure is selected for further debugging from this evidence.

- **Performance alternatives remain hand-offs.** The inherited decoder rejection ledger and stage-5 owner-accepted linear performance limitation remain intact. Embedding/head distribution alternatives and further terminal tuning in [preserved policy](preserved_policy.md) are unmeasured candidates, not rejected or demonstrated improvements. No extra performance experiment is a prerequisite to closing this review.

- **Some documentation retains historical wording.** The opening comments in `tt/serving.py` and `tt/serving_state.py` still say hardware validation is pending, while their actual reduced probe artifacts exist. Some earlier paragraphs in the runtime/memory audits also retain pending-performance language before later completed updates. The final README and concrete artifacts give the correct outcome. These are documentation cleanup notes, not reasons for another runtime run or review round.
