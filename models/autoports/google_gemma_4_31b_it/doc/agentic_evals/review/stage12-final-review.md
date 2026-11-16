# Stage Review

Verdict: more-work-needed

Independent review of stage 12, google/gemma-4-31B-it, live branch `v5-extended-run`, supplied HEAD `8a68d14dbf6788609fe0356d2a8c67e2b37b08d1`. Inspection is host-only. Final canonical comparison includes the terminal full-BF16 panel, resume and broad-output evidence. Later addenda below supersede explicitly earlier snapshots.

## Required Work

- P1: Authorize a continuation window for the uncompleted precision-policy serving and evaluation gates.
  Evidence: P19's terminal production-Q32 fixed108 panel is rejected with two protocol anomalies. Full-BF16 has now completed the fixed108 panel and all cap extensions, passed the14-case resume check, and produced passing broad component results, with the wrapper143/unknown child-status qualification preserved. KV-only BFP8 passes only two current protocol controls; it has not earned full candidate acceptance. No repaired candidate has measured-free-memory serving sizing, matched vLLM rate rows, or a retained85-task scored rerun. `continuation-proposal.json` is explicitly `proposed_not_authorized` and requests12 additional local hours.
  Why this matters: completed diagnostic prerequisites do not select a production policy or demonstrate reference-band task quality. The retained SWE85 result remains below band even if all three unknown verifier outcomes pass. Diagnostic replay rates and memory inventories cannot replace matched serving measurements. No evidence-backed plan admits the entire remaining chain in the current window.
  Required next step: **human authorization** of the additional local window proposed in [continuation-proposal.json](../continuation-proposal.json). The12hours is a proposed authorization budget, not a guaranteed runtime forecast. After authorization, establish remaining candidate acceptance/disposition, measure free memory and validate serving/pool sizing without reducing262144context, remeasure relevant vLLM rows, and admit the retained85-task rerun only if its measured projection fits the available budget and12-hour run limit. Preserve all first outcomes, fetched-solution penalties and unknown verifier grades. Until then, perform no further model/device job, keep the selected head-BFP8 policy and report extended SWE **not passed yet**. Do not repeat completed full-BF16 diagnostic checks solely for stronger formatting or to manufacture a child exit0 record.

All remaining substantive required work depends on that new authorization. This is **more-work-needed**, not clean-pass or extended-mode release readiness.

## Other Concerns

- Extended SWE remains not passed yet. I recounted85 adjudicated rows:23 confirmed successes,26 wrong,22 fetched-solution failures,11 primary anomalies and3 unknown verifier outcomes. The point score stays null; bounds23–26/85 and Wilson envelope18.760–41.046% remain below64.8%. No denominator change or fabricated grade is justified. A faithful future verifier recovery may resolve unknowns; they cannot rescue this band even if all three pass.
- The parent reports historical task rescheduling estimates of about4.95h at8agents or5.51h at7agents at unchanged per-call duration, before validation/rates/closure. These are planning projections, not a minimum-time proof or a waiver. No matched current-candidate rates support an admitted rerun in the remaining window.
- Diagnostic histories reconstruct canonical prompts and use new seeds42/43/44. Conditional HF agreement establishes behavior on those histories, not native token/RNG identity, spontaneous incidence or task correctness. Historical TT comparators preceding Q32 remain historical.
- The live index includes unrelated staged benchmark REPORT/stage_usage paths. Exclude them and unrelated root debug artifacts from any stage-owned checkpoint. No clean-pass checkpoint is authorized by this verdict; any evidence-only checkpoint must be clearly labeled as such and follow the owner's applicable instructions. No push, CI or publication is authorized.

## Hard-Check Gaps

- Final runner-owned usage is not a pre-closure requirement. I inspected installed `scripts/multigoal` `StageUsage.write`: terminal time, thread-family usage and device timeline are assembled and written to REPORT/stage_usage, then committed locally at terminal call sites. Preserve current-run provenance and verify that official write after closure; do not fabricate final usage or reuse v3 IDs.
- Telemetry uses exact4096/128 B1/C1, selected head-BFP8 vLLM TTFT755.631794ms and decode34.770637 tokens/s. Unavailable device metrics are null, appropriately avoiding vLLM profiling and host-wall substitutes. Packet status should retain the independent more-work-needed verdict when closure is recorded.

## Anomaly Ledger

- Observed anomaly: long-prefix P19 resumed prefill differed from fresh execution.
  Evidence: production `resume/prefill_resume.json`; current HiFi4 Q32 and BF16 ring-phase code changes.
  Affected path: fresh/resumed prefill attention partitioning and ring reduction phase.
  Control or comparison: identical-input partition diagnostics and production negative controls.
  Likely subsystem: query partition geometry and reduction rounding.
  Investigation performed: independently recounted14/14 passing cases, top1 agreement1.0 throughout, minimum PCC0.999898151 and applicable failing negative controls; inspected source diff and unmodified production admission.
  Resolution: fixed for the measured production resume scope. This does not select P19.

- Observed anomaly: two protocol failures remain under production-Q32 P19.
  Evidence: fixed108 raw index89 and cap-extended index54, matched `full-q32-protocol-controls/runs/` outputs.
  Affected path: free-running structured tool boundaries.
  Control or comparison: same saved fixtures/seeds with all-group BF16/HiFi4.
  Likely subsystem: precision-sensitive generation; exact attribution to an individual reduced group is not established by changing both down and KV precision.
  Investigation performed: inspected actual P19 and full-control continuations, runtime policies and source pins; full controls stop naturally at64 and19 tokens with valid tool boundaries.
  Resolution: controlled diagnosis, more-work-needed for a validated serving repair. P19 remains rejected and full BF16 unselected.

- Observed anomaly: three fixed108 continuations reached caps while progressing.
  Evidence: `fixed108/cap-extensions-root/expected-prefixes.json` and raw extended runs.
  Affected path: diagnostic completion limits.
  Control or comparison: same seeds, fixtures and doubled caps.
  Investigation performed: independently compared complete original token prefixes; all three are exact. Extensions stop naturally at263,337 and1386 tokens. Index54 remains the classified protocol anomaly above.
  Resolution: cap investigation completed; no remaining progressing cap in this panel. Completion does not imply task correctness.

- Observed anomaly: caret/contraction and conditional missing-tool behavior.
  Evidence:108 raw P19 outputs contain18 caret continuations; panel identifies two additional contraction motifs. `setup/swe85-conditional-hf/` contains full-prompt and conditional reference controls.
  Affected path: saved-history diagnostic generation.
  Control or comparison: pinned HF and historical full-TT comparisons, including HF22 and Sklearn two-window HF2.
  Investigation performed: checked22 raw HF hashes and rederived1370 positions with weighted top1 agreement0.986131386861. Both Sklearn raw HF hashes match, each window has63/64 top1 agreement; channel token101 is forced at source190, while both models prefer newline107. HF gives the channel probability0.135294. The intervening unmeasured middle is not claimed covered.
  Resolution: controlled within stated conditional scope, not evidence of native incidence or a selected-policy cure.

- Observed anomaly: extended SWE below band and verifier unknowns.
  Evidence: `current/swe-extended85-head-bfp8/collected-final/final-score-with-adjudication.json`,85 linked coordinator artifacts, original per-task outcomes.
  Affected path: benchmark task quality and verifier infrastructure.
  Control or comparison: per-task served/full precision investigations; canonical score accounting preserves original outcomes and fetched-solution penalties.
  Investigation performed: recounted row causes and null grades, verified every coordinator path exists, checked measured3.523453h against explained8.789750h.
  Resolution: bounded investigation/reporting complete; extended result not passed yet, with substantive repair work above. Unknown grades remain unknown.

## Scope Inspected

- Goal/skills: AGENTS.md, bringup/run_contract.json, bringup/agentic_evals/inputs.json; installed stage-review, agentic-serving goal12, datatype-sweep anomaly rules and tt-device-usage.
- Artifacts: final-review entrypoint and prior closure review/addenda; production Q32 resume, acceptance, fixed108, cap extensions and full controls; corrected GPQA; SWE85 adjudication; HF22 and Sklearn controls; source provenance manifests; post-format host test log; mandatory current-run telemetry; installed runner usage implementation.
- Re-derived evidence: all11 acceptance artifact hashes match; all290 archived source hashes match; current provenance hashes match and non-import Python ASTs match across all snapshot/current pairs. Raw fixed108 comprises108 outputs and28,236 tokens. Host log records62 tests and60 subtests passing. Resume14/14 and GPQA157/198 were independently recounted; GPQA replaces exactly12 affected prompts unconditionally and retains186 originals.
- Acceptance outputs: inspected all six actual TT replies and paired HF evidence. Two terminate naturally and match HF; four are coherent128-token bounded prefixes. Full262144 boundary execution is distinct from semantic long-context accuracy. Long-prefix output retrieves ORCHID-742 inside fenced JSON, not strict raw JSON. AR18.504 tokens/s under tracking1 is diagnostic only.
- Code: diffs in tt/multichip_decoder.py, optimized_decoder.py, kv_pool_budget.py and generator_vllm.py, and corresponding host/provenance checks. HiFi4 alone selects Q32; established lower-fidelity serving geometry remains unchanged. BF16 ring prefill now gets absolute-row phase preservation; MLP blocking accounts for BF16 activations. Pool overrides preserve advertised context and reject conflicting/invalid values.
- Commands: read-only cat/sed/rg/git diff/status and short standard-library JSON, hashing, AST and arithmetic analyses. No device, TTNN import, server, HF execution, container, network or implementation edit. Only this review is written.

## Residual Risk

This verdict does not declare extended-mode release readiness. It recognizes completed bounded investigation and permits truthful not-passed-yet reporting; it does not convert the two-case full-BF16 control into a production repair. Later admitted work must establish the remaining candidate/serving/rate/list gates and obtain a new independent review. A prepared full-BF16 panel may resolve only part of this requirement. Do not extend the current50h budget ending2026-11-16 01:15:25UTC without authorization.

## Targeted terminal addendum: full-BF16 Q32 fixed108

Verdict remains **more-work-needed**. This supersedes the earlier snapshot's statement that full BF16 has only two diagnostic cases. Canonical final reconciliation remains pending; active later diagnostic arms are not included in this addendum.

The current-Q32 full-BF16 fixed108 panel is terminal: `setup/full-q32-production-validation/fixed108/completion.json` records4000.206804s, exit0, within the3500–6000s forecast. I independently rehashed all146 admission pins without mismatches, read all108 runtime policies against the full-BF16 config, and recounted27,077 generated tokens. Every result names its manifest fixture and respects its cap. Eighteen outputs contain caret tails; final adjudication is90 normal,18 controlled motifs,0 other anomalies and0 inconclusive. This means no additional protocol defect in this saved-history panel, not zero anomaly incidence or task correctness.

All six originally progressing caps are resolved. I compared all six first-extension token prefixes against the original256/512/1024-token prefixes and both further Requests2931 continuations against their full512-token prefixes: all are exact. Four first extensions stop naturally at327,633,1834 and1084 tokens. The two Requests continuations stop naturally at532 and773 tokens after the second extension, each with token50 and a complete tool handoff. All151 first-extension pins and164 corrected-retry pins match. No progressing cap remains in this panel.

I inspected actual full-policy54/89 outputs: both have valid `<|tool_call>` openers and complete tool handoffs, matching the earlier two-case controls. Cases105/107 start `[100,45518,107,101,48,...]`, a structurally valid thought-channel opener at the saved boundary followed by a valid tool call. No matching HF control is supplied for those two outputs, and no HF-equivalence claim is justified.

The failed first Requests extension attempt remains preserved. Its log shows a manifest-wrapper TypeError at job-name validation. Source inspection places that validation before the TTNN/model imports and device-open call. The corrected retry validates the list before model execution, completes in178.839224s, and preserves both exact prefixes. This is a resolved preparation error, not a hidden model failure or discarded evaluation outcome.

Historical evidence hook provenance is intact: I checked all six changed historical non-Python files against the exact-byte archive at `bringup/artifacts/stage12-artifact-hook-current-snapshot`. Each before/after hash matches its manifest and each change is terminal whitespace only. The final cached hook record exits0 across1721 candidates. Historical hashes continue to identify archived bytes; the active full-policy pins remain unchanged.

Remaining P1 scope is narrowed accordingly: the current full-policy fixed108 panel no longer needs rerunning merely to complete this prerequisite. Full-policy resume/broad acceptance, candidate disposition and serving/memory validation, matched vLLM rates, and the retained-list scored rerun remain unproven at this snapshot. Any group-isolation arms must be reviewed on their own terminal evidence. Completed diagnostic work does not select a policy or replace the85-task result. Admission must still fit the authorized time and closure reserves; unmet gates require an authorized continuation, not fabricated acceptance.

## Targeted addendum: KV/down isolation arms

The two terminal arms under `setup/full-q32-production-validation/protocol-group-bisect/` narrow the protocol diagnosis. I verified all160 source pins in each admission and inspected actual runtime policies and all four continuations. ArmA changes only KV to BFP8 and produces valid tool boundaries in both cases. ArmB changes down projections from BF16/HiFi4 to BFP8/LoFi with BF16 KV retained; Sympy again emits the closing marker as an opener, while Sphinx remains structurally valid. Wall times140.349800s and169.036021s are diagnostic, not vLLM performance.

This supersedes the earlier uncertainty about whether either reduced group alone can reproduce the Sympy defect: down BFP8/LoFi alone is sufficient in this exact saved-history/seed case. It does not independently identify which part of the dtype/fidelity pair causes it, explain all native anomalies, or prove KV-only policy acceptance. No full-panel/broad/serving/rate/list evidence for the KV-only candidate is supplied. The full-BF16 resume/broad work remains subject to separate terminal review. Existing P1 remains.

## Targeted addendum: full-BF16 production resume

The full-BF16 Q32 resume prerequisite is complete at `setup/full-q32-production-validation/full-bf16-resume-candidate/`. I independently rehashed all173 admission pins and recounted raw `run/prefill_resume.json`:14/14 pass, all top1 agreement1.0, minimum PCC0.999931449144, and24/24 applicable negative controls fail as expected. Lengths2560/16384 cover offsets32,64,132,1024,1029, near-end cache hits and final-token resumes, with unchanged0.99 mean-PCC/0.98 minimum-PCC/0.98 top1 bars. Completion318.855938s is within its200–700s range. The isolated resume fixture's16384 maximum is not a served context reduction or proof of advertised-context broad acceptance.

This closes the full-BF16 resume portion of P1. Broad acceptance is active and is not judged from partial output. Serving/memory validation, policy selection, matched rates and retained-list evaluation remain incomplete. The owner's final broad-job admission preserves a2700s closure reserve and prohibits subsequent experiments; no concrete review workload disproves that reserve, but this is not a guarantee of completion or permission to exceed the deadline.

## Targeted terminal addendum: full-BF16 broad checks

The recorded full-BF16 broad checks are complete at `setup/full-q32-production-validation/full-bf16-broad-acceptance-candidate/`, with an explicit process-status qualification. The outer wrapper returned143 while its child continued. The saved pidfd observer subsequently observed both child processes exit at1794786272.6902, giving1346.724652s wall; the actual exit statuses were unavailable after reaping. `completion.json` correctly keeps child exit_code null. This is not a clean-exit0 claim. The complete run log reaches both context boundaries and host compatibility calls, then normal device/cluster close messages. No Traceback, AssertionError, allocation-failure or OutOfMemory marker appears.

I independently rehashed all172 admission pins and all11 result hashes. The runtime inventory covers60 layers with BF16 weights/cache and HiFi4 kernels. Raw accuracy is98/100 AIME prefill,98/100 in each of three decode repeats,480/512 natural top1 and512/512 natural top5/top100, meeting the configured bars. Tracking1 counters record297 decode and511 natural replays without additional captures. Both262143/262144 prompt boundaries and final legal decode position execute; host sampling compatibility is equal. This is execution-boundary evidence, not semantic accuracy across the entire context.

I read all six actual qualitative completions: two naturally stop and four remain coherent bounded128-token prefixes, not completed tasks. The pinned chat/reference/tokenizer revision is unchanged. The32083-token retrieval returns ORCHID-742 inside fenced JSON. AR4096/128 B1/C1 TTFT1407.571406ms and decode16.982450 tokens/s are diagnostic under tracking1, not vLLM serving rates. Diagnostic memory inventory does not replace measured-free-memory sizing and serving validation at the intended batch.

These completed outputs close the recorded broad-check prerequisite, while preserving unknown child exit status as an execution-provenance limitation. They do not admit or select full BF16 for serving. No repeat of completed fullpanel/resume/broad checks is required solely to strengthen formatting or to manufacture an exit0 record. Remaining P1 is evidence-backed final candidate disposition/acceptance as needed, serving/pool validation, matched vLLM rate rows and the retained85-task rerun. A KV-only candidate still has only two-case current evidence and cannot inherit the full-policy panel. No further device experiment is admitted in the current window; the proposed continuation is not operator authorization. Verdict remains **more-work-needed** pending final canonical reconciliation.

## Final canonical comparison

The final comparison read REPORT SHA256 `5441ad2ebd3a22b954b121cd23f3e43c34d20434ef79c5566388f07373cf90e6`, results SHA256 `e5ddadf77842438f3262dd945af052323979c1cc2a6db022984febd74d7bd98f`, learned/work_log, entrypoint and concrete continuation proposal. The REPORT's leading terminal summary and final work-log entry correctly distinguish component evidence, unknown child exit code, unchanged selected policy/rates/scores, and an unapproved continuation. Phase1/phase2/correctedGPQA remain passed under their declared band/timing rule; SWE85 retains null point score,23 confirmed successes,3 unknowns and the same85-task denominator. No replacement scored repair or release-ready status is asserted.

Some older repeated status fields still described full-BF16 resume as active, and one historical REPORT paragraph retained active/unadmitted arm labels. I notified the owner to update current labels or mark them historical. These are reconciliation notes with unchanged underlying terminal evidence, not a new experimental requirement or a separate reason to prolong the review. Source-formatting hooks after this inspection must preserve exact previous bytes and verify semantic equivalence, as the existing archives do.

Final verdict: **more-work-needed**. Required next step: **human authorization** of the proposed additional local-only window. Completed investigation and truthful bounded reporting are retained; unmet serving/rate/list gates are not relabeled as passed. Final runner-owned usage remains a terminal runner responsibility, not a pre-closure missing artifact.

## Final narrow reconciliation check

Checked the corrected canonical files: REPORT SHA256 `50c569181aa04d88e0af15e71e3ff4ceccfa03ef8d949f7c85ac4c7d6f8a8fb4`; results `e7a05e1660279778c3a568a32da1a254351f072f85f7c213845808617ce276e0`; learned `08664fa474b27e7850c518da09fb22b8b520426480b2264670acd08dea2cc1d0`. The stale current-status notes identified above are reconciled: results.current_precision_disposition and pipeline identify terminal full-BF16 resume/broad component evidence, wrapper143 and unknown child status; learned's control record points to the canonical disposition. Earlier precision narration is explicitly headed as a historical snapshot.

The saved gate exits2 because it requires a numeric score.solved; score.n is present and equals85, while solved/value/interval remain null because three verifier outcomes are unknown. The gate diagnostic's combined wording does not mean the denominator is missing. No grade fabrication or denominator adjustment is appropriate. This was a read-only canonical check plus this review addendum; no experiment or implementation change was performed.

Reconciliation is checked. Verdict remains **more-work-needed**, with the same sole substantive Required Work item and explicit **human authorization** of the proposed additional local window as its next step. No new finding or experiment is requested.
