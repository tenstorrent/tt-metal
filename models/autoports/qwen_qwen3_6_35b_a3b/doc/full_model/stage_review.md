# Stage Review

Verdict: more-work-needed

Independent review, round 1, of stage 6/full-model for **Qwen/Qwen3.6-35B-A3B**, performed on 2026-11-14. Inspected the live `main` worktree at `958dbadc5c4eb4b6b6ce73f792a0b862ba8f9e35`, including staged stage-owned files. No hardware, target imports, servers, tests, installed gate reruns, resets, or implementation edits were performed. This report is the reviewer's only write.

The final all-40-layer accuracy, maximum-context acceptance, and isolated traced performance have credible, correctly scoped evidence. The explicit retained-state resumed-prefill requirement remains unsatisfied. The cap instruction allows nonblocking hand-offs; it does not turn this failed required check into a passing stage. The finding below is an acceptance-contract gap, **not a claim that the investigation proved a TT-specific cache or arithmetic bug**.

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

## Hard-Check Gaps

- The stage06 shell checker covers degeneracy, context declarations, and serving-manifest structure. It does not enforce the full-model accuracy, split-sampling, genuine continuation, native policy, or human-output-review requirements by itself. Those were inspected separately here.
- I inspected the post-gate schema reconciliation without rerunning or importing the checker. The current manifest has 52 supported, 30 not-supported, and four missing rows. All 52 supported evidence paths exist locally; the 30 declarations are strings and the prefix-cache exclusion names `recurrent_state`. The recorded 82 formatting corrections are consistent with the current structure. There is no post-edit installed-gate PASS, and the four missing rows remain decisive regardless of format.
- Earlier B3 accuracy, B32 capability, serving commands, external-cache, and qualitative artifacts predate the final lifetime edit. They are not byte-identical final-source reruns. The current source preserves their relevant behavior; the later change releases dead singleton K/V aliases before SDPA. Final-source all-40 accuracy/performance, maximum-context acceptance, and reduced tracked Watcher replay provide relevant later coverage. This does not justify a blanket claim that every historical probe was repeated, or require an unrelated full rerun under the cap.
- Maximum-context testing establishes successful execution/cache-boundary behavior and measured prepared allocation at 262144, not HF accuracy across a 262K prompt. Batch 32 evidence establishes the recorded fixed-slot execution and inactive-state scope; it is not B32 HF accuracy. These limits are disclosed rather than silently substituted for stronger claims.
- Native component profiling covers real layers 0/3 and the terminal path. I inspected compact operation groups, native metadata and report excerpts, not the expanded 1.4 GB CSV or 171 GB raw capture. Their raw execution completeness is supported by the retained coverage validation; this review does not independently regenerate that conversion.

## Anomaly Ledger

- **Observed anomaly:** Three resumed-prefill numerical failures.
  **Evidence:** `readiness/prefill_resume_check.failed.json`; `AUTOFIX_RESUME.md`.
  **Affected path:** All-40-layer genuine retained-state eager continuation.
  **Control or comparison:** Same case-level HF sensitivity; exact reduced bounded/accepted continuation parity; ten negative-control rejections.
  **Likely subsystem:** Public-call recurrent rounding and singleton/chunk grouping are plausible sensitivity mechanisms; no unique TT root cause established.
  **Investigation performed:** AutoDebug/AutoFix evidence plus current source inspection of reset, absolute suffix slicing, positions/page windows and recurrent consumption boundary.
  **Resolution:** **more-work-needed** for the explicit passing contract; baseline HF sensitivity separately controlled.

- **Observed anomaly:** Batch composition previously changed another request's routed MoE output.
  **Evidence:** `AUTOFIX.md`, `generator_replay_probe/mixed_moe.json`, `generator_replay_probe/lifetime_fixed_watcher.json`, `batched_teacher/result.json`; shared commit `69a03c857a5366dd575dc9a7a655b7248c15fad8`.
  **Affected path:** Configured multirow fixed-slot decode.
  **Control or comparison:** Real-input isolation diagnostics, solo/mixed comparison and final reduced tracked replay; all-40 B3 top-5/top-100 pass.
  **Likely subsystem:** Launch-global stochastic rounding depends on other experts' token counts.
  **Investigation performed:** Shared argument is propagated into the program-cache attributes and compute define; default remains true, fixed-batch decode alone selects false. Dtype/fidelity/layout are preserved.
  **Resolution:** **fixed** within the recorded coverage; capacity-1 fast path and prefill retain their prior arithmetic policy.

- **Observed anomaly:** Full-context preparation originally failed with L1/static-CB overlap before any long prefill.
  **Evidence:** `context_acceptance/result.json`, `AUTOFIX_CONTEXT.md`, `context_acceptance/l1_repair_tracked.json`, `context_acceptance/final.json`.
  **Affected path:** Singleton full-attention SDPA with the full page-table allocation.
  **Control or comparison:** Reduced failing pressure probe, stronger passing repaired probe, then one actual all-40 long acceptance.
  **Likely subsystem:** Dead K/V tensors and the loop alias retained L1 storage across SDPA.
  **Investigation performed:** Current decoder deletes those aliases after ordered cache updates. Final 262143-token prefill plus decode at 262143 succeeds, writes the final page row, preserves preceding rows and ends at the next-position sentinel 262144.
  **Resolution:** **fixed**; no context reduction and no repeated maximum-context sweep required.

- **Observed anomaly:** Profile v1/v2 host memory failures and v3 marker loss.
  **Evidence:** Preserved failure artifacts; `performance/profile_reduced_v4_drains/component_summary.json`, native metadata, component reports; shared commit `958dbadc5c4eb4b6b6ce73f792a0b862ba8f9e35`.
  **Affected path:** Profiling/capture conversion, separate from unprofiled inference.
  **Control or comparison:** V4 completed with 32,896 native executions, 16 sessions per component/rank and 748 native metadata entries; failed captures are not accepted performance evidence.
  **Likely subsystem:** Tracy push handling and insufficient eager/setup drains.
  **Investigation performed:** Inspected shared handler guards, bounded profiler harness, completed validation and compact operation groups. Recomputed the per-rank model/sampler kernel sums from the group CSV.
  **Resolution:** **fixed** for this reduced profile. Unclassified report categories remain a classification limitation; native operands/kernels remain available.

- **Observed anomaly:** Maximum-context tensor inventory omits callable model/helper owners.
  **Evidence:** `context_acceptance/final.json`, `context_acceptance/memory_audit.md`, `doc/context_contract.json`.
  **Affected path:** Reporting only.
  **Control or comparison:** Authoritative allocator census: 25,372,641,280 allocated + 8,606,073,856 free = 33,978,715,136 bytes/chip, unchanged across prepared/post-prefill/post-decode endpoints.
  **Likely subsystem:** Inventory traversal returned early for callable instances.
  **Investigation performed:** Compared source/provenance and arithmetic; current runner changed reporting, while the eight recorded non-runner sources match. BF16 KV is 2,684,354,560 bytes/chip and three residual workspaces are 3,221,225,472 bytes/chip; neither is double-counted with the census.
  **Resolution:** **controlled** by explicit partial-inventory labeling and measured allocator totals. A future detailed inventory can use the corrected visitor; no repeated long acceptance is needed for bookkeeping.

## Scope Inspected

- **Goal/skill paths:** `doc/full_model/stage_goal.txt`; installed 0.1.24 `stage-review`, `full-model`, `tt-enable-tracing`, `tt-device-usage`, `qualitative-check`, and model-bringup startup instructions; repository `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`; installed stage06/check-serving source. The supplied cap instruction governed this inspection-only final review.
- **Core implementation:** `tt/model.py`, `generator.py`, `prefill.py`, `prefill_validation.py`, `cache_contract.py`, `serving.py`, `serving_state.py`, fixed-slot helpers, changed multichip-decoder paths and both shared-code commits. The model requires the 1x4 mesh, loads all selected layers, applies HF offset final RMSNorm and untied vocabulary head. Inherited residual/KV/CCL/precision policy remains; storage sharing does not substitute a single-chip or host model path.
- **Trace/lifecycle finding:** Generator preparation warms eager prefill shapes, decode, sampling, remap and output helpers before the first model/sampling captures. Stable sampler logits, token input, positions and cache stay allocated. The model trace advances positions; common sampling writes `tt_out_tok=self.tokens`. Captures occur in `prepare`, releases in `close`, not in request reset/mode changes. No blanket corruptible allocation exemption was found in these generator paths. Explicit host sampling/logits and teacher-forced input uploads are separate from token-out performance.
- **Runtime evidence:** Final prefill and teacher artifacts/logs plus adjacent provenance, isolated `performance/lifetime_fixed.json`, replay/Watcher, context acceptance, sampler comparison/probe, B3/B32, serving-command and external-cache/host-sampling probes, fallback/provenance audits, serving contract, readiness failure, HF control and reduced parity.
- **Batched prefill finding:** The final reduced replay's mixed lengths 31/65/257 match their solo last-token logits exactly for all three rows, both with unchanged and changed unused pages: PCC 1, maximum error 0 and zero unequal elements. The probe's assertion permits a tolerance, but its actual recorded tensors meet equality. Reordered submission, simultaneous state remap and owned async outputs have separate serving-command evidence. No vLLM-stage burst or server-level mixed-shape acceptance is inferred.
- **Numerical finding:** Independently checked final readiness rows against logs: both use 100 predictions with 98% top-1, 100% top-5 and 100% top-100. All five recorded sources match each final accuracy provenance. The exact pinned HF AIME24 chat reference has 161 prompt tokens, 100 generated tokens and top-100 records; the retained reference summary/cache metadata identify the revision and exact-match reuse.
- **Performance finding:** All nine source hashes in the isolated final performance artifact match current files. The 128-input/128-output workload has 127 timed decode steps: TTFT 2919.990325 ms, caller-observed decode 25.128268 ms/token = 39.795819 t/s/u. Queued production is separately labeled 39.869407 t/s/u. Both use 127 model and sampling replays, zero per-token token/position/page/activity refreshes or captures, and stable 8452 program entries. True HF logits-only teacher forcing is separately measured at 40.649627 t/s/u with 99 forced uploads; fixed-token replay is not mislabeled teacher forcing.
- **Profile/policy finding:** Native LM-head metadata shows BF16 input/BFP8 weight and HiFi2 matmul; the extra HiFi4 metadata entry belongs to the tail-reservation kernel. Routed sparse matmuls show BFP8/LoFi, with inherited FP32/BF16 special paths retained. Same-rank reduced kernel sums recompute to 1.471036125 ms model and 0.4548093125 ms sampler on the maximum-total rank (23.6161%). The separate all-40 sampler diagnostic is 0.487787433 ms, about 1.9448% of queued token-out time. Neither is an additive all-40 wall-time decomposition; sampling is not the dominant token-out cost in the evidence.
- **Read-only commands:** `cat`, bounded `sed`/`tail`, `rg`, `wc`, `git status/show/diff HEAD`, and Python standard-library JSON/CSV/path/source-hash analysis. No acceptance checker was executed. Actual generated text and prompt metadata were read for all six HF/TT pairs.
- **Checkpoint/push state:** Parent reports a local stage checkpoint is planned and no push. Shared fixes already have the two local SHAs above. During review the parent staged lossless checkpoint packaging for raw evidence that conflicts with repository hooks, retaining original local paths; no inference or measurement bytes changed. Final stage checkpoint SHA is not yet available and is not inferred from reflog. Packaging/restore details are in `checkpoint_evidence/manifest.json` and its README.

## Residual Risk

- This is a static/artifact review; it does not certify every native kernel or the entire historical binary dependency chain. Current-source tracked replay and accuracy are stronger evidence than historical source claims, but have their stated shape/depth scopes.
- Request admission remains responsible for off-batch page ownership and legal resident continuation extents. The low-level API validates submitted mappings and rejects cache rebinding after preparation; later integration must honor the documented scheduler responsibilities.
- No vLLM adapter/server validation is claimed or required in this stage review. The pinned plugin inventory is a preparatory interface map.
- The required continuation gate remains unresolved after the recorded AutoFix attempt. Optional optimization, profiler-category and inventory improvements can be handed off, but cannot replace that gate or justify a passing verdict.
