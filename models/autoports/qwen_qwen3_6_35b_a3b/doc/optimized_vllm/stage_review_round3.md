# Stage Review

Verdict: pass-with-carried

Stage 10, optimized-vLLM, Qwen/Qwen3.6-35B-A3B. Independent follow-up, round 3 of 3, reviewed 2026-11-18. This is a review of the round-2 findings and documentation delta, not a fresh whole-stage acceptance run.

The two remaining required findings are closed by the explicit owner revision, not by improved performance. Both numerical failures and the installed checker's exit 2 remain intact. The owner accepts precisely these two speed shortfalls, retains the validated configuration and prohibits further stage-10 speed work. No correctness, accuracy, sampling, trace, context, cleanup or candidate-promotion exception follows from that decision. No remaining Required Work was found within this follow-up scope.

Path shorthand: `M = models/autoports/qwen_qwen3_6_35b_a3b`, `D = M/doc/optimized_vllm`, `R = M/readiness_vllm`. Paths below are relative to the tt-metal checkout unless stated otherwise.

## Required Work

None under the revised owner contract.

- **RW2 — closed as a stage-10 acceptance blocker by explicit owner decision; measured deficit remains.** `R/optimized_acceptance_perf/vllm_result.json` still records 128 input / 128 output / concurrency 1, mean TPOT 31.426704362196126 ms and median TTFT 2969.7476220026147 ms. Independently recomputing `1000 / TPOT` gives **31.820072142305893 tokens/s/user**. The inherited useful-work calculation, `1,107,522,496 / 512e9 / 0.10`, gives 21.63129875 ms and **46.229309278066346 tokens/s/user**. The raw result has not changed. The exact message in `/home/ttuser/qwen36-35b-a3b/multigoal_logs/10-10-optimized-vllm.resume-2.message.txt` explicitly accepts this row; `D/owner_acceptance.json` reproduces it verbatim. Both limitations files preserve `raw_gate_status: FAIL`, the measured result, the floor and `waiver_basis: explicit-owner-decision`.

- **RW3 — closed as a stage-10 acceptance blocker by the same scoped decision; canonical P4 deficit remains.** `M/doc/eval_readiness/long_answer_load.json` still reports **2.36 tokens/s/user** for the first 1000-output bucket of the canonical 32 users × 6000 outputs run. The raw `long_answer_load/requests.jsonl` contains 32 error-free records, every output count is 6000 and the maximum prompt count is 74. Re-summing the layer and terminal terms in `D/eval_batch32_necessary_floor.json` gives **10,745,186,240 useful bytes/rank/step**, hence the necessary floor **4.764924390924285 tokens/s/user**. The owner explicitly accepts this recorded shortfall. P4 remains `original_model: not_run`, `fix_hours: 0.5786`; neither HF-same behavior nor a two-hour waiver is asserted. The separate 32 × 1000 candidate diagnostic is not substituted for this canonical result, and its prototype remains unadopted.

- **The raw checker failure is reconciled, not hidden or converted to a pass.** `D/owner_acceptance_stage_check.json` records the CPU stage-check invocation and exit 2. Its log has exactly one `CRITICAL` line: `long_answer_load/functional_floor`, 2.36 versus 4.76. The other reported eval findings are advisory. Inspection of installed `scripts/check_eval_readiness.py::waiver_reason` confirms that it recognizes matched-original behavior or the fix-time threshold, with evidence and reason prerequisites, and has no explicit-owner branch. The owner's later instruction changes this row's acceptance requirement; the unchanged checker's numerical failure is therefore not an unmet requirement under that revised contract. The `gate_waived` documentation fields mean owner disposition, not successful automatic waiver evaluation. No checker modification or fabricated passing exit is needed or claimed.

## Other Concerns

- **RW1 — remains closed.** Round 2 accepted the narrow 2/4/32 fused-mask dispatch restriction, host coverage across accepted capacities, representative capacity-8 tracked serving and 17 exact greedy controls. The tracked delta from retained runtime checkpoint `5313d9b211e767489e88847a84607f6a142edb3f` contains documentation only. There is no new implementation change that invalidates that closure. This round does not claim fresh hardware validation at every capacity.

- **Mixed-traffic ITL — open advisory, carried.** The retained three-budget maxima are 25.6824 / 92.4546 / 186.7118 seconds against the 1-second bound. The changed README continues to say no budget qualifies and that 2048 is retained for diagnosis. The owner decision does not imply that the ITL bound was met. Background output caps remain caps, not proof of complete 16384-token answers. No further speed experiment is requested in stage 10.

- **Load forecast — open advisory, carried.** The canonical planning estimate remains 15.08 hours against 14.4 hours. The unchanged canonical result records a 263168-token KV pool; `32 × (74 + 12000) = 386368` exceeds it. The README retains the extrapolation and potential-preemption caveats and does not promote the candidate's shorter diagnostic forecast. Stage 11 must use its actual release plan and measured answer lengths; this is not measured eval accuracy or demonstrated release duration.

- **Bounded quality evidence — remains an evidence limit, carried.** No new output was generated and no output artifact changed in this delta. Round 2's content-only candidate tails omit separately collected reasoning, and the retained canonical forced-ignore-EOS tails do not locate the first EOS. The changed docs do not promote either performance probe to qualitative acceptance or claim a matched HF result. The prior shared qualitative evidence remains the separate, bounded evidence; no broader quality conclusion or quality waiver is introduced.

- **Untried optimization and the P4 prototype — open conditional handoff, carried.** The combined patch remains promising and unadopted. Component and mixed-serving success do not establish full candidate sampling, accuracy or 6000-token acceptance. If a later authorized stage promotes it, the previously recorded full candidate gates remain necessary. The existing geometry, transport, embedding, head-to-sampler and optional batched-prefill ideas remain unmeasured handoff opportunities. They are not a request to resume speed work in this stage.

- **Previous documentation note — closed for the identified summaries.** The README now gives `fix_hours: 0.5786`; checklist E1/E2 mark the completed measurements and cleanup correctly. The incomplete checkpoint's chronological work-log entries and the historical round-2 verdict remain preserved and are superseded by the explicit 2026-11-18 owner entry. The final checklist repair paragraph still says “no floor waiver or stage pass”; read in its historical campaign context, it does not override the later owner decision. Clarifying that sentence and updating pending-review/status fields are ordinary closeout bookkeeping, not a new review blocker.

Retain these advisory and evidence limits under `Carried to next stage` in the closeout work log, together with this report. The accepted speed deficits must remain visible as known limitations; they are not unresolved stage-10 optimization work.

## Hard-Check Gaps

- The installed checker cannot represent this explicit owner revision. Preserve `owner_acceptance_stage_check.json` and its log alongside `owner_acceptance.json` and this report. A future automated consumer must distinguish the raw numerical failure from the scoped acceptance decision; this review does not promise that an unmodified runner will advance automatically.
- P4 request summaries lack complete raw SSE/timestamp arrays and retained reasoning, limiting independent timing/content reconstruction. The recorded failed result remains failed; stronger evidence retention alone is not new Required Work for this documentation-only closure.
- The unadopted prototype's full accuracy, sampling and long-load gates were not run. Those gaps apply to future promotion, not to the restored implementation. Raw combine payloads and some build/server logs remain persistent local artifacts under ignore rules, as recorded in round 2; a commit alone may not contain everything needed to repeat that analysis.
- This follow-up inherits the unaffected round-2 acceptance evidence. It does not rerun or re-review all original serving, trace, context and qualitative checks. The documentation consistency check and precommit log are not substitutes for those gates.

## Anomaly Ledger

- Observed anomaly: both selected-path speed floors remain numerically failed.
  Evidence: unchanged primary raw result, canonical P4 result and requests, necessary-floor arithmetic, prior AutoFix records and `D/owner_acceptance.json`.
  Affected path: 128/128/1 serving and first-bucket throughput for 32 users × 6000 outputs.
  Control or comparison: exact recorded floor calculations and unchanged retained configuration; no HF-same control is claimed.
  Likely subsystem: previously investigated serving/model execution cost; this round adds no causal attribution.
  Investigation performed: verify exact owner message, rederive rates/floors, check raw artifact identity and the two limitation records' scope.
  Resolution: **controlled by explicit owner acceptance for these two stage-10 rows; numerical shortfalls are not fixed.**

- Observed anomaly: stage checker still exits 2 after the owner acceptance is documented.
  Evidence: `D/owner_acceptance_stage_check.json`, log and installed `waiver_reason` implementation.
  Affected path: automatic stage acceptance reporting.
  Control or comparison: exactly one critical result, matching the owner-accepted P4 row; no explicit-owner branch in the checker.
  Likely subsystem: checker contract schema.
  Investigation performed: inspect raw log, waiver prerequisites and unchanged `original_model`/`fix_hours` fields.
  Resolution: **controlled and disclosed; raw failure preserved, owner contract revision separately recorded.**

- Observed anomaly: historical documents still describe incomplete acceptance or unwaived floors.
  Evidence: round-2 report, chronological work log and final checklist repair paragraph versus the dated owner entry.
  Affected path: handoff interpretation.
  Control or comparison: verbatim owner source, scoped limitations and current top-level disposition.
  Likely subsystem: chronological documentation.
  Investigation performed: inspect changed summaries and confirm the earlier raw findings were not rewritten.
  Resolution: **controlled by dated supersession; remaining wording/status cleanup is advisory closeout work.**

- Prior anomalies outside this delta retain round 2's dispositions: capacity dispatch fixed; unmatched sampled-row comparator controlled; packed-input signed-zero/subnormal mapping controlled within the unadopted component fixtures; empty candidate tails controlled as an evidence-retention limit. No changed runtime or output path reopens those findings.

## Scope Inspected

- Goal/skill paths: original `10-10-optimized-vllm.prompt.txt`; original contract and round-3 brief in `D`; exact `resume-2.message.txt`; installed tt-model-bringup 0.1.24 `stage-review/SKILL.md`, `model-bringup/SKILL.md` startup, relevant `optimize/SKILL.md` and `vllm-integration/SKILL.md` requirements. The active skill catalog supplies the installed tt-autodebug dependency. No package setup, installation or package mutation was performed.
- Artifact paths: all seven tracked documentation deltas, the documentation closeout in `1aed567e046bb6ceaed20f7391c5555e27f8f760`, `D/owner_acceptance.json`, `owner_decision.md`, stage-check JSON/log, consistency JSON, cleanup JSON, final precommit log, round-2 report and round-3 brief. Raw primary result, canonical long-answer result/request records and batch-32 floor terms were inspected only to validate affected claims. The brief reproduces round 2's Required Work and Other Concerns verbatim.
- Code paths: only the installed checker's waiver logic was inspected anew. Retained model/native implementation was not re-reviewed; `git diff` against the retained runtime checkpoint shows documentation-only tracked changes. Raw primary/P4/floor artifacts and the round-2 report are byte-identical to HEAD.
- Commands run: read-only `rg`, `cat`, bounded `sed`/`tail`, `git status`, `git diff`, `git log`, `git show`, and standard-library Python artifact parsing, byte comparisons and arithmetic. No tests, servers, hardware operations, TTNN imports, profiling, resets, signals, implementation edits, commits or pushes were performed by this review. The only write is this report.
- Workspace: live `tt-metal` branch `main`, HEAD `1aed567e046bb6ceaed20f7391c5555e27f8f760`, retained runtime checkpoint `5313d9b211e767489e88847a84607f6a142edb3f`. `owner_acceptance_cleanup.json` reports no serving processes or TT device holders from read-only `/proc` inspection.
- Push state: the supplied run plan authorizes local checkpointing only and prohibits autonomous push. The stage records no push occurred; this review performs none and infers no push authorization from repository history.

## Residual Risk

- The accepted implementation remains below both original speed floors and has severe measured mixed-traffic latency. Owner acceptance changes stage disposition, not measured performance or future workload capacity.
- The default load forecast exceeds its planning allowance and extrapolates beyond measured answer length. Broader quality, longer-answer cache pressure and future candidate acceptance remain bounded by the recorded evidence.
- This verdict closes the prior required findings under the explicit revised owner instruction. The stage owner still performs ordinary local closeout: link this report, record carried notes, finalize status fields and checkpoint the stage-owned documentation without pushing. Those bookkeeping steps require no new hardware run or review round.
