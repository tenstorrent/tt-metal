# Stage 11 runner-check remediation work log

The reported CRITICAL finding is **refuted as a completion defect under the owner-approved contract**. It is reproduced as a genuine incompatibility in the enclosing runner's pinned 0.1.24 checker. That exact script still exits 2; this log does not claim otherwise. The user explicitly permits a concrete refutation in the stage work_log.

The original owner message at `multigoal_logs_s11/11-11-benchmark.resume-1.jsonl:11`, sent 2026-11-18T05:11:01Z, says: “APPROVED - a third benchmark attempt exactly as in doc/benchmark/run/retry-proposal/README.md, with the eight-hour client limit and the existing three-hour setup limit.” The [retained transcript excerpt](run/runner-resume-owner-message.json) independently establishes the authority behind [owner-approval.json](run/owner-approval.json). This supersedes the original default two-hour client bound. No new approval or inference attempt is assumed.

## Reproduction and causal isolation

From the tt-metal checkout, with `MODEL_DIR=models/autoports/qwen_qwen3_6_35b_a3b` and `HF_MODEL=Qwen/Qwen3.6-35B-A3B` exported:

1. `bash /home/ttuser/qwen36-35b-a3b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/prompts/model_bringup_multigoal/11-benchmark.check.sh` exits **2**: `budget_seconds (client clock) must be positive and at most 7200`. [Fresh exact-script output](run/runner-resume-reproduction.log).
2. `bash /home/ttuser/qwen36-35b-a3b/skills-v24/plugins/tt-model-bringup/prompts/model_bringup_multigoal/11-benchmark.check.sh` exits **0** on the same artifacts. [Fresh approval-aware output](run/runner-resume-approved-check.log). The context check passes with full 262144 capacity, zero critical findings and historical nested-log advisories.
3. A diagnostic imported the cached checker and reproduced its exception, then replaced only its `batch_plan` reference in memory with the committed approval-aware implementation. Every cached evidence gate passed, with no answer/protocol advisories. [Isolation output](run/runner-resume-isolation.log). This is a causal diagnostic, not a claim that the exact installed script passed. It writes no package or measurement files.

The existing workflow repair is local commit `f2873162ea1b8b4971aa9b41c83a551e7c1d53e5` in `/home/ttuser/qwen36-35b-a3b/skills-v24`, plugin version 0.1.26. `batch_plan` honors an explicit finite positive owner budget with approver/reason/source, rejects conflicting limits, and retains the original default guard when no approval exists. The evidence checker change is only its over-budget error wording; all measurement gates remain. The installed package is read-only and remains 0.1.24. The enclosing runner selected that stale package after the approved run used the repaired workflow.

Focused regression command: `cd /home/ttuser/qwen36-35b-a3b/skills-v24 && timeout 120 python3 -m pytest tests/test_benchmark_stage.py -k 'budget or owner_approval' -q`. Result: **29 passed, 168 deselected**. Coverage includes default rejection, malformed/conflicting approval, summary provenance mismatch, exceeded approved budget, and missing performance evidence. No additional source change is needed.

## Completion evidence

The [final report](run/REPORT.md), frozen inputs and raw measurements remain valid: 32/32 accuracy responses, all 182 measured performance requests, batches 1/8/16/32, both lone-user rows, and both full-phase roofline estimates for every batch. MMLU-Pro extraction is 14/14; IFEval extraction is 18/18; no blank-at-cap responses or truncations. All 40 layers, selected precision and full context remain verified. The [prior closing record](run/attempt3-support/setup/closing-record.log) records client 22744.684/28800 seconds and setup 7209.514/10800 seconds. This remediation continues the original finalization clock in the report/summary; it does not restart the budget.

No device was opened, reset or reserved during this remediation. No serving process or foreign process was stopped. No raw accuracy/performance artifact, context contract, installed package or original approval was changed. The original failed checker output is preserved.

Runner disposition: accept this explicit refutation, or select the already repaired 0.1.26 checker at the path above when rechecking this approved run. Repeating the pinned 0.1.24 script will continue to reject the authorized budget. Do not lower recorded elapsed time, fabricate a 7200-second run, or rerun inference to conceal the checker mismatch.

Prior local report/evidence commits: `ed1a846828af7068d3e38ca01e36bbea7750a8f8` and `29769ead54cbe4f1a5d7433fb6e86c3f97f42b5b`. Remediation evidence is committed locally; its SHA is provided in the final response. Nothing is pushed.

Independent CPU audit confirmed the original owner message, both checker outcomes, planner-only causal isolation and unchanged validators. `answers.py`, `evidence.py`, `roofline.py`, `subsets.py` and `responses.py` are byte-identical across the cached and repaired packages. Approval authenticity was checked against original user input; the code itself validates its structure and consistency.

Final verification after report-clock refresh and normal hooks: exact cached script exit 2 (same sole budget defect); approved workflow script exit 0. Normal pre-commit suite exit 0; [hook output](run/runner-resume-hooks.log). Refreshed client clock 22939.786/28800 seconds; setup 7209.514/10800 seconds. Original raw measurements and context contract remain unchanged.
