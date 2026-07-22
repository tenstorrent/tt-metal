---
name: autofix
description: "Fix tricky software, TTNN, model-correctness, performance, hang, or integration bugs with forked subagents. Use when a hard bug needs a disciplined repair loop: run autotriage or autodebug if needed, test each proposed bug in isolation, keep proven fixes, refute wrong hypotheses, and rerun diagnosis with new evidence if needed."
---

# AutoFix

Use this skill when a hard bug needs more than ordinary local debugging. This is the entry point for hard bugs. `$autodebug` is inspection-only; `$autofix` is the follow-through loop that turns its hypotheses into verified fixes or refutations.

## Inputs

Start from an existing `AUTOTRIAGE.md` or `AUTODEBUG.md` report when one is present and relevant to the current failure. If there is no report, or it clearly describes an older state or a different symptom, create a fresh report with `$autotriage` for hang/tt-triage cases or `$autodebug` otherwise.

For a hanging process, device stall, watcher/LLK assert, dispatch timeout, CCL/fabric wait, or any failure with tt-triage output, start with `$autotriage` before `$autodebug`. If the process is still alive and no triage log exists, try to capture tt-triage evidence with `$autotriage` first, then use that diagnosis as the starting report. If `$autotriage` cannot capture evidence or its report does not explain the issue, fall back to `$autodebug` and continue the normal repair loop.

Also read the current failing command, logs, work log, tests, reports, and `git status`. Understand which changes are already present before editing anything.

## Forked Subagents

Do the diagnosis and hypothesis experiments with forked subagents when the environment supports them. The main agent should coordinate, review evidence, and integrate only proven fixes; it should not carry the full run/fix/retest transcript in its own context.

Use an xhigh forked subagent for the initial `$autotriage` pass when a hang or tt-triage case needs a fresh report. Give it the current symptom, failing command, triage logs or permission to capture triage from the live hang, relevant source focus paths, and explicit instruction to produce `AUTOTRIAGE.md` before editing implementation code.

Use an xhigh forked subagent for the initial `$autodebug` pass when a fresh source-only report is needed. Give it the current symptom, failing command, relevant logs, and explicit instruction to produce `AUTODEBUG.md` without editing implementation code.

Then use one forked subagent per proposed bug or tightly related hypothesis group. Give each subagent:

- the relevant `AUTOTRIAGE.md` or `AUTODEBUG.md` finding;
- the original failing command and observed failure;
- the files or subsystem it should inspect first;
- the requirement to design a focused verify/refute experiment;
- permission to implement the smallest fix only after the hypothesis is verified;
- the requirement to return a concise result: verdict, experiment commands, fix summary, diff or patch location, verification commands, and remaining uncertainty.

Each subagent should work in its forked context/worktree. The main agent should review the returned evidence and diff before applying or keeping any fix in the main working state. If a subagent's fix is speculative, unverified, or solves a different problem, discard it and record the refutation.

If forked subagents are not available in the current environment, state that limitation and follow the same loop serially, keeping the hypothesis notes concise.

## Repair Loop

Treat each AutoTriage or AutoDebug headline finding or proposed bug as a hypothesis, not as truth.

For each proposed bug in turn:

1. State the hypothesis, the evidence AutoTriage or AutoDebug gave, and the prediction it makes.
2. Design the smallest focused experiment that can verify or refute it. Prefer a narrow unit/component test, shape probe, instrumentation, lowered-argument check, or controlled A/B run over broad trial-and-error.
3. Run the experiment and record the exact command, result, and interpretation.
4. If the hypothesis is refuted, write down why and move to the next proposed bug.
5. If the hypothesis is verified, implement the smallest fix at the right intervention boundary. Prefer a fix inside `models/autoports/<model>/`; if the fault is in a shared tt-metal kernel outside it, first try a model-side dtype/shape/topology change that avoids the failing path, and only if none exists capture a minimal repro and file the framework bug rather than landing a core-kernel patch as the stage fix.
6. Prove the fix works by rerunning the focused experiment and the original failing check, plus any nearby correctness or watcher/perf checks that the risk requires.
7. Keep only changes with evidence. Apply a verified fix to every occurrence of the flagged pattern, not just the one under review. Revert speculative edits that did not verify the hypothesis or fix the failure.

Do not batch several unproven fixes together. The purpose of this skill is to connect each code change to a verified cause.

For accuracy bugs, use a localization ladder before broad changes: compare module or layer boundaries, split prefill from decode, and run CPU oracle substitutions such as reference final norm, LM head, softcap, or sampling on accelerator hidden states. This separates hidden-state drift from final projection, postprocessing, and top-k extraction bugs. `models/common/validation_tools.py` has decorator-based tools that make it convenient to replace parts of a model with reference torch implementations, which can be a rapid and convenient way to prove or rule out whether a suspected accuracy bug is indeed caused by one specific part of the ttnn model code. You can even use this to "bisect" search the source code space of the model to find the smallest part of it that needs to be replaced with the torch reference code in order to fix the bug. For localisation or confirmation/refutation of a suspected bug this can be a significant timesaver.

Treat "higher precision" or "safer numerics" as a hypothesis, not a fix. Test precision, compute-kernel, math-mode, and dtype changes with focused A/B runs, and revert them when they are unchanged or worse.

For dtype or cache-related accuracy failures, do not collapse a passing higher-precision run into a claim of inherent numerical instability. Keep a precision-policy ledger for the failing and passing runs: activation dtype, weight dtype groups, math fidelities, cache dtype, CCL payload dtype, page/block size, cache layout, page-table policy, and update/read ops. A passing BF16 cache run proves only that the failure is sensitive to that boundary. Before blaming a low-precision cache or kernel, run at least one same-cache high-precision control and one exact-shape cache/page-table probe. Exact-shape means the probe imports the model's page block size, local KV heads, batch slots, cache allocation helper, mapper, and update row contract; hard-coded near misses are smoke tests, not proof.

For paged-cache decode failures, also check allocation coverage against the attention op's rounded read window. Compute and log the first top-k miss index, absolute sequence position, page/block id, and dynamic chunk boundary. If the miss is a cliff at a page, tile, power-of-two, or chunk boundary, run an over-allocation control before pursuing numerical-instability explanations.

If every AutoTriage hypothesis has been refuted or fixed and the original hang remains, do not keep guessing from the stale report. Update the visible evidence and try another `$autotriage` pass if fresh triage can be captured; otherwise run `$autodebug` in a fresh forked subagent from the new state. Continue with the new report.

If every AutoDebug hypothesis has been refuted or fixed and the original problem remains, do not keep guessing from the stale report. Update the visible evidence, then run `$autodebug` again in a fresh forked subagent from the new state. Continue with the new report.

Stop only when the bug is fixed with evidence, the remaining blocker is outside the current environment or project scope, or the report plus experiments show a legitimate limitation that needs human/product direction.

## TTNN Experiment Examples

- For traced-decode or serving output symptoms - token doubling, greedy nondeterminism, wrong output at the capture position, per-replica divergence, serving-only repetition - check the symptom table in `$tt-enable-tracing` first; each row names the likely mechanism and a focused experiment.
- Compare one decoder subcomponent against HF or the single-chip TTNN baseline with identical inputs and weights.
- For top-k accuracy drift, probe the earliest layer/step where logits or hidden states diverge enough to change rank order, then substitute CPU/reference tail components to isolate whether the bug is in the hidden-state producer or the output head/postprocessing.
- Print or assert the lowered TTNN op inputs: logical shape, physical/padded shape, dtype, layout, memory config, program config, compute config, mesh mapper, and runtime args.
- Add a temporary targeted test for a suspicious cache/page-table/current-position boundary, then keep it only if it belongs as a durable regression.
- Run a minimal watcher check when the hypothesis involves CCL, async completion, semaphores, NOC/L1 bounds, cache updates, or trace replay.
- For a wrong-output or low-PCC symptom in fused-QKV attention, test a QKV fused-order mismatch early, using the emit-driven order rather than assuming a fixed permutation.
- Use profiler or `tt-perf-report` only when the hypothesis is performance-related or timing evidence is needed.

## Reporting

If this skill is used inside a bringup stage, update that stage's work log. For standalone use, write `AUTOFIX.md`.

Record:

```text
# AutoFix Report

## Starting Evidence
- AUTOTRIAGE.md or AUTODEBUG.md source, or reason a fresh AutoTriage/AutoDebug pass was run.
- Original failing command/log/report.

## Hypothesis Experiments
- Hypothesis:
  Experiment:
  Result:
  Verdict: verified / refuted / still uncertain
  Evidence artifact(s):
  Fix, if any:
  Verification:

## Final Status
- Fixed / blocked / limitation / still failing.
- Commands that prove the final state.
- Remaining risks or follow-up evidence needed.
```

Keep the report concise and evidence-led.
