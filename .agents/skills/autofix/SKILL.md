---
name: autofix
description: "Fix tricky software, TTNN, model-correctness, performance, or integration bugs with forked subagents. Use when a hard bug needs a disciplined repair loop: run autodebug if needed, test each proposed bug in isolation, keep proven fixes, refute wrong hypotheses, and rerun diagnosis with new evidence if needed."
---

# AutoFix

Use this skill when a hard bug needs more than ordinary local debugging. This is the entry point for hard bugs. `$autodebug` is inspection-only; `$autofix` is the follow-through loop that turns its hypotheses into verified fixes or refutations.

## Inputs

Start from an existing `AUTODEBUG.md` report when one is present and relevant to the current failure. If there is no report, or it clearly describes an older state or a different symptom, use `$autodebug` first to create a fresh report.

Also read the current failing command, logs, work log, tests, reports, and `git status`. Understand which changes are already present before editing anything.

## Forked Subagents

Do the diagnosis and hypothesis experiments with forked subagents when the environment supports them. The main agent should coordinate, review evidence, and integrate only proven fixes; it should not carry the full run/fix/retest transcript in its own context.

Use an xhigh forked subagent for the initial `$autodebug` pass when a fresh report is needed. Give it the current symptom, failing command, relevant logs, and explicit instruction to produce `AUTODEBUG.md` without editing implementation code.

Then use one forked subagent per proposed bug or tightly related hypothesis group. Give each subagent:

- the relevant `AUTODEBUG.md` finding;
- the original failing command and observed failure;
- the files or subsystem it should inspect first;
- the requirement to design a focused verify/refute experiment;
- permission to implement the smallest fix only after the hypothesis is verified;
- the requirement to return a concise result: verdict, experiment commands, fix summary, diff or patch location, verification commands, and remaining uncertainty.

Each subagent should work in its forked context/worktree. The main agent should review the returned evidence and diff before applying or keeping any fix in the main working state. If a subagent's fix is speculative, unverified, or solves a different problem, discard it and record the refutation.

If forked subagents are not available in the current environment, state that limitation and follow the same loop serially, keeping the hypothesis notes concise.

## Repair Loop

Treat each AutoDebug headline finding or proposed bug as a hypothesis, not as truth.

For each proposed bug in turn:

1. State the hypothesis, the evidence AutoDebug gave, and the prediction it makes.
2. Design the smallest focused experiment that can verify or refute it. Prefer a narrow unit/component test, shape probe, instrumentation, lowered-argument check, or controlled A/B run over broad trial-and-error.
3. Run the experiment and record the exact command, result, and interpretation.
4. If the hypothesis is refuted, write down why and move to the next proposed bug.
5. If the hypothesis is verified, implement the smallest fix at the right intervention boundary.
6. Prove the fix works by rerunning the focused experiment and the original failing check, plus any nearby correctness or watcher/perf checks that the risk requires.
7. Keep only changes with evidence. Revert speculative edits that did not verify the hypothesis or fix the failure.

Do not batch several unproven fixes together. The purpose of this skill is to connect each code change to a verified cause.

For accuracy bugs, use a localization ladder before broad changes: compare module or layer boundaries, split prefill from decode, and run CPU oracle substitutions such as reference final norm, LM head, softcap, or sampling on accelerator hidden states. This separates hidden-state drift from final projection, postprocessing, and top-k extraction bugs. `models/common/validation_tools.py` has decorator-based tools that make it convenient to replace parts of a model with reference torch implementations, which can be a rapid and convenient way to prove or rule out whether a suspected accuracy bug is indeed caused by one specific part of the ttnn model code. You can even use this to "bisect" search the source code space of the model to find the smallest part of it that needs to be replaced with the torch reference code in order to fix the bug. For localisation or confirmation/refutation of a suspected bug this can be a significant timesaver.

Treat "higher precision" or "safer numerics" as a hypothesis, not a fix. Test precision, compute-kernel, math-mode, and dtype changes with focused A/B runs, and revert them when they are unchanged or worse.

If every AutoDebug hypothesis has been refuted or fixed and the original problem remains, do not keep guessing from the stale report. Update the visible evidence, then run `$autodebug` again in a fresh forked subagent from the new state. Continue with the new report.

Stop only when the bug is fixed with evidence, the remaining blocker is outside the current environment or project scope, or the report plus experiments show a legitimate limitation that needs human/product direction.

## TTNN Experiment Examples

- Compare one decoder subcomponent against HF or the single-chip TTNN baseline with identical inputs and weights.
- For top-k accuracy drift, probe the earliest layer/step where logits or hidden states diverge enough to change rank order, then substitute CPU/reference tail components to isolate whether the bug is in the hidden-state producer or the output head/postprocessing.
- Print or assert the lowered TTNN op inputs: logical shape, physical/padded shape, dtype, layout, memory config, program config, compute config, mesh mapper, and runtime args.
- Add a temporary targeted test for a suspicious cache/page-table/current-position boundary, then keep it only if it belongs as a durable regression.
- Run a minimal watcher check when the hypothesis involves CCL, async completion, semaphores, NOC/L1 bounds, cache updates, or trace replay.
- Use profiler or `tt-perf-report` only when the hypothesis is performance-related or timing evidence is needed.

## Reporting

If this skill is used inside a bringup stage, update that stage's work log. For standalone use, write `AUTOFIX.md`.

Record:

```text
# AutoFix Report

## Starting Evidence
- AUTODEBUG.md source or reason a fresh AutoDebug pass was run.
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
