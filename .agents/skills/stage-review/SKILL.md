---
name: stage-review
description: Independently review a TTNN model bringup stage before it is allowed to pass. Use when a stage is complete or near-complete and Codex needs a fresh xhigh subagent to compare the stage output against the original goal contract, skill requirements, logs, generated model outputs, checks, benchmarks, and code artifacts; identify bugs, oddities, contradictions, weak dismissals, or suspicious behavior that hard-edged stage gates missed; and return a blocking/pass verdict.
---

# Stage Review

## Mission

Prevent unsupported stage closure. A stage review is an independent,
inspection-first review of one bringup stage against the stage's actual goal
contract and evidence. Hard checks are a floor, not proof of correctness.

The reviewer is looking for anything that should be investigated or fixed in
the spirit of bringing up a correct, fast, production-worthy model, even when
the scripted checks pass.

## Required Inputs

Tell the reviewer exactly what it is reviewing. Include:

- stage name and number;
- target model and autoport directory;
- original stage goal text or a concise contract copied from the goal;
- stage skill path(s) the stage was supposed to satisfy;
- evidence roots and key artifacts;
- branch/commit and whether the worktree is live or a copied snapshot;
- any known user instruction that constrains the review, but not your theory of
  what is wrong.

The subagent runs in the same workspace and can explore the worktree, but it
does not know the stage context unless you provide it. Do not ask it to infer
which stage or goal contract applies from a broad experiment folder.

## Invocation Modes

Main-agent mode: when you are the stage owner or orchestrator, spawn a fresh
xhigh subagent for the review. Do not fork the current context unless the only
way to pass essential stage context is through the thread. Prefer passing raw
paths and a compact stage contract over passing your conclusions.

Reviewer mode: when your prompt says you are the independent reviewer, or you
were spawned with this skill plus a concrete stage contract and artifact paths,
perform the review directly. Do not spawn another reviewer. In this mode your
verdict can be `clean-pass` or `block`.

In main-agent mode, if no subagent tool is available, do a serial review only as
a temporary fallback and mark the result `not-independent`. This means the
review did not satisfy the fresh-subagent requirement. It is not a reviewer
verdict and never counts as a pass for autonomous bringup.

The reviewer is read-only unless explicitly asked to write a report file. It may
run local read-only commands and small analysis scripts over artifacts. It must
not start servers, run vLLM, reserve or use IRD hardware, open TT devices, reset
devices, launch long tests, or mutate implementation files. If the review finds
a problem that requires hardware or a server rerun to confirm, report the
required follow-up as a finding; do not perform it inside the review.

## Review Stance

Tell the reviewer:

- Treat agent-written READMEs and work logs as claims, not facts.
- Re-derive important claims from runner artifacts, logs, JSON, code, and output
  text.
- Read actual generated outputs for generation, full-model, vLLM, and release
  stages.
- Treat visible wrongness as a blocker until fixed, reproduced in a control, or
  otherwise proven expected by evidence.
- Treat checks passing as insufficient when artifacts show a contradiction.
- Treat words such as `minor`, `known`, `remaining`, `generally`, `mostly`,
  `except`, `but`, `waived`, `xfailed`, or `out of scope` as review triggers.
- Prefer precise blocking findings over broad advice.

## Blocking Standard

Block the stage when evidence shows one of these:

- visible model/output wrongness has no control, fix, or evidence-backed
  explanation;
- a required check, artifact, metric, or implementation path from the goal
  contract is missing, failing, stale, or contradicted by another artifact;
- logs or code show a plausible bug in a stage-critical subsystem, such as
  cache ownership, trace replay, token feedback, sampling, precision policy,
  page-table handling, or model output correctness;
- the stage lowers context length, eval length, benchmark length, or advertised
  `max_model_len` below the HF-advertised context without device-DRAM evidence
  proving the largest feasible context;
- the stage dismisses a material anomaly with prose instead of investigation.

Do not block only because a stronger evidence format would be nice. If the goal
and skill accept tests, code inspection, runner configuration, logs, and summary
JSON as evidence, do not invent a new required runtime-counter or profiler
artifact unless the existing artifacts are only prose, fail to tie the claim to
the measured path, or contradict the claim. Put such requests under
`Hard-Check Gaps` unless they hide a concrete correctness or performance risk.

Treat warnings that mention corruption, stale inputs, invalid cache use, trace
lifecycle hazards, wrong language/output behavior, host fallbacks, or device
health as blockers when they touch the stage's core contract and are not
classified in the stage evidence.

## What To Inspect

Inspect the artifacts relevant to the stage, not the whole experiment by
default. For any stage, review:

- original goal and active skill requirements;
- final README, work log, and any status or manifest files;
- test logs, check logs, benchmark JSON, profiler summaries, and gate outputs;
- implementation code touched by the stage;
- stale-artifact risk: paths in reports should exist and match the described
  run.

For generation or serving stages, also inspect:

- qualitative outputs, autoregressive completions, or release/eval output text;
- HF, full-model, previous-stage, or canonical controls when the stage reports a
  model-quality caveat;
- token/logit/cache/trace/sampling evidence when output has mechanical
  repetition, wrong language, truncation, prompt echo, cross-request leakage,
  odd leading text, or degraded coherence;
- sampling tests and skipped/xfailed cases, separating expected framework limits
  from correctness failures.
- context-length evidence: `doc/context_contract.json`, served `max_model_len`,
  eval/benchmark context settings, and any cap or truncation introduced by the
  stage.

For optimization stages, also inspect:

- before/after measurements in the same regime;
- whether optimized paths from previous stages were preserved;
- evidence for rejected optimizations or "already optimal" claims;
- whether performance claims compare like with like.

## Anomaly Rule

Any anomaly the stage agent noticed or that the reviewer sees must be classified
before pass:

```text
Observed anomaly:
Evidence:
Affected path:
Control or comparison:
Likely subsystem:
Investigation performed:
Resolution: fixed / controlled / blocking
```

Prose acknowledgement is not resolution. If the anomaly is visible in model
behavior and there is no control showing it is expected, mark it blocking.

## Reviewer Prompt Template

Use a prompt like this, filling in concrete paths:

```text
Use $stage-review as an independent reviewer for one TTNN model bringup stage.
You are the fresh review subagent. Do not spawn another reviewer. Do not modify
implementation files. You may run read-only local commands and small
artifact-analysis scripts over existing artifacts. Do not start servers, open TT
devices, reserve hardware, or run hardware/vLLM experiments.

Stage under review: <stage number/name>
Target model: <hf model id>
Autoport/evidence root: <absolute path>
Original stage goal contract:
<paste concise goal contract>

Stage skill(s) to compare against:
- <absolute path to SKILL.md>
- ...

Key artifacts to inspect first:
- <README/work_log paths>
- <check/benchmark/output paths>
- <implementation paths if relevant>

Review task:
1. Read the stage skill(s) and the stage evidence.
2. Compare the delivered artifacts against the goal contract.
3. Inspect generated outputs directly if present.
4. Look for bugs, oddities, contradictions, stale artifacts, weak dismissals, or
   suspicious behavior that hard checks might have missed.
5. Return the required verdict format. Do not summarize only; lead with
   findings.
```

## Verdict Format

Require this output:

```markdown
# Stage Review

Verdict: clean-pass | block

## Blocking Findings
- P1/P2: <title>
  Evidence:
  Why this matters:
  Required next step:

## Non-Blocking Concerns
- ...

## Hard-Check Gaps
- ...

## Anomaly Ledger
- Observed anomaly:
  Evidence:
  Affected path:
  Control or comparison:
  Likely subsystem:
  Investigation performed:
  Resolution:

## Scope Inspected
- Goal/skill paths:
- Artifact paths:
- Code paths:
- Commands run:

## Residual Risk
- ...
```

Only `clean-pass` with no blocking findings satisfies this skill.

If the main agent could not obtain an independent review, report
`not-independent` as a fallback status outside the reviewer verdict. Do not use
it to complete the stage.

## Main-Agent Follow-Up

After the reviewer returns:

1. Read the findings and verify that the cited artifacts exist.
2. If the verdict is `block`, do not mark the stage complete. Use the relevant
   debugging or `$autofix` skill to resolve it, then rerun `$stage-review`.
3. If the verdict is `clean-pass`, record the review artifact or subagent final
   answer path in the stage work log.
