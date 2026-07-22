---
name: stage-review
description: Independently review a TTNN model bringup stage before it is allowed to pass. Use when a stage is complete or near-complete and Codex needs a fresh xhigh subagent to compare the stage output against the original goal contract, skill requirements, logs, generated model outputs, checks, benchmarks, and code artifacts; identify bugs, oddities, contradictions, weak dismissals, or suspicious behavior that hard-edged stage gates missed; and return a clean-pass or more-work-needed verdict.
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
verdict can be `clean-pass` or `more-work-needed`.

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
- Treat visible wrongness as required work until fixed, reproduced in a control,
  or otherwise proven expected by evidence.
- Treat checks passing as insufficient when artifacts show a contradiction.
- Treat words such as `minor`, `known`, `remaining`, `generally`, `mostly`,
  `except`, `but`, `waived`, `xfailed`, or `out of scope` as review triggers.
- Prefer precise required-work findings over broad advice.

## More-Work-Needed Standard

Return `more-work-needed` when evidence shows one of these:

- visible model/output wrongness has no control, fix, or evidence-backed
  explanation;
- a required check, artifact, metric, or implementation path from the goal
  contract is missing, failing, stale, or contradicted by another artifact;
- logs or code show a plausible bug in a stage-critical subsystem, such as
  cache ownership, trace replay, token feedback, sampling, precision policy,
  page-table handling, or model output correctness;
- a release or serving stage has failed accuracy, API conformance, benchmark
  target, or missing/incomparable metric rows that are only disclosed, not
  fixed or supported by a row-specific current issue waiver;
- a text LLM release has failed `meta_ifeval` or `meta_gpqa_cot` without a
  current linked issue proving the correct canonical implementation fails the
  same eval in the same way;
- the stage lowers advertised model capability, such as context length, eval
  length, benchmark length, served `max_model_len`, image resolution, timestep
  schedule, cache/state size, or another model-specific contract item, without
  evidence that a hard physical device limit prevents the advertised capability
  from fitting or running and proving the largest feasible value;
- the stage dismisses a material anomaly with prose instead of investigation.
- an optimized precision policy selects a slower higher-precision or
  higher-fidelity path even though a faster lower-precision/fidelity candidate
  passed real target-model weights or recorded target activations, and the only
  blocker is random/synthetic "representative semantics" PCC.
- a dominant matmul geometry sweep mixes evidence across incompatible precision
  policies, for example measuring a smaller-core or residual-grid geometry only
  under BFP8/HiFi while the final or mandatory policy is BFP4/LoFi. Geometry is
  not rejected until the material geometry candidates have been measured under
  the same dtype/fidelity policy, or an exact op-contract blocker is recorded.
- a perf "winner" from a work log or README is accepted without re-measuring it
  on device, or a decode/prefill/combined perf claim does not state which of the
  three it covers.
- a shipped precision silently diverges from the compiler-emitted precision (for
  example a blanket low-precision override of a mixed emitted policy) without a
  recorded, evidence-backed reason.
- prompt-based quality evidence violates `$qualitative-check`, for example an
  instruct/chat model is judged only from raw completion prompts, a base model
  is judged through invented chat prompts, or prompt-format evidence is missing.
- a stage from full-model onward can generate text but did not run the
  `$qualitative-check` shared suite, or did not record why the suite was
  impossible.

Do not return `more-work-needed` only because a stronger evidence format would
be nice. If the goal and skill accept tests, code inspection, runner
configuration, logs, and summary JSON as evidence, do not invent a new required
runtime-counter or profiler artifact unless the existing artifacts are only
prose, fail to tie the claim to the measured path, or contradict the claim. Put
such requests under `Hard-Check Gaps` unless they hide a concrete correctness or
performance risk.

Treat warnings that mention corruption, stale inputs, invalid cache use, trace
lifecycle hazards, wrong language/output behavior, host fallbacks, or device
health as required work when they touch the stage's core contract and are not
classified in the stage evidence.

A review verdict of `more-work-needed` means exactly that: the stage is not
ready to pass yet. It is a remediation trigger, not permission for the stage
owner to set the Codex goal to terminal `blocked`. The stage owner must treat
each finding as work: fix it directly when the cause is obvious, or use
`$autofix` when the fix is not obvious or the first direct fix does not close
the gate. Only a later, explicit `$autofix` failure or an unrecoverable external
dependency can justify terminal goal `blocked`.

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
- prompt-shape evidence for LLM stages: valid logical prompt or prefill lengths
  that are not divisible by internal chunk, tile, block, page, or trace sizes.
  A public model/generator/serving path that rejects such lengths needs more
  work unless the HF model itself has that semantic restriction.
- `$qualitative-check` evidence: prompt-format metadata, rendered prompts or
  token ids, HF/full-model controls, and the shared qualitative suite as soon as
  the stage can generate text. Missing suite output is required work unless the
  stage records a concrete capability blocker.
- capability-contract evidence: for nonstandard or non-LLM models, the
  model-specific equivalent of context length, cache/state semantics, mode
  switches, and advertised input/output limits.

For release stages, also inspect:

- the final release markdown and report data JSON, not only `RUN_NOTES.md`;
- every failed accuracy, API conformance, benchmark target, and
  missing/incomparable metric row;
- waiver evidence for any failed row claimed to be non-blocking. A valid waiver
  names the row, links a current issue or release note, and shows the correct
  canonical implementation or harness target fails for the same non-autoport
  reason. Disclosure alone is not a waiver;
- mandatory text-LLM quality gates such as `meta_ifeval` and `meta_gpqa_cot`.
  Treat these as required work unless a current linked issue proves the correct
  canonical implementation fails the same eval in the same way.

For optimization stages, also inspect:

- before/after measurements in the same regime;
- whether optimized paths from previous stages were preserved;
- evidence for rejected optimizations or "already optimal" claims;
- whether rejected optimizations were actually earned. For any material
  optimization that would remove an op, collective, reshard, or layout
  conversion, a first TTNN/API error is not enough. Require evidence that the
  stage adapted shape, layout, padding, or weight packing and retried, or used
  `$autofix`. Accept rejection only when the adapted path is measured slower,
  fails correctness for an understood reason, or a minimal repro proves the op
  cannot express the required contract;
- whether multi-device optimization candidates were measured as coherent
  families. A rejection is not earned if the stage tried a dtype/fidelity
  change under one residual/collective topology, tried a fused CCL/matmul path
  under another topology, and never measured the compatible combination. For
  material row-parallel or column-parallel boundaries, require evidence for the
  residual-layout contract, collective placement, activation/CCL dtype,
  projection packing or separation, and persistent-buffer plan used by the
  candidate;
- whether lower-movement residual contracts were actually tested. If a fused
  matmul+reduce-scatter, reduce-scatter, or similar candidate was measured only
  with an immediate all-gather/full replication back to the old contract, that
  does not reject the sharded/fractured residual family. Require an adapted
  stack-compatible measurement through the next consuming norm/residual/MLP or
  attention boundary, or a minimal repro showing that boundary cannot consume
  the layout;
- whether the final default path reproduces the selected best candidate. If the
  final default run is materially slower than the candidate evidence, the stage
  must either fix the default wiring or report the final reproduced number as
  the optimized result instead of claiming the earlier candidate result;
- whether the selected dtype/fidelity policy actually appears in measured
  runtime rows. Policy names, constructor defaults, and JSON fields do not
  prove that dominant matmuls used the intended weight dtype or math fidelity.
  If `tt-perf-report` or equivalent profiler rows show BF16/BFP8 where the
  selected policy claims BFP4, or HiFi where the selected policy claims LoFi,
  return `more-work-needed` unless the report records an exact op-contract
  blocker and uses the real measured dtype policy as the final result;
- whether a reduced-precision or reduced-fidelity candidate was rejected only
  by synthetic/random-weight evidence. Synthetic tests can reveal crashes and
  shape bugs, but they do not by themselves justify a slower higher-precision
  fallback when real target-model weights pass. A test called
  "representative semantics" is still synthetic if it uses random/synthetic
  tensors. If the stage says the candidate was faster on real weights but
  rejected for such a synthetic PCC, return `more-work-needed` unless there is
  model-visible correctness failure, real-weight trace/runtime failure,
  unacceptable latency, or an exact op-contract blocker;
- whether dominant matmul geometry was swept under the dtype/fidelity policy
  being selected. A core-count, shard-width, `in0_block_w`, `per_core_N`, or
  output-subblock result from BFP8/HiFi does not validate BFP4/LoFi geometry,
  and a BFP4/LoFi result on one geometry does not reject the other material
  geometries. Small `in0_block_w` values are not enough just because they used
  to look reasonable: if a material row keeps a small value such as `2`, `4`,
  or `8`, check whether larger legal divisors of the tiled K dimension and the
  input shard's K-tile width were measured under the selected dtype/fidelity
  policy. Legal values do not need to be powers of two; values such as `3`,
  `5`, `7`, `10`, `14`, or `16` may be valid depending on the shape and shard.
  Otherwise require an exact L1, divisibility, padding, or op-contract blocker.
  If the final report leaves a dominant row `SLOW`, has low DRAM
  utilization, reports missing output subblocks, or cites a larger-core blocker
  without a precision-locked smaller-core or residual-grid candidate, return
  `more-work-needed`;
- whether performance claims compare like with like.

For datatype-sweep stages, also inspect:

- the candidate matrix, not only the selected config;
- whether recorded dtype and compute-fidelity fields are actually consumed by
  the measured runtime path, using model summaries, propagation checks, or
  profiler/perf-report rows rather than JSON alone;
- whether every material BFP4 matmul group considered or selected has a
  BFP4+LoFi candidate, or an exact TTNN/runtime blocker plus `$autofix`
  evidence;
- whether a "fastest evaluated config" claim is only true because an obvious
  legal precision/fidelity candidate was missing.

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
Resolution: fixed / controlled / more-work-needed
```

Prose acknowledgement is not resolution. If the anomaly is visible in model
behavior and there is no control showing it is expected, return
`more-work-needed`.

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

Verdict: clean-pass | more-work-needed

## Required Work
- P1/P2: <title>
  Evidence:
  Why this matters:
  Required next step:

## Other Concerns
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

Only `clean-pass` with no required work satisfies this skill.

If the main agent could not obtain an independent review, report
`not-independent` as a fallback status outside the reviewer verdict. Do not use
it to complete the stage.

## Main-Agent Follow-Up

After the reviewer returns:

1. Read the findings and verify that the cited artifacts exist.
2. If the verdict is `more-work-needed`, do not mark the stage complete or
   terminal `blocked`. Treat the findings as the next stage work item. Use the
   relevant debugging skill or `$autofix` to resolve it, then rerun
   `$stage-review`.
3. If the verdict is `clean-pass`, record the review artifact or subagent final
   answer path in the stage work log.
4. After `clean-pass`, create local checkpoint commits for stage-owned changes
   in each touched repo, including `tt-metal` and `vllm` when applicable. Never
   push these commits from autonomous bringup.
5. Record each repo, branch, and commit SHA in the stage work log. Do not
   include unrelated dirty files in a checkpoint commit. If stage-owned changes
   cannot be isolated from unrelated dirty state, treat that as more required
   work and do not mark the stage complete.
