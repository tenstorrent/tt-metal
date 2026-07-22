# Autonomous Model Bringup

This directory contains everything needed to point an AI coding agent at a
HuggingFace model and have it brought up on Tenstorrent hardware — correct,
optimized, multi-chip, and serving through vLLM — without a human steering it.

It has three parts:

```text
.agents/
  prompts/model_bringup_multigoal/   # the ten stage goals, one prompt file each
  skills/                            # the knowledge the agent works from
  scripts/multigoal                  # the runner that chains the stages together
```

## Quick Start

From a built tt-metal checkout with its `python_env` active, Codex installed
and authenticated, and the agent dependencies installed
(`python -m pip install -r .agents/requirements.txt`):

```bash
python .agents/scripts/multigoal \
  --replace HF_MODEL=org/Your-Model-Here \
  --replace MODEL_DIR=models/autoports/org_your_model_here \
  .agents/prompts/model_bringup_multigoal/*.txt
```

That runs all ten stages back to back. Expect a full bringup to take several
hours of unattended work. Results land in `models/autoports/<model>/`, where
`<model>` is the HF model id lowercased with non-alphanumerics replaced by
underscores.

## How It Works

Each prompt file is one *goal*: a stage with explicit completion requirements.
The runner sends each goal to a fresh agent session, waits for it to reach a
terminal state, verifies the result, and only then starts the next stage. The
stages build on each other:

| Stage | Delivers |
|---|---|
| 01 functional-decoder | A correct TTNN decoder layer, validated against the HF reference (PCC) |
| 01b fused-decoder (optional) | The functional layer with `$graph-fusing` applied — dedicated ops, merged structural ops, folded adjacencies — before optimize |
| 02 optimized-decoder | The same layer made fast on one chip: precision, sharding, program configs |
| 03 multichip-decoder | The layer parallelized across the chip mesh |
| 04 optimized-multichip-decoder | The multichip layer optimized: async collectives, fused ops |
| 05 full-model | The whole model end to end: embeddings, layer stack, LM head, generation |
| 06 optimized-full-model | The full model traced and optimized, with honest perf accounting |
| 07 datatype-sweep | The fastest weight/activation/KV datatype config that still meets accuracy |
| 08 vllm | The model serving real requests through the Tenstorrent vLLM plugin |
| 09 optimized-vllm | The serving path optimized: async decode, trace reuse, on-device sampling |
| 10 tti-release | The tt-inference-server release workflow run and customer-facing readiness report |

The `$skill` references inside each prompt attach the matching skill from
`.agents/skills/` — that is where the engineering knowledge lives (how to
validate a paged KV cache, how to read a perf report, how to debug a trace
failure). Prompts say *what done means*; skills say *how to get there*.
Every hardware-facing stage also loads `$tt-device-usage`, the shared runbook
for safe TT device access, reset/list recovery, tt-triage hang capture, and
ARC/ERISC/remote-Ethernet recovery.

## Context-Length Contract

The bringup target is the full context length advertised by the HuggingFace
model config. Do not quietly reduce the model to a smaller `max_model_len`,
serving context, eval context, or benchmark context.

A smaller context is acceptable only when the target hardware cannot fit the
full model plus KV cache in device DRAM. In that case, the model must support
the largest context that reasonably fits, and the stage evidence must include a
byte calculation or a failed full-context capacity probe showing the limit.
Runtime, profiling cost, convenience, or test-harness speed are not valid
reasons to reduce context.

Each model keeps this as a handoff artifact:

```text
models/autoports/<model>/doc/context_contract.json
```

The artifact records the HF-advertised context, the current supported context,
any DRAM limit, and the evidence behind it. Functional decoder bringup creates
it. Multichip, full-model, optimized-full-model, datatype-sweep, vLLM, and
release stages update or verify it because tensor parallelism, full-stack memory
use, and KV-cache dtype can change the feasible context.

Each stage leaves its evidence under `models/autoports/<model>/doc/<stage>/`:
a `README.md` with the results, a `work_log.md` with the journey, and the
artifacts (perf reports, accuracy logs, watcher output) that back them up.

## Verification Gates

A stage README and work log are claims to inspect, not verification by
themselves. After a stage's goal completes, the runner looks for a sibling
check script (for example `05-full-model.check.sh` next to
`05-full-model.txt`) and runs it. Check scripts are plain bash, readable by
anyone, and exit with:

| Exit | Meaning | Runner response |
|---|---|---|
| 0 | pass | continue to the next stage |
| 1 | advisory failure | one remediation attempt, then record it and continue |
| 2 | critical failure | one remediation attempt, then record it and stop launching later stages |
| other (3 = checker error) | the check itself is broken | retry once, then stop: a disabled guardrail must not pass silently |

A *remediation attempt* is a fresh agent goal that receives the check's output
as a bug report: fix the underlying cause or refute it with evidence. The
current checks verify, among other things, that the model's generated text is
not mechanically degenerate (doubled tokens, single-token collapse — decode
loop bugs that accuracy metrics cannot see).

The runner records everything in its log directory: `STATUS.md` is the
scoreboard (one row per stage: goal status and gate verdict), `manifest.txt`
has the details, and `*.check-N.log` files hold each check's output. The
runner's exit code tells you why it stopped: `0` all stages green, `3`/`5` a
goal ended blocked or failed, `6` a stage failed verification critically, `7`
the verification harness itself is broken (fix it and resume).

Multigoal launches that include runner-side check scripts require
`MODEL_DIR`; pass it as `--replace MODEL_DIR=models/autoports/<model>`. The
runner records that exact path in `manifest.txt` and exports it to every check
script so verification is scoped to the intended autoport directory. The
checker still has an `HF_MODEL` fuzzy-match fallback for manual one-off use,
but unattended experiments should not rely on it.

## Useful Flags

- `--replace OLD=NEW` — substitute text in every prompt. Model bringup runs
  should always pass both `HF_MODEL` and `MODEL_DIR`; replacements with
  identifier-like names are also exported into check-script environments.
- `--start-index N` — resume a run from stage N without repeating earlier
  stages (after a fix, a machine move, or a harness stop). This starts a fresh
  thread for stage N.
- `--resume-stage N --log-dir DIR` — recover an existing terminal stage from
  `DIR/manifest.txt` by resuming its recorded `stage_N_thread_id`, sending a
  continuation turn in that same thread, running the stage check if it
  completes, and then continuing later stages. Use this for `usageLimited`,
  `budgetLimited`, or auth-account recovery where `--start-index` would lose
  the stopped thread's context.
- `--dry-run` — validate the prompt sequence and show what would run.
- `--check-retries N` / `--no-checks` / `--check-error-policy stop|continue`
  — gate behavior knobs; the defaults are the recommended ones.
- `--log-dir DIR` — where the manifest, STATUS.md, and per-stage logs go.

One sharp edge worth knowing: the agent backend caps a goal's objective at
4000 characters *after* `HF_MODEL` substitution. The runner validates every
prompt up front so a too-long prompt fails at launch rather than hours in, and
`scripts/check_agent_prompt_lengths.py` (wired into pre-commit) measures the
same invariant when editing prompts.

## Extending It

- **New stage**: add a numbered prompt file with explicit completion
  requirements, and reference the skills it should use with `$skill-name`.
- **New gate**: add `<prompt-stem>.check.sh` beside the prompt. Keep it
  deterministic, scope it to the model under test, and follow the exit-code
  convention above. Calibrate any threshold against both known-good and
  known-bad artifacts before trusting it.
- **New knowledge**: extend a skill in `.agents/skills/`. Prefer mechanisms
  over model-specific answers, and prefer a check over advice — enforcement
  generalizes; prose gets skimmed.
