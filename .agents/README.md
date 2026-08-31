# Autonomous Model Bringup

This directory contains everything needed to point an AI coding agent at a
HuggingFace model and have it brought up on Tenstorrent hardware — correct,
optimized, multi-chip, and serving through vLLM — without a human steering it.

It has three parts:

```text
.agents/
  prompts/model_bringup_multigoal/   # the eleven stage goals, one prompt file each
  prompts/diffusion_bringup_multigoal/  # the ten stage goals for diffusion models
  skills/                            # the knowledge the agent works from
  scripts/multigoal                  # runner (Codex app-server backend)
  scripts/multigoal-claude           # runner (Claude Code backend), same flags
```

Two interchangeable runners drive the same prompts and skills: `scripts/multigoal`
(the Codex app-server backend) and `scripts/multigoal-claude` (the Claude Code
port). They share the runner-side logic: manifest, stage checks, resume, and gate
policy. Pick whichever agent backend you have. The Claude runner adds
`--model`, `--effort`, `--permission-mode`, `--claude-bin`, and
`--claude-config-dir`, and has no objective length cap. Everything below applies
to both unless noted.

## Quick Start

From a built tt-metal checkout with Codex installed and authenticated, and the
agent dependencies installed in the checkout's `python_env`
(`./python_env/bin/python -m pip install -r .agents/requirements.txt`):

```bash
./python_env/bin/python .agents/scripts/multigoal \
  --replace HF_MODEL=org/Your-Model-Here \
  --replace MODEL_DIR=models/autoports/org_your_model_here \
  .agents/prompts/model_bringup_multigoal/*.txt
```

The runner disables Codex shell profile initialization by default. This keeps
agent-issued commands in the environment inherited by app-server instead of
reactivating an unrelated virtual environment from a login profile. A caller
that deliberately needs profile initialization can override the default with
`--config shell_environment_policy.experimental_use_profile=true`.

That runs all eleven stages back to back. Expect a full bringup to take several
hours of unattended work. Results land in `models/autoports/<model>/`, where
`<model>` is the HF model id lowercased with non-alphanumerics replaced by
underscores.

To use the Claude Code backend instead, swap the runner. It needs no extra
Python packages, only the `claude` CLI on `PATH` and authenticated:

```bash
python .agents/scripts/multigoal-claude \
  --replace HF_MODEL=org/Your-Model-Here \
  --replace MODEL_DIR=models/autoports/org_your_model_here \
  .agents/prompts/model_bringup_multigoal/*.txt
```

The Claude runner strips `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the
agent environment so an ambient key cannot silently move a multi-hour run onto
metered billing ahead of the subscription login. Point it at a specific login
with `--claude-config-dir`.

## How It Works

Each prompt file is one *goal*: a stage with explicit completion requirements.
The runner sends each goal to a fresh agent session, waits for it to reach a
terminal state, verifies the result, and only then starts the next stage. The
stages build on each other:

| Stage | Delivers |
|---|---|
| 01 functional-decoder | A correct TTNN decoder layer, validated against the HF reference (PCC) |
| 02 fused-decoder | The functional layer graph fused into faster equivalent TTNN operations |
| 03 optimized-decoder | The fused layer made fast on one chip: precision, sharding, program configs |
| 04 multichip-decoder | The layer parallelized across the chip mesh |
| 05 optimized-multichip-decoder | The multichip layer optimized: async collectives, fused ops |
| 06 full-model | The whole model end to end: embeddings, layer stack, LM head, generation |
| 07 optimized-full-model | The full model traced and optimized, with honest perf accounting |
| 08 datatype-sweep | The fastest weight/activation/KV datatype config that still meets accuracy |
| 09 vllm | The model serving real requests through the Tenstorrent vLLM plugin |
| 10 optimized-vllm | The serving path optimized: async decode, trace reuse, on-device sampling |
| 11 tti-release | The tt-inference-server release workflow run and customer-facing readiness report |

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
it. Fused-decoder, optimized-decoder, multichip, full-model,
optimized-full-model, datatype-sweep, vLLM, and release stages update or verify
it because graph changes, tensor parallelism, full-stack memory use, and
KV-cache dtype can change the feasible context.

Each stage leaves its evidence under `models/autoports/<model>/doc/<stage>/`:
a `README.md` with the results, a `work_log.md` with the journey, and the
artifacts (perf reports, accuracy logs, watcher output) that back them up.

## Verification Gates

A stage README and work log are claims to inspect, not verification by
themselves. After a stage's goal completes, the runner looks for a sibling
check script (for example `06-full-model.check.sh` next to
`06-full-model.txt`) and runs it. Check scripts are plain bash, readable by
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
- `--start-index N` — assign stage number N to the first prompt file supplied.
  It does not skip files: to resume without repeating earlier stages, pass only
  the prompt files beginning at stage N. This starts a fresh thread for stage N.
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

## Other Tracks

`prompts/model_bringup_multigoal/` is for autoregressive LLM decoders. Two other
pipelines share the same runners, prompts format, and gate convention:

- **`prompts/diffusion_bringup_multigoal/`** brings up a diffusion model
  (image, video, or audio DiT plus VAE plus text encoder) in `models/tt_dit/`.
  Diffusion has no KV cache, no paged decode, and no token loop, so the LLM
  stages do not apply. Ten stages: one DiT block, full DiT, text encoder, video
  VAE, audio VAE, scheduler, full pipeline, multichip, optimize, datatype sweep.
  Its own skills are the `diffusion-*` ones plus `$functional-dit-block`,
  `$adaln-conditioning`, `$multiaxis-rope`, `$text-encoder-port`, `$vae-port`,
  and `$denoise-loop-scheduler`. See that directory's `README.md`.
- **`prompts/forge_goals/`** starts a decoder bringup from compiler output
  instead of from scratch: `01-forge-functional-decoder-from-emit.txt` from a
  pre-generated EmitPy emit, and `02-forge-functional-decoder-from-ir.txt` from a
  TTNN IR (`.mlir`) dump via `$forge-functional-decoder-from-ir`. Both are
  drop-in replacements for stage 01.

### Gates shipped inactive

Two check scripts are present but named `*.check.sh.disabled` so the runner does
not pick them up. Each needs something before it can be trusted. Drop the
`.disabled` suffix to activate.

- `03-optimized-decoder.check.sh.disabled` requires the stage to have run
  `$shard-advise` and captured `report.json` plus `final_ir.mlir`. Activating it
  makes the sharding advisor mandatory, which needs the separate tt-mlir setup in
  `skills/shard-advise/SETUP.md`.
- `08-datatype-sweep.check.sh.disabled` expects the sweep to run *after* the TTI
  release stage: it requires `doc/tti_release/post_release_sweep_benchmark.json`.
  In the stage order above the sweep runs at 08 and release at 11, so that
  handoff does not exist yet. Reorder the stages or relax the check first.

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
