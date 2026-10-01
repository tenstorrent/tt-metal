# Gated model bring-up

Brings up chunked prefill of a Hugging Face transformer on Tenstorrent hardware in fixed, gated steps, built from the
start for the way the inference server (tt-d-gen) drives it. Agents do the creative work, one fresh `claude -p` per
step; tests that were frozen before any implementation existed decide whether a step passed.

**What it is and how it works** (roles, the step loop, the serving contract, perf, runs so far):
https://claude.ai/artifact/L5rXDnJoEpjsEL33s3wmSC. This file is the how-to.

## Before you start

- Run from the repo root with `export PYTHONPATH=$PWD` (the shell's default points at another checkout) and
  `source python_env/bin/activate`.
- A clone of tt-d-gen at `/localdev/$USER/tt-d-gen` (the serving-contract step reads it at its latest commit).
  Optional: `github.com/AleksKnezevic/disagg_lb` next to it, for the settings proven on LoudBoxes.
- Large files go under `/localdev/$USER/bringup/<model>/` (checkpoint, goldens, weight cache, runs). If your home
  quota is small, run the orchestrator with `TT_METAL_CACHE=/localdev/$USER` so JIT builds land there too.
- To use the skill as `/bringup`, link `skill/bringup` into `~/.claude/skills/`; link the files in `agents/` into
  `~/.claude/agents/` to start them by name from a session.

## Start

In Claude Code type `/bringup` and name the model: the skill interviews you, writes the spec, launches the run and
supervises it. By hand:

```bash
B="python -m models.demos.common.bringup"
O="python -m models.demos.common.bringup.orchestrator"
$B new --model <slug> --hf-id <org/name>       # or: $B new --prior <slug> --mesh R,C (same checkpoint, new mesh)
# fill models/demos/<slug>/bringup/spec.yaml (the template's comments say what each field is)
$B approve intake --spec <spec>
python -m models.demos.common.bringup.plan.ledger_gen --spec <spec> --early --write
$B init-run run1 --spec <spec>
$O run --spec <spec> >> /localdev/$USER/bringup/<slug>/runs/run1/orchestrator.log 2>&1 &
python -m models.demos.common.bringup.dashboard.export --spec <spec>   # dashboard/index.html (+ teletext.html)
```

Exit codes: 0 done (or "complete with N deferred"), 3 waiting for a person, 1 a task STOPPED after its retries.
`$O resume --spec <spec>` continues after any of them; `$O pause --spec <spec>` stops before the next task.

## When the run waits for you

| Point | What you decide | Command |
|---|---|---|
| intake | the spec | `$B approve intake --spec <spec>` (any spec edit voids it) |
| after SC.1 | the serving contract's questions (KV dtype for the decode side, slots, prefix reuse) | answer in the spec's `serving:` section, re-approve the intake |
| PL.1 | the sharding plan | `$B approve plan --spec <spec>` |
| X.2 | which perf opportunities to try | add `P.*` tasks (one per pick, with `ab.env` / `ab.change`), `$B approve perf` |
| each P.* | apply or not, from its A/B table (`runs/<run>/ab/<task>/table.md`) | `$O decide --task P.1 --accept\|--reject\|--rerun --spec <spec>` |
| a DEFERRED step | launch op-gen for it | below |
| a board hang | reset the board | then `$O resume` |

Other commands: `$B status`, `$B rerun --from <task>` (resets it and everything after it), `$B fork --from <task>
--name <run>`, `$B compare --spec <a> --other <b>`.

## Steps and task ids

| Step | Tasks | Who | Gate |
|---|---|---|---|
| intake | spec.yaml | person + `/bringup` | `approve intake` |
| reference | R.1 checkpoint, R.2 HF parity, R.3 chunked + graph replay | agent | PCC vs HF >= 0.9999, chunked == one-shot, graph replays exactly |
| trim | R.4 (layer subsets only) | script | only layers 0..last subset layer kept, tensors byte-identical (F47) |
| goldens | G.<rung> | script | manifest with content hash, every layer and chunk |
| box | B.1 | script | mesh opens, collectives exact |
| serving | SC.1 | `serving-contract` agent | `serving_contract.md` has every section and the tt-d-gen sha; every listed contract test exists; a runner test is listed (F58) |
| plan | PL.0 ledger, PL.1 plan | agent + person | memory fits per-chip DRAM, every step mapped, a `## Serving contract` section, approved |
| implement | C.<block>.<step>, S.<block>.<nn> | agent | frozen component test plus the step's contract tests, then the swap order; a C task may end DEFERRED |
| assemble | M.1 | agent | the all-device model, hidden state on the device |
| integrate | L.<rung> | script (fix agent on failure) | per-layer trail, state, final hidden, top-5, and the served KV format (`state_bits_*`) |
| contract | K.1 | agent | engine API (layout, table, acks, input, read-back) plus the runner contract tests |
| perf | X.1, X.2, P.*, X.3 | script + person + agent | profile; each pick decided by the owner on its A/B report; final 56k check |
| fork tests | O.1 | agent | a test case for every `ttnn.bringup` call the model makes |
| settings | Z.1 | `settings-audit` agent (smaller model) | settings lint over the whole model, an accuracy rung, all contract tests |

## Switches: one place each

- **Framework**: `defaults.yaml` holds every framework switch with its default (component and swap review, retry
  policy, thresholds, timeouts, contract and profile options). A model overrides a key in its `spec.yaml` under the same
  name, e.g. `agents.component_review: all`. Change a default only in that file; `selftest/test_defaults.py` fails on a
  module that keeps its own.
- **Model**: `models/demos/<model>/tt/settings.py`, a `Settings` table (`core/model_settings.py`): every fidelity,
  chunk size, dtype and implementation choice with its default, allowed values and the owner decision behind it.
  `<PREFIX><NAME>` environment variables override it for experiments and A/B reports. The orchestrator fails an agent
  step that reads the environment anywhere else in the model's code (`testing/settings_lint.py`); Z.1 moves anything
  left over. A value a gate checks (`serving.kv_dtype`) lives in the spec.
- **Precision rule**: component and swap tests always run at maximum precision (each precision switch's
  `max_precision` value in `tt/settings.py`, applied by `testing/model_precision.py`; switch
  `tests.component_max_precision`). Shipped precision, lowered by perf picks, is judged end to end by the ladder and
  the contract tests only.
- **Forks** (`ttnn/ttnn/bringup`): behaviour comes in as op arguments set from the model's settings; an environment
  knob only for diagnostics, marked `diagnostic` on its line.

## Serving contract (F58)

SC.1's agent (`agents/serving-contract.md`) writes, per model: `bringup/serving_contract.md` (a how-to per part of the
model, every rule cited to the tt-d-gen file it comes from), `bringup/contract_tests.yaml` (each test and the step whose
gate runs it: a component step name, or `adapter` for K.1) and the frozen tests in `tests/bringup/contract/`.
`ledger_gen` adds each test to its step's gate, and each step's brief carries its how-to section. To run them by hand:
`python -m models.demos.common.bringup.testing.serving --run <step|adapter|all>`.

## Perf picks (F57)

A pick has no gate before or after its agent. The agent makes the change behind a `tt/settings.py` switch and runs the
gate's frozen tests once; at the first failure it stops (nothing is profiled while a frozen test fails). The
orchestrator then measures the change off (`ab.env`) and on (`ab.change`), both set explicitly: every frozen test, the
ladder at `ab_rungs` (default: the gate's rung and the last rung) and one plain profile. It writes `table.md` and waits.
`decide --accept` runs the gate and commits; `--reject` reverts the task's files (REJECTED counts as done). Per-op
profiling runs on the representative layers only (`BRINGUP_PROFILE_LAYERS` picks others).

```yaml
- id: P.3
  step: perf
  role: perf
  ab: {env: {XING_EXPERTS_FIDELITY: hifi4}, change: {XING_EXPERTS_FIDELITY: hifi2}}
  ab_rungs: [last, s56320]   # optional
```

## Steps TTNN has no op for (F46)

An implement agent that finds no TTNN op, composition or fork for a component step may defer it to op-gen
(`tt_metal/third_party/tt_ops_code_gen`): it writes `<bringup_dir>/op_requests/<op>/`, the task becomes DEFERRED, and
the step runs on the CPU through `testing/cpu_bridge.py`. Launching op-gen is always the owner's call:

```bash
$B op-requests --spec <spec>                          # requests and their status
$B approve op-request <op> --spec <spec>              # an edit voids it
$B op-export <op> --spec <spec>                       # prompt + golden suite into the op-gen tree; prints next steps
$B op-ready <op> --from <clone>/ttnn/ttnn/operations --spec <spec>   # ttnn.bringup.<op>, tasks reset
$O resume --spec <spec>
```

## Layout

| Path | What |
|---|---|
| `defaults.yaml` | every framework switch and its default |
| `orchestrator.py` | the run loop |
| `agents/` | `bringup-engineer.md` (every step), `serving-contract.md` (SC.1), `settings-audit.md` (Z.1) |
| `briefs/` | the brief template and the per-role text |
| `core/` | spec, defaults, ledger, gate runner, metrics, freeze, runs, model settings helper |
| `reference/` | reference interface, block-graph runner, HF parity, golden generator and reader |
| `testing/` | component and swap tests, ladder, contract, serving contract, settings lint, profiler, templates |
| `tests/` | generic pytest entry points (box, ladder, contract, profile, positions); they read `BRINGUP_SPEC` |
| `plan/` | memory check, components map, approvals, ledger generator, opportunity list, op requests |
| `intake/`, `knowledge/` | checkpoint check; the repo map and known issues every agent reads first |
| `dashboard/` | the dashboard exporter and pages |
| `skill/bringup/SKILL.md` | the `/bringup` skill (intake and supervision) |
| `selftest/` | the framework's own tests, CPU only: `python -m pytest -q models/demos/common/bringup/selftest/` |
| `dev/BREADCRUMBS.md` | how the framework was built (F1 to F58) |

## Rules the runner enforces

- A gate passes only if deps passed, frozen files are unchanged, the device was used only through
  `scripts/run_safe_pytest.sh` or `scripts/tt-probe.sh`, the command exits 0, every metric meets its threshold and
  every artifact exists.
- A test freezes only after it passes with the CPU reference and fails on its mistake sweep. Component and swap tests
  freeze without a review agent when their sweep passes (`agents.component_review`, `agents.swap_review`).
- An agent step fails if the tree changed outside the brief's paths, a command reached the device without a safe
  runner, the known-issues file lost its format, or a changed file reads the environment outside `tt/settings.py`.
- With `serving.kv_dtype` set, every ladder rung must run on that KV format: one cache format for the ladder, the
  contract and serving.
- Commits stage only the task's paths; formatting runs before testing and hashing, so the tested bytes are committed.
- Approvals are voided by any edit to what was approved.
