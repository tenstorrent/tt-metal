# Gated model bring-up

A framework for bringing up chunked prefill of a transformer on Tenstorrent hardware in fixed, gated steps. Agents do
the creative work, one fresh `claude -p` per step. Tests that were written, checked against the CPU reference and frozen
before any implementation exists decide whether a step passed. Overview: `docs/pipeline_design.html`, published at https://claude.ai/artifact/L5rXDnJoEpjsEL33s3wmSC.
The ERNIE-4.5 bring-up (`models/demos/ernie45_d_p/`) is the run the design came from.

## Start

Use the `/bringup` intake skill (`skill/bringup/SKILL.md`), or by hand:

```bash
export PYTHONPATH=$PWD          # the shell's default points at another checkout
B="python -m models.demos.common.bringup"
$B new --model <slug> --hf-id <org/name>                 # scaffold models/demos/<slug>/bringup/
# fill models/demos/<slug>/bringup/spec.yaml (template comments say what each field is)
$B approve intake --spec models/demos/<slug>/bringup/spec.yaml
python -m models.demos.common.bringup.plan.ledger_gen --spec <spec> --early --write
python -m models.demos.common.bringup.orchestrator run --spec <spec>
```

The orchestrator stops with exit 3 when a person is needed (approve the plan, pick performance items) and exit 1 when a
task is STOPPED after the implementer and the debugger both failed three times. `orchestrator resume` continues.

## Steps and task ids

| Step | Tasks | Who | Gate |
|---|---|---|---|
| intake | spec.yaml | person + `/bringup` | `approve intake` |
| reference | R.1 checkpoint, R.2 HF parity, R.3 chunked + graph replay | agent (reference role) | PCC vs HF >= 0.9999, chunked == one-shot, graph replays exactly |
| goldens | G.<rung> | script | manifest with content hash, every layer and chunk |
| box | B.1 | script | mesh opens, collectives exact |
| plan | PL.0 ledger, PL.1 plan | agent (plan role) + person | memory computed from the checkpoint fits per-chip DRAM, every step mapped, approved |
| implement | C.<block>.<step>, S.<block>.<nn> | agent (test role, freeze, implement role) | frozen component test, then the swap order |
| integrate | L.<rung> | script (fix agent on failure) | per-layer trail, state, final hidden, top-5 |
| contract | K.1 | agent (contract role) | engine API: layout, table, ack timing, engine input, producer read-back |
| perf | X.1 profile, X.2 opportunities, picked items | script + person | warm profile; each pick: faster and every accuracy gate still passes |

## Layout

| Path | What |
|---|---|
| `core/` | spec, ledger (tasks.yaml + state.json + lock), gate runner, metrics, freeze, runs (resume, rerun, fork, compare) |
| `reference/` | reference interface and block-graph runner, HF parity, chunked check, golden generator and reader |
| `testing/` | component and swap tests, ladder, serving contract, profiler, test templates |
| `tests/` | generic pytest entry points (box, ladder, contract, profile); they read `BRINGUP_SPEC` |
| `plan/` | memory check, components map, approvals, ledger generator, opportunity list |
| `intake/` | checkpoint check |
| `knowledge/` | repo map and known issues every agent reads first, and their format check |
| `orchestrator.py`, `agents/`, `briefs/` | the run loop, the agent definition (passed with `--agents`), brief templates |
| `dashboard/` | the dashboard exporter and page |
| `skill/bringup/SKILL.md` | the intake skill (install under `.claude/skills/` to use it as `/bringup`) |
| `selftest/` | the framework's own tests (CPU only, run with `scripts/run_safe_pytest.sh --no-precompile`) |
| `dev/` | the ledger of the framework's own build (`[bringup][F<n>]` commits) |

Everything outside git lives under `/localdev/$USER/bringup/<model>/`: `hf/`, `golden/`, `tt_cache/<mesh>/<version>/`,
`profiles/`, `runs/<run>/` (logs, briefs, agent transcripts).

## Rules the runner enforces

- A gate passes only if deps passed, frozen files are unchanged, a device gate uses `scripts/run_safe_pytest.sh` or
  `scripts/tt-probe.sh`, the command exits 0, every metric meets its threshold and every artifact exists.
- A task with tests fails until its tests are frozen; freezing requires PASS with the CPU reference and FAIL with a zero stub.
- An agent step fails if the tree changed outside the brief's paths, if any command reached the device without a safe
  runner, or if the known-issues file lost its format.
- Commits stage only the task's paths; formatting runs before testing and hashing, so the tested bytes are committed.
- The plan gate recomputes memory from the checkpoint's tensor shapes; approvals are voided by any edit to what was approved.
