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

The orchestrator stops with exit 3 when a person is needed (approve the plan, pick performance items, decide a pick) and exit 1 when a
task is STOPPED after the implementer and the debugger both failed three times. `orchestrator resume` continues. A run
whose only unfinished tasks are DEFERRED ends with exit 0, "complete with N deferred" (below).

## Steps and task ids

| Step | Tasks | Who | Gate |
|---|---|---|---|
| intake | spec.yaml | person + `/bringup` | `approve intake` |
| reference | R.1 checkpoint, R.2 HF parity, R.3 chunked + graph replay | agent (reference role) | PCC vs HF >= 0.9999, chunked == one-shot, graph replays exactly |
| trim | R.4 (layer subsets only) | script | after the HF sanity and parity passed: only layers 0..last subset layer stay on disk, tensors byte-identical (F47) |
| goldens | G.<rung> | script | manifest with content hash, every layer and chunk |
| box | B.1 | script | mesh opens, collectives exact |
| plan | PL.0 ledger, PL.1 plan | agent (plan role) + person | memory computed from the checkpoint fits per-chip DRAM, every step mapped, approved |
| implement | C.<block>.<step>, S.<block>.<nn> | agent (test role, freeze, implement role) | frozen component test, then the swap order; a C task may end DEFERRED (op request accepted by the checker) |
| integrate | L.<rung> | script (fix agent on failure) | per-layer trail, state, final hidden, top-5 |
| contract | K.1 | agent (contract role) | engine API: layout, table, ack timing, engine input, producer read-back |
| perf | X.1 profile, X.2 opportunities, picked items | script + person | warm profile; each pick: the owner decides on its A/B report, then its gate (below) |

## Steps TTNN has no op for (F46)

An implement agent that finds no proper TTNN op for a component step (no op, no composition, no fork fits) may defer it
to the op code generator (op-gen, `tt_metal/third_party/tt_ops_code_gen`); the plan may also tag it `OPGEN` in
components.yaml. The agent writes `<bringup_dir>/op_requests/<op>/` (`plan/op_request.py new`: request.yaml with the
evidence, op_prompt.txt, feature_spec.py, reference.py, bind.py); when its gate fails and `plan/op_request.py check`
passes, the task becomes DEFERRED: dependents run, the step stays on the CPU reference, and a device model calls it
through `testing/cpu_bridge.py`, whose transfers are not `host_transfers_per_layer` (metrics `deferred_cpu_steps`,
`deferred_cpu_ms`). A rejected request is a failed attempt. Launching op-gen is always the owner's call:

```bash
$B op-requests --spec <spec>                                  # requests and their status
$B approve op-request <op> --spec <spec>                      # the owner's approval (an edit voids it)
$B op-export <op> --spec <spec>                               # prompt + golden suite into the op-gen tree; prints the
                                                              # next steps (submodule commit, gitlink, push, run_eval.py)
$B op-ready <op> [<op> ...] --from <clone>/ttnn/ttnn/operations --spec <spec>   # ttnn.bringup.<op>, tasks reset
python -m models.demos.common.bringup.orchestrator resume --spec <spec>
```

## Perf picks: the owner decides on an A/B report (F57)

A pick (step `perf`, role `perf`) has no gate before or after its agent. The agent makes the change behind a switch
and runs the gate's frozen tests once, accuracy first: at the first failure it stops and writes the failing metrics
into BREADCRUMBS (`testing/profile.py` refuses to profile a pick whose frozen test failed in the attempt,
`testing/accuracy_guard.py`). Then the orchestrator measures the change off (the task's `ab.env`, the old path) and
on: every frozen test of the gate, the ladder at `ab_rungs` (default: the gate's rung and the spec's last rung) and
one plain profile, each with its own results dir under `runs/<run>/ab/<task>/<old|new>/`. It writes `table.md` there
(component checks, min layer PCC, final hidden, logits PCC, top1, top5, device ms total and of the gated section,
whether the gate would pass) and the record to state.json, and waits for the owner (exit 3):

```bash
O="python -m models.demos.common.bringup.orchestrator"
$O decide --task P.3 --accept --spec <spec>   # resume runs the gate (as tasks.yaml has it) and commits on PASS
$O decide --task P.3 --reject --spec <spec>   # resume reverts the change under the task's paths: REJECTED, dependents run
$O decide --task P.3 --rerun --spec <spec>    # resume measures again (e.g. after a board reset)
$O resume --spec <spec>
```

If the owner accepts a change that fails a frozen test, dropping that test from the pick's gate in tasks.yaml is
their edit (as for Xing P.1). A new `brief` or `ab` voids the report: the agent runs again.

```yaml
- id: P.3
  step: perf
  role: perf
  ab: {env: {XING_EXPERTS_FIDELITY: hifi4}}   # the old path
  ab_rungs: [last, s56320]                     # optional
```

## Layout

| Path | What |
|---|---|
| `core/` | spec, ledger (tasks.yaml + state.json + lock), gate runner, metrics, freeze, runs (resume, rerun, fork, compare) |
| `reference/` | reference interface and block-graph runner, HF parity, chunked check, golden generator and reader |
| `testing/` | component and swap tests, ladder, serving contract, profiler, test templates |
| `tests/` | generic pytest entry points (box, ladder, contract, profile); they read `BRINGUP_SPEC` |
| `plan/` | memory check, components map, approvals, ledger generator, opportunity list, op requests and their export |
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
- A swap test gates every swapped step vs the CPU step on the same inputs (`checks="steps"`, F49), so swap tests are
  frozen without a test-role review unless `agents.swap_review` names the block type. `BRINGUP_IMPL=mutate:<kind>`
  (testing/mutate.py) proves on the CPU that a test catches a wrong module.
- A component test runs built-in checks chosen by its output kind, on the golden and on second inputs
  (`checks="auto"`, F56, testing/component_checks.py). It is frozen without a test-role review only when
  `agents.component_review` leaves its block type out (default `all`: reviewed) and its CPU mistake sweep
  (`BRINGUP_IMPL=mutations`) catches every standard mistake; otherwise the review starts with the sweep's log.
- An agent step fails if the tree changed outside the brief's paths, if any command reached the device without a safe
  runner, or if the known-issues file lost its format.
- Commits stage only the task's paths; formatting runs before testing and hashing, so the tested bytes are committed.
- The plan gate recomputes memory from the checkpoint's tensor shapes; approvals are voided by any edit to what was approved.
