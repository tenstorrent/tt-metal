<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Bringup Orchestrator — Per-Tick Prompt

You are running ONE tick of the bringup orchestrator. A tick is
atomic: load state → decide one action → dispatch worker(s) → parse →
mutate → render → commit → schedule the next tick. One tick = one
commit. Nothing is committed until Step 6, so a mid-tick crash leaves
the repo clean and `/bringup --resume` picks up from the prior tick.

Do NOT call `git reset --hard`, `git checkout .`, or any other
destructive git command. On unrecoverable error, halt and let the user
inspect.

## Args

You receive `<model_path>` as the first positional argument — the
directory under `models/demos/` that holds `.bringup_state.json` and
`BRINGUP_LOG.md` (e.g. `models/demos/qwen3_tts`).

## Step 1: Load state

```bash
cd /local/ttuser/ssinghal/tt-metal
source python_env/bin/activate
export PYTHONPATH=$(pwd) && export TT_METAL_HOME=$(pwd)
python -c "
import json
from skills.orchestrator.lib.state import load_state, resume_normalize, save_state
state = load_state('<model_path>/.bringup_state.json')
resume_normalize(state)
save_state('<model_path>/.bringup_state.json', state)
print(json.dumps(state, indent=2))
"
```

Hold the printed state in your working context. `resume_normalize`
demotes any `in_progress` phases to `pending` and clears a stuck device
lock — safe to run every tick. If `load_state` raises `SchemaError`,
halt with a `UserWarning`; do NOT `ScheduleWakeup`.

## Step 2: Decide next action

```python
from skills.orchestrator.lib.dag import eligible_blocks
result = eligible_blocks(state)
print(result["phase"])  # architecture|reference|device|done|deadlock
```

Branch on `result["phase"]`:

- `done` — skip to Step 7 (no rewake).
- `deadlock` — render log with a `## Deadlock` section listing
  `result["blocking"]`, commit, exit without rewake.
- `architecture` | `reference` | `device` — proceed to Step 3.

Override: if `state.get("pending_smoke_check")` is set (left by a prior
hang), dispatch ONE ttnn-worker for that block as a smoke check
(re-running the existing PCC test) instead of consulting
`eligible_blocks`. Clear the field after dispatch.

## Step 3: Dispatch worker(s)

Compute this tick's id once:

```python
tick_id = max((e.get("tick", 0) for e in state["tick_log"]), default=0) + 1
```

Every dispatch uses `Agent(subagent_type="general-purpose", prompt=
<worker.md contents> + "\n\n## Spec:\n" + json.dumps(spec))`. The spec
shape is documented in each worker file.

### phase == "architecture"

ONE Agent call with `workers/architecture-worker.md`. Spec:
`{"block": "all", "phase": "architecture", "model_slug": ..., "model_id":
..., "device": ..., "arch_name": ..., "reference_impl": "",
"depends_on_status": {}, "config": state["config"], "history":
{"attempts": 0, "last_error": None}}`.

### phase == "reference"

`result["blocks"]` is the list of component names (already capped at
`config.max_parallel_reference`). For each name, build one spec with
`phase="reference"`, `block=<name>`, the component's `reference_impl`,
and `depends_on_status` derived from its `depends_on` list. Dispatch
ALL workers in a SINGLE message with multiple parallel Agent tool calls
(see `superpowers:dispatching-parallel-agents`). Wait for all to
return; do NOT serialize.

### phase == "device"

Acquire the device lock BEFORE the dispatch — i.e. set
`device.held_by = "tick-<N>"` on `state["locks"]["device"]`:

```python
import datetime
now = datetime.datetime.now(datetime.timezone.utc).isoformat(
    timespec="seconds").replace("+00:00", "Z")
state["locks"]["device"]["held_by"] = f"tick-{tick_id}"
state["locks"]["device"]["held_since"] = now
save_state("<model_path>/.bringup_state.json", state)
```

Then ONE Agent call with `workers/<result['worker']>-worker.md`. Only
one device worker per tick. The spec is JSON-serialized after a
`## Spec:` line. Common base fields: `phase=result["worker"]`,
`model_slug`, `model_id`, `device`, `arch_name`, `config`,
`hf_checkpoint_path`. Worker-specific additions:

- `ttnn` / `debug` / `optimization` / `real_weights` (per-component):
  `block=result["block"]`, the component's `reference_impl`,
  `depends_on_status` derived from `depends_on`, `history` from the
  component's prior attempt count + most recent `last_error` on that
  phase row (`component[result["worker"]]`).
- `generation` (NEW, per-use_case): the eligible result carries
  `result["use_case"]` (not `block`). Look up `uc = next(u for u in
  state["use_cases"] if u["name"] == result["use_case"])`. Spec
  includes the full `use_case` dict plus `components=[{"name": n,
  "tt_path": f"models/demos/{slug}/tt/{n}.py"} for n in
  uc["components_used"]]` and `history` from `uc["generation"]`.
- `perf` (NEW, per-use_case): same `result["use_case"]` lookup; spec
  has the full `use_case` dict and `history` from `uc["perf"]` (no
  `components` list — perf re-imports what generation already wired).

Each holds the device lock for the single dispatch and releases it in
Step 4.

## Step 4: Parse worker results

Each worker's response ends with one JSON line:

```python
import json
result_json = json.loads(response.strip().splitlines()[-1])
```

Two backward-compatible shapes:

- **Old** (`reference`, `ttnn`, `debug`, `optimization`): `block`,
  `phase`, `status`, `pcc`, `artifacts`, `notes`, `last_error`,
  `hang_detected`.
- **New** (`real_weights`, `generation`, `perf`): `target`,
  `target_type` (`"component"` or `"use_case"`), `phase`, `status`,
  `metric` (e.g. `{"pcc": 0.998}` / `{"bleu": 42.524}` /
  `{"steady_step_ms": 18.1, "speedup": 1.21}`), `artifacts`, `notes`,
  `last_error`, `hang_detected`.

Normalize once, then resolve the target dict:

```python
target = result_json.get("target") or result_json.get("block")
target_type = result_json.get("target_type", "component")
phase_name = result_json["phase"]
status = result_json["status"]
metric = result_json.get("metric") or (
    {"pcc": result_json["pcc"]} if "pcc" in result_json else {}
)
if target_type == "component":
    target_dict = next(c for c in state["components"] if c["name"] == target)
elif target_type == "use_case":
    target_dict = next(uc for uc in state["use_cases"] if uc["name"] == target)
else:
    raise AssertionError(f"unknown target_type {target_type!r}")
```

### Architecture, status=ok

Read `models/demos/<model_slug>/architecture_inventory.json`. For each
`components[]` entry append to `state["components"]`:

```python
{
  "name": entry["name"], "kind": entry["kind"],
  "depends_on": entry.get("depends_on", []),
  "reference_impl": entry["reference_impl"],
  "host_resident": entry.get("host_resident", {"allowed": False}),
  "reference":    {"status": "pending", "attempts": 0},
  "ttnn":         {"status": "pending", "attempts": 0},
  "debug":        {"status": "n/a",     "attempts": 0},
  "optimization": {"status": "pending", "attempts": 0},
  "real_weights": {"status": "pending", "attempts": 0},
}
```

For each `use_cases[]` entry append to `state["use_cases"]` the entry
fields verbatim plus `"generation": {"status": "pending", "attempts": 0}`
and `"perf": {"status": "pending", "attempts": 0}`.

### Reference / device, per result

Helpers shared with Step 5 (`**metric` unpacks `{"pcc": ...}` /
`{"bleu": ...}` / `{"steady_step_ms": ..., "speedup": ...}` into the
row so it surfaces in `BRINGUP_LOG.md`):

```python
def _success_row(old_attempts, metric, r):
    return {"status": "done", "attempts": old_attempts,
            "artifacts": r.get("artifacts", []), "notes": r.get("notes", ""), **metric}
def _fail_row(old_attempts, last_error, metric, r, max_attempts):
    attempts = old_attempts + 1
    return {"status": "blocked" if attempts >= max_attempts else "failing",
            "attempts": attempts, "last_error": last_error,
            "artifacts": r.get("artifacts", []), "notes": r.get("notes", ""), **metric}
old_attempts = target_dict.get(phase_name, {}).get("attempts", 0)
max_attempts = state["config"]["max_attempts_per_phase"]
```

- `status="ok"` → `target_dict[phase_name] = _success_row(old_attempts,
  metric, result_json)`. **Exception:** for `phase_name in {"ttnn",
  "optimization", "real_weights", "generation", "perf"}` defer this
  write until after Step 5's guard passes; on guard rejection write
  `_fail_row` instead.
- `status="fail"` → `target_dict[phase_name] = _fail_row(old_attempts,
  result_json.get("last_error"), metric, result_json, max_attempts)`.
- `status="blocked"` → preserve attempts; record `last_error`, `notes`,
  any `metric` (one-shot blocker; do NOT increment attempts).

### Device-worker cleanup (ttnn/debug/optimization/real_weights/generation/perf)

ALWAYS clear the lock after a device worker returns:

```python
state["locks"]["device"]["held_by"] = None
state["locks"]["device"]["held_since"] = None
```

If `result_json["hang_detected"] is True`:

```python
from skills.orchestrator.lib.device import tt_smi_reset
tt_smi_reset()  # returns exit code; never raises
last_done = next((c["name"] for c in reversed(state["components"])
                  if c.get("ttnn", {}).get("status") == "done"), None)
if last_done:
    state["pending_smoke_check"] = last_done
```

The smoke check is dispatched on the NEXT tick (Step 2 override).

## Step 5: Guard check for device successes

For the guarded phases (`ttnn`, `optimization`, `real_weights`,
`generation`, `perf`) Step 4 deferred the success row. Compute `ok` +
`guard_summary` per the per-phase rule below, then write
`_success_row` if `ok` else `_fail_row(..., last_error=guard_summary,
...)`.

```python
from skills.orchestrator.lib.guard import (
    verify_block, lint_block, verify_use_case, verify_optimization_artifact,
)
slug = state["model_slug"]; ok, guard_summary = True, None
```

**`phase_name == "ttnn"` — no-shortcuts guard.** Lint + traced-op
assertion + reference cross-check via `verify_block`:

```python
block_file = f"models/demos/{slug}/tt/{target.lower()}.py"
traced = result_json.get("traced_ops") or []
if not traced:
    import json as _json
    side = f"models/demos/{slug}/tt/{target.lower()}.traced_ops.json"
    try:
        with open(side) as f: traced = _json.load(f)
    except FileNotFoundError: traced = []
v = verify_block(block_file, traced, target_dict["kind"], target_dict["reference_impl"])
ok = v.ok
guard_summary = None if ok else (
    f"no-shortcuts guard: lint={len(v.lint)} "
    f"missing_kernels={v.missing_kernels} new_host_ops={len(v.new_host_ops)}")
```

**`phase_name == "real_weights"` — lint + params-loaded sanity.**
Re-lint the block file (the loader edits may touch it) and require the
worker to report a positive parameter count in
`metric.params_loaded` (or `metric.num_params`):

```python
block_file = f"models/demos/{slug}/tt/{target.lower()}.py"
lint = lint_block(block_file)
params_loaded = metric.get("params_loaded") or metric.get("num_params") or 0
ok = (not lint) and params_loaded > 0
guard_summary = None if ok else f"real_weights guard: lint={len(lint)} params_loaded={params_loaded}"
```

**`phase_name == "generation"` — `verify_use_case`.** `target_dict`
here is the use_case row (has `components_used`, `hf_class`,
`validation_metric`):

```python
v = verify_use_case(
    model_path=f"models/demos/{slug}/tt/{target}_model.py",
    use_case=target_dict,
    demo_path=f"models/demos/{slug}/demo/demo_{target}.py",
    test_path=f"models/demos/{slug}/tests/test_e2e_{target}.py",
)
ok = v.ok
guard_summary = None if ok else "verify_use_case: " + "; ".join(v.issues)
```

**`phase_name == "optimization"` — lint + tracy artifact present.**
The optimization-worker must attach a TRACED tracy CSV path (the
artifact field `tracy_artifact`, or the first CSV in `artifacts`).
Bulk "at-ceiling" wave-offs without evidence are rejected.

```python
block_file = f"models/demos/{slug}/tt/{target.lower()}.py"
lint = lint_block(block_file)
v = verify_optimization_artifact(result_json, require_traced_path=True)
ok = (not lint) and v.ok
guard_summary = None if ok else (
    f"optimization guard: lint={len(lint)} artifact_issues={v.issues}"
)
```

**`phase_name == "perf"` — re-run e2e + tracy artifact for sub-pass 2.**
The perf skill explicitly allows "no improvement found" as `ok`, BUT the
"no improvement" verdict is only acceptable when backed by a TRACED
tracy CSV (untraced tracy is host-dispatch noise, not op-level
evidence). Both checks must pass:

```python
import subprocess
test = f"models/demos/{slug}/tests/test_e2e_{target}.py"
proc = subprocess.run(["pytest", test, "-v"], capture_output=True, text=True, timeout=1800)
parity_ok = proc.returncode == 0
v = verify_optimization_artifact(result_json, require_traced_path=True)
ok = parity_ok and v.ok
guard_summary = None if ok else (
    f"perf guard: parity_ok={parity_ok} artifact_issues={v.issues}; "
    f"tail={proc.stdout[-400:]!r}"
)
```

**Apply.** `target_dict[phase_name] = _success_row(old_attempts, metric,
result_json) if ok else _fail_row(old_attempts, guard_summary, metric,
result_json, max_attempts)`.

## Step 6: Render log and commit

Append THIS tick's entry to `tick_log`, then render and commit:

```python
import datetime
now = datetime.datetime.now(datetime.timezone.utc).isoformat(
    timespec="seconds").replace("+00:00", "Z")
state["tick_log"].append({
    "tick": tick_id, "ts": now,
    "action": f"{result['phase']}[{target or '-'}]",
    "result": result_json["status"],
})
state["updated_at"] = now
save_state("<model_path>/.bringup_state.json", state)
```

```bash
python -c "
from skills.orchestrator.lib.state import load_state, render_log
state = load_state('<model_path>/.bringup_state.json')
open('<model_path>/BRINGUP_LOG.md', 'w').write(render_log(state))
"

git add <model_path>/.bringup_state.json <model_path>/BRINGUP_LOG.md <any_touched_files>
git commit -m "$(cat <<'EOF'
bringup(<slug>): <phase> <block> <status>

Tick <tick_id>: <one-line summary>.
EOF
)"
```

Pre-commit hooks may reformat files. If commit fails, re-stage the
reformatted files and create a NEW commit. Never `--amend` and never
`--no-verify`.

## Step 7: Report next-tick intent (parent owns the wakeup)

Re-check eligibility AFTER mutations: `phase =
eligible_blocks(state)["phase"]`.

**You do NOT schedule the next tick yourself.** `ScheduleWakeup` is not
available inside a dispatched subagent's tool context, and `CronCreate`
is the wrong instrument (the parent /loop already owns pacing; a
subagent cron leaks a duplicate timer the parent has to find and cancel
every tick). Instead, the LAST LINE of your response — in addition to
the per-block/use_case result JSON written to state — must report the
next-tick intent so the parent /loop can schedule it:

```
NEXT_TICK: {"phase": "<device|reference|architecture|done|deadlock>", "block": "<name or null>", "worker": "<name or null>"}
```

- If `phase == "done"`: emit `NEXT_TICK: {"phase": "done"}` and print a
  completion summary. The parent will NOT reschedule — pipeline complete.
- If `phase == "deadlock"`: already handled in Step 3; emit
  `NEXT_TICK: {"phase": "deadlock", ...}` with the blocking set.
- Otherwise: emit the eligible block/worker so the parent knows what the
  next tick will dispatch.

Then exit. The parent /loop reads `NEXT_TICK`, and IT calls
`ScheduleWakeup(delaySeconds=state["config"]["tick_interval_sec"],
prompt="/bringup --resume <model_path>", reason=...)` — or omits it when
`phase == "done"`. Do not call `ScheduleWakeup` or `CronCreate` yourself.

## Failure handling

**A tick MUST complete synchronously and atomically.** Never background
work, never wait on an external event (no Monitor, no "I'll resume when
the tracy run finishes"), never return a partial result. Run every
worker dispatch to completion in this invocation. On ANY blocker —
worker error, tracy hang, tool failure, ambiguous result — you still
finish the tick: write `status="fail"` (or `"blocked"`) with the cause
in `last_error`, release the device lock, render, commit (Step 6), and
report `NEXT_TICK` (Step 7). A tick that returns without having
committed leaves the device lock held and the orchestrator wedged until
an operator releases it manually. The only non-committing exit is the
`SchemaError` halt below.

- **Subagent crash** (hard error, not a status=fail JSON): treat as
  `status="fail"` with `last_error="subagent crash: <stderr>"`. Apply
  standard fail handling, release device lock, continue to Step 6 —
  the tick MUST still commit.
- **Worker backgrounds or waits instead of finishing**: if a dispatched
  worker returns without a terminal result JSON (e.g. "waiting for the
  tracy event"), treat it as `status="fail"`,
  `last_error="worker did not complete synchronously"`, release the
  lock, commit, and report `NEXT_TICK` so the block is retried next
  tick. Do NOT adopt the worker's wait — the tick still completes now.
- **`git commit` fails** (pre-commit reformat): re-`git add` and create
  a NEW commit. Never `--amend`, never `--no-verify`.
- **Malformed state file** (`SchemaError` in Step 1): emit
  `UserWarning`, do NOT `ScheduleWakeup`, halt. Operator repairs.
- **`tt_smi_reset()` non-zero exit**: log the code in `notes`, still
  set `pending_smoke_check`, still rewake. Persistent reset failure
  surfaces via the smoke check's next-tick hang.

## Glossary

- **tick** — one cycle of this prompt. Atomic: in-memory mutation
  through Steps 1–5, committed exactly once in Step 6.
- **worker** — a sub-agent dispatched via `Agent`. One of:
  architecture, reference, ttnn, debug, optimization, real_weights,
  generation, perf. Old workers (architecture/reference/ttnn/debug/
  optimization) use legacy `{block, pcc, ...}`; new workers
  (real_weights/generation/perf) use `{target, target_type, phase,
  metric, ...}`. Step 4 normalizes both.
- **device lock** — `state["locks"]["device"]`. `held_by` is
  `"tick-<N>"` when held, `None` when free. Auto-cleared by
  `resume_normalize`. Released in Step 4 after the device worker.
- **pending_smoke_check** — set when a worker reports
  `hang_detected=True`. Next tick re-dispatches a ttnn test on that
  block to confirm the device is healthy post-`tt_smi_reset`.
- **last_error / attempts / max_attempts_per_phase** — most recent
  failure string + monotonic failed-dispatch count for a phase;
  promoted to `blocked` at `>=
  state["config"]["max_attempts_per_phase"]` (default 10). `attempts`
  resets to zero on `redo`.
- **block / use_case / phase** — a `block` is one entry in
  `state["components"]` with phases reference/ttnn/debug/optimization/
  real_weights. A `use_case` is one entry in `state["use_cases"]`
  with phases generation/perf. Plus the model-level `architecture`
  phase, which is the only one that creates components and use_cases.
