<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Bringup Orchestrator — Per-Tick Prompt

You are running ONE tick of the bringup orchestrator. A tick is the
atomic unit: load state, decide one action, dispatch one or more
workers, parse results, mutate state, render the log, commit, then
schedule the next tick. One tick = one commit. Nothing is committed
until Step 6, so a mid-tick crash leaves the repo clean and the next
`/bringup --resume` picks up from the prior tick's commit.

Do NOT call `git reset --hard`, `git checkout .`, or any other
destructive git command. If something goes wrong, halt and let the user
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

Then ONE Agent call with `workers/<result['worker']>-worker.md` (one of
`ttnn`, `debug`, `optimization`). Spec uses `phase=result["worker"]`,
`block=result["block"]`, and `history` filled from the component's prior
attempt count + most recent `last_error`. Only one device worker per
tick.

## Step 4: Parse worker results

Each worker's response ends with one JSON line:

```python
import json
result_json = json.loads(response.strip().splitlines()[-1])
```

Required keys: `block`, `phase`, `status` (one of `ok`/`fail`/`blocked`),
`pcc`, `artifacts`, `notes`, `last_error`, `hang_detected`.

### Architecture, status=ok

Read `models/demos/<model_slug>/architecture_inventory.json`. For each
entry, append to `state["components"]`:

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
}
```

### Reference / device, per result

Locate `component = next(c for c in state["components"] if c["name"] ==
result_json["block"])` and the phase key (one of `reference`, `ttnn`,
`debug`, `optimization`):

- `status="ok"` → `component[phase] = {"status": "done", "pcc":
  result_json["pcc"], "attempts": old_attempts, "artifacts":
  result_json["artifacts"], "notes": result_json.get("notes", "")}`.
- `status="fail"` → `attempts = old_attempts + 1`. If `attempts >=
  state["config"]["max_attempts_per_phase"]` set status to `blocked`,
  else `failing`. Carry `last_error=result_json["last_error"]` and
  `pcc=result_json["pcc"]`.
- `status="blocked"` → preserve attempts; record `last_error`.

### Device-worker cleanup (ttnn/debug/optimization)

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

## Step 5: Guard check for TTNN successes

When the dispatched worker was `ttnn` AND it returned `status="ok"`,
run the no-shortcuts guard BEFORE marking `ttnn.status="done"`:

```python
from skills.orchestrator.lib.guard import verify_block, lint_block

block_file = f"models/demos/{state['model_slug']}/tt/{result_json['block'].lower()}.py"
traced = result_json.get("traced_ops") or []
if not traced:
    import json as _json
    side = f"models/demos/{state['model_slug']}/tt/{result_json['block'].lower()}.traced_ops.json"
    try:
        with open(side) as f: traced = _json.load(f)
    except FileNotFoundError: traced = []

component = next(c for c in state["components"] if c["name"] == result_json["block"])
verdict = verify_block(block_file, traced, component["kind"], component["reference_impl"])
```

`verify_block` composes `lint_block`, `assert_traced_ops`, and
`cross_check_reference` and exposes `.ok`. If `verdict.ok is False`,
treat the result as `status="fail"` with `last_error="no-shortcuts
guard: " + summary` and re-apply the standard fail handling
(attempts++, blocked at max). Otherwise mark ttnn done.

## Step 6: Render log and commit

Append THIS tick's entry to `tick_log`, then render and commit:

```python
import datetime
now = datetime.datetime.now(datetime.timezone.utc).isoformat(
    timespec="seconds").replace("+00:00", "Z")
state["tick_log"].append({
    "tick": tick_id, "ts": now,
    "action": f"{result['phase']}[{result_json.get('block', '-')}]",
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

## Step 7: Schedule next tick (or exit)

Re-check eligibility AFTER mutations:

```python
phase = eligible_blocks(state)["phase"]
```

- `phase == "done"` — print a completion summary and DO NOT call
  `ScheduleWakeup`. This is the final tick.
- `phase == "deadlock"` — already handled in Step 3; exit.
- otherwise:

```python
ScheduleWakeup(
    delaySeconds=state["config"]["tick_interval_sec"],
    prompt="/bringup --resume <model_path>",
    reason=f"tick {tick_id} done; next: {phase}",
)
```

Then exit. The harness rewakes you at the configured interval.

## Failure handling

- **Subagent crash** (not a status=fail JSON, but a hard error): treat
  as `status="fail"` with `last_error="subagent crash: <stderr>"`.
  Apply standard fail handling, release any device lock, continue to
  Step 6 — the tick MUST still commit.
- **`git commit` fails** (pre-commit reformat): re-`git add` and create
  a NEW commit. Do not `--amend`. Do not skip hooks.
- **Malformed state file** (`SchemaError` in Step 1): emit
  `UserWarning`, do NOT `ScheduleWakeup`, halt. Operator repairs by hand.
- **`tt_smi_reset()` non-zero exit**: log the code in this tick's
  `notes`, still set `pending_smoke_check`, still rewake. Persistent
  reset failure surfaces via the smoke check's next-tick hang.

## Glossary

- **tick** — one cycle of this prompt. Atomic: in-memory mutation
  through Steps 1–5, committed exactly once in Step 6.
- **worker** — a sub-agent dispatched via `Agent`. One of:
  architecture, reference, ttnn, debug, optimization. Each returns a
  JSON line.
- **device lock** — `state["locks"]["device"]`. Field `device.held_by`
  is `"tick-<N>"` when held, `None` when free. Auto-cleared by
  `resume_normalize`. Released in Step 4 after the device worker returns.
- **pending_smoke_check** — set when a worker reports
  `hang_detected=True`. The next tick re-dispatches a ttnn test on
  that block to confirm the device is healthy post-`tt_smi_reset`.
- **last_error** — most recent failure string for a phase. Rendered in
  `BRINGUP_LOG.md` for `failing`/`blocked` phases by `render_log`.
- **attempts** — monotonic count of failed dispatches for a phase.
  Resets to zero on `redo`. At `>= max_attempts_per_phase` the phase
  auto-promotes to `blocked`.
- **max_attempts_per_phase** — `state["config"]["max_attempts_per_phase"]`.
  Default 10.
- **block** — one entry in `state["components"]`. Has phases
  reference, ttnn, debug, optimization.
- **phase** — one of `reference`, `ttnn`, `debug`, `optimization` for
  components, plus `architecture` (model-level). Architecture is the
  only phase that creates components.
