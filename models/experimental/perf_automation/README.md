# Agent Loop — handler guide

Welcome. If you're reading this, you're about to own a stage of the optimization
loop. This page tells you what the loop is, how to run it, and — the important
part — how to drop your work in without breaking anyone else's.

The short version: **the machine already runs end to end today.** It walks every
stage, reaches a terminal, writes a ledger, and passes its tests — using stand-in
"mock" handlers where the real work will go. Your job is to swap one mock for the
real thing, run the test, and watch it stay green. No big-bang integration day.

---

## 1. What the loop does

We point the tool at a model and a metric. The **Before Loop** profiles a clean
baseline and discovers the model's files, perf test, and PCC gate. Then the
**Agent Loop** takes over: it finds the slowest group of ops, asks an LLM which
optimization from the playbook to try, applies that one edit, checks the model is
still numerically correct (PCC), re-measures, and either keeps the change or
reverts it. Then it does it again. It stops when it hits your target, runs out of
budget, or runs out of ideas.

Exactly one step in that cycle is an LLM making a judgment call (which lever to
try). Everything else — profiling, routing, the PCC gate, the keep/revert
decision — is plain deterministic Python. That split is on purpose: the
interesting decisions get an agent, and everything we need to be reproducible and
resumable stays code.

## 2. Quick start (BGE-M3)

Run the Before Loop once to produce a baseline, then start the loop:

```bash
# 1) profile a clean baseline + discover the model (writes runs/<id>/)
python -m agent.before_loop \
    /localdev/gtobar/tt-metal/models/demos/wormhole/bge_m3 \
    --metric device_ms --target 11 --input 128

# 2) run the Agent Loop, picking up from runs/latest
python -m agent.loop
```

Right now step 2 runs the **mock skeleton** — it will march ROUTE → SELECT →
APPLY → … → DONE in a couple of fake iterations without touching the device or
spending a cent. That's intentional. It proves the wiring works before any real
handler exists. As you replace mocks with real handlers, the same command starts
doing real work.

Want to see the machine move without even running the Before Loop? The test does
it on a fixture:

```bash
pytest tests/test_engine.py -q -o addopts=
```

## 3. How it actually works (the 60-second mental model)

There is no big loop function. There's a **dumb dispatcher** (`agent/engine.py`):

```python
while state not in TERMINAL:
    next_state = HANDLERS[state](ctx)   # a handler does its work, returns the next state's name
    save(next_state)                    # checkpoint after every step
    state = next_state
```

That's the whole engine. Each **handler** is a function `handler(ctx) -> "NEXT_STATE"`.
The loop you see in the diagram is an *emergent* property: `GATE_PCC` returns
`"REPAIR_PCC"`, `REPAIR_PCC` returns `"VERIFY"`, `VERIFY` returns `"GATE_PCC"` —
they point at each other, so the engine walks in a circle until a counter says
stop. The engine never "decides to loop"; it just keeps calling the next handler.

Two consequences worth internalizing:

- **State lives on disk, not in a variable.** Counters like `pcc_fix_attempts`
  live in `state.json`. Kill the process mid-repair, restart, and the engine
  reads the last state and continues *exactly* where it was. Free crash-resume.
- **Handlers never open files directly.** Everything goes through `ctx` (a
  `LoopContext`): `ctx.state` is the live checkpoint dict, `ctx.manifest` is the
  immutable run config, `ctx.baseline_profile()` gives you the tagged buckets,
  `ctx.ledger.append(row)` records an experiment, `ctx.record_agent_call(...)`
  logs tokens + cost. So "what ROUTE hands to SELECT" is just `ctx.state[...]`.

## 4. The walking skeleton: what's real, what's mock

| Real today | Mock today (your job) |
|---|---|
| `ROUTE` — picks the top bucket, routes to candidate levers | `SELECT`, `APPLY`, `VERIFY`, `REPAIR_CODE`, `REPAIR_PCC` |
| `LOG` + `CHECK_EXIT` — writes the ledger row, decides continue/stop | `GATE_PCC`, `REMEASURE`, `DECIDE`, `COMMIT`, `REVERT` |

`ROUTE` and `LOG` are your templates — read them first (`route.py`, `log_exit.py`).
They're short, real, and show the shape every handler should take. The mocks all
live in `mocks.py` with the control flow already correct.

## 5. Claiming a stage

Here's the move, start to finish. We'll make `GATE_PCC` real for BGE-M3.

**Step 1 — find your leaf.** Open `mocks.py`, find `gate_pcc`. Notice the routing
is already written and correct:

```python
def gate_pcc(ctx) -> str:
    v = _measure_pcc(ctx)                 # <-- this leaf is fake; everything below is real
    ctx.state["last_verdict"] = v
    if v["status"] == "ok":        return states.REMEASURE
    if v["status"] == "pcc_low":
        ...                               # routes to REPAIR_PCC or REVERT based on the counter
    ...
```

You are **only** replacing `_measure_pcc`. Do not touch the `return states.X`
lines — that routing is the design, and it's what keeps the state graph correct
while everyone works in parallel.

**Step 2 — implement the real leaf.** Make your own file, `agent/handlers/gate_pcc.py`,
and write the real measurement. For BGE-M3, discovery already put the e2e test and
its threshold in the manifest, so you read them straight off `ctx`:

```python
# agent/handlers/gate_pcc.py
from .. import states

def _measure_pcc(ctx) -> dict:
    pcc_entry = ctx.manifest["pathmap"]["pcc"]["end_to_end"]
    test, threshold = pcc_entry["path"], pcc_entry["threshold"]   # e.g. ".../test_bge_m3.py::test_e2e", 0.99
    try:
        measured = run_pcc_test(ctx, test)        # your code: run pytest, parse the PCC number
    except Exception as exc:
        return {"status": "crash", "error": str(exc)}             # crash -> code-repair
    if measured >= threshold:
        return {"status": "ok", "pcc": measured}
    return {"status": "pcc_low", "pcc": measured}                 # too low -> pcc-repair

def gate_pcc(ctx) -> str:
    ...  # copy the routing from mocks.gate_pcc verbatim, now calling your real _measure_pcc
```

**Step 3 — wire it in.** One line in `__init__.py`:

```python
from . import gate_pcc as _gate_pcc          # add this import
...
states.GATE_PCC: _gate_pcc.gate_pcc,         # was: mocks.gate_pcc
```

**Step 4 — stay green.** Run the skeleton test. It should still walk to a
terminal, now with your real PCC gate in the path:

```bash
pytest tests/test_engine.py -q -o addopts=
```

That's the whole rhythm. Repeat per stage. Because each stage is its own file and
the only shared touch-point is that one registry line, two people can do this at
once without colliding.

## 6. The four rules that keep integration painless

1. **Signature is sacred:** `handler(ctx) -> next_state_string`. Always a name
   from `agent/states.py` (or a terminal).
2. **Fill the leaf, never the routing.** The `return states.X` lines encode the
   state machine. If you change where a stage goes, you've changed the design —
   talk to the team first.
3. **One file per stage.** The shared files are tiny on purpose: `states.py`
   (names) and `handlers/__init__.py` (the registry). Everything else is yours.
4. **Talk through `ctx`, not the filesystem.** Read `ctx.state` / `ctx.manifest`,
   write `ctx.state` / `ctx.ledger`. Then resume and telemetry just work.

## 7. The repair loop (so the inner cycle makes sense)

Between APPLY and REMEASURE the loop can self-heal. Two budgets, by failure type:

- The agent wrote code that won't parse / import / run → that's mechanical, give
  it room: up to **5** `REPAIR_CODE` attempts, then abandon the lever.
- The edit ran fine but PCC dropped below the threshold → that's a correctness
  dead-end, give it a little: up to **2** `REPAIR_PCC` attempts, then discard.

Both counters live in `state.json` and reset when `SELECT` picks a new lever. The
budgets are in `states.py` (`MAX_CODE_FIX`, `MAX_PCC_FIX`).

## 8. When something looks wrong

| You see… | It usually means… |
|---|---|
| `EngineError: no handler registered for state 'X'` | you renamed a state or forgot to register it in `__init__.py` |
| `EngineError: returned unknown state 'X'` | a handler returned a typo'd name — use the `states.*` constant, not a string literal |
| `EngineError: exceeded N steps` | a handler routes in a circle with no counter to break it — check your `return` |
| `ValueError: invalid value '...' for dimension` | a bucket tag isn't in the routing vocabulary (`router.VOCABULARY`) — e.g. compute-bound is `flop`, not `compute` |
| loop ends `STOPPED` immediately | all candidates already `tried`, or budget/max-iter hit — check `state.json` |

## 9. Commands & flags

| Command | What it does |
|---|---|
| `python -m agent.before_loop <model_dir> --metric device_ms --target 11 --input 128` | profile baseline + discover the model |
| `python -m agent.loop [runs_root]` | run the Agent Loop from `runs/latest` (default `runs/`) |
| `pytest tests/test_engine.py -q -o addopts=` | walk the skeleton to DONE on a fixture |
| `python -m agent.before_loop --help` | full flag reference (metric, devices, input matching, mock toggles) |

`--help` on the Before Loop is the source of truth for its flags — `--input`
matching, `--devices`, `--metric`, and the `--mock-*` toggles for hardware-free
testing are all documented there.

## 10. Where things live

```
agent/
  engine.py        the dispatcher (don't edit — it's done)
  states.py        state names + TRANSITIONS contract + repair budgets
  loop_context.py  the ctx seam: state, manifest, ledger, telemetry helpers
  loop.py          `python -m agent.loop` entry point
  handlers/
    __init__.py    the registry — swap a mock for your module here, one line
    route.py       REAL — your template (Member 1)
    log_exit.py    REAL — your template (Member 2)
    mocks.py       stand-ins; copy the routing, replace the leaf
runs/<id>/         one run: state.json, manifest.json, ledger.jsonl, profiles/
tests/
  test_engine.py   the walking-skeleton test you keep green
  fixtures/loop/   the entry fixtures (after_before_loop/)
```

For the full design — every stage's inputs/outputs, the fixtures for working
offline, and the open TBDs — see `PLAN_AGENT_WORKFLOW.md` (§8 is the loop).
