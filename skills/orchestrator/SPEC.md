# SPEC: Bringup Orchestrator

**Status:** Design approved, not yet implemented.
**Date:** 2026-05-27
**Owner:** ssinghal@tenstorrent.com

## Purpose

Automate the full TTNN model bring-up pipeline — `architecture → reference → ttnn → debug → optimization` — for any HuggingFace model on any supported Tenstorrent device. Drive it autonomously via a thin tick-based orchestrator that dispatches sub-agents bound to the existing per-phase skills under `skills/`.

## Goals

- One command starts a bring-up: `/bringup <hf_model_id> --device <dev>`.
- Pipeline is fully resumable: `/clear`, session crash, machine reboot, or `tt-smi -r` does not lose progress.
- State lives on disk (`.bringup_state.json`), committed each tick. BRINGUP_LOG.md is a rendered view.
- Every block must end up on the TTNN device unless the chosen reference TTNN implementation also keeps that block host-resident (e.g. qwen3_tts speech tokenizer decoder).
- Bottom-up bring-up: small primitives (RMSNorm, Linear) finish before composite blocks (Attention, MLP), which finish before integrators (DecoderLayer, full model).
- Failures auto-route to a debug worker, retry up to 10× per phase per block, then mark `blocked` and continue independent work.

## Non-goals

- Wall-clock budgets / overnight timeouts. Stop signals are `attempts == max` or pipeline complete.
- Parallel TTNN workers. The device has contention; ttnn/debug/optimization phases are strictly serial across blocks.
- A new state machine that replaces or competes with `BRINGUP_LOG.md`. The log stays; it just gets auto-rendered.
- Cross-machine orchestration. Single host, single device or mesh selected at start.

---

## Architecture

Three layers:

### 1. `/bringup` skill (entry point)

Lives at `skills/orchestrator/SKILL.md`. Triggered by the user typing `/bringup ...`.

**First-run behavior** (no `--resume`):
1. Validate args, resolve `<hf_model_id>` → `model_slug` and `models/demos/<model_slug>/` folder.
2. If `.bringup_state.json` already exists at that path, refuse — direct user to `--resume`.
3. Create folder, write skeleton state, commit `chore(<slug>): start bringup`.
4. **Internally invoke `/loop` via the `Skill` tool**, passing `/bringup --resume models/demos/<slug>` as the sub-prompt. The user only types `/bringup` once; the loop is self-driven from there.
5. The first tick under `/loop` runs the architecture worker.

**Resume behavior** (`--resume <model_path>`):
1. Validate `.bringup_state.json` exists, `schema_version` matches.
2. Clear any stale `locks.device.held_by`.
3. Demote any `phase.status=in_progress` rows to `pending` (worker died mid-run; safer to redo than to assume success).
4. Smoke-check: re-run the PCC test for the most recently `ttnn.status=done` block. If it fails, escalate to user before resuming.
5. Schedule the next tick.

**Manual nudges:**
- `--redo <slug>:<block>:<phase>` — flips one phase cell back to `pending`, resets `attempts`.
- `--skip <slug>:<block>:<phase> --justify "<text>"` — sets `host_resident.allowed=true` with justification. Logged in `tick_log`.
- Direct edits to `.bringup_state.json` are supported. Next tick honors them if the schema check passes.

### 2. Orchestrator tick (short-lived dispatcher)

A single tick is the atomic unit of progress. Per tick:

1. Load `.bringup_state.json`. If the pipeline-complete check (§3 step 4) passes → exit, no rewake.
2. Pick the next action by the decision tree in §3.
3. Dispatch worker subagent(s) via the `Agent` tool — parallel for reference phase, serial for everything else.
4. Wait synchronously inside the tick for the worker(s) to return.
5. Parse worker results → mutate JSON → re-render `BRINGUP_LOG.md` from JSON.
6. `git add` the state file, log, and any worker-touched files; commit with `bringup(<slug>): <phase> <block> <status>`.
7. `ScheduleWakeup(delaySeconds=60, prompt="/bringup --resume models/demos/<slug>", reason=...)` for the next tick. Done.

Tick context stays small: load JSON, dispatch, write JSON. Workers hold the heavy code-reading context, and their context dies with them at the end of the tick.

### 3. Worker subagents

One per phase, each invoked via `Agent` with a worker prompt that immediately invokes the relevant existing skill:

| Worker | Existing skill invoked | Device lock | Parallelism |
|---|---|---|---|
| `architecture-worker` | `architecture` | no | 1 (once per bringup) |
| `reference-worker` | `reference` | no | up to `max_parallel_reference` (default 4) |
| `ttnn-worker` | `ttnn` | yes | 1 |
| `debug-worker` | `superpowers:systematic-debugging` then `debug` | yes | 1 |
| `optimization-worker` | `optimization` | yes | 1 |

Every worker returns a structured JSON result: `{block, phase, status: ok|fail|blocked, pcc, artifacts, notes, last_error}`. The tick — not the worker — is responsible for state mutation.

---

## Phase rules

### Tick decision tree (in order)

1. **Architecture pending?** → dispatch `architecture-worker`. One block.
2. **Any blocks with `reference.status ∈ {pending, failing}` and `attempts < max`?** → dispatch up to `max_parallel_reference` `reference-worker`s in a single tick. No device lock.
3. **Device-touching phases drain a single FIFO queue.** Candidates: blocks where `reference.status=done`, all `depends_on` have `ttnn.status=done`, the block itself has `ttnn` or `optimization` not yet `done`, **and the block's current phase status is not `blocked`**.
   - Priority within the queue: `ttnn.status=failing` blocks routed to **debug-worker** first; then `ttnn.status=pending`; then `optimization.status=pending`.
   - Exactly one device-worker per tick. `locks.device.held_by` set for the duration; next tick refuses device dispatch until cleared.
4. **Pipeline complete?** Every component `ttnn=done` (or `host_resident.allowed=true` with justification) AND `optimization=done|skipped`. → stop, no rewake.
5. **Deadlock check.** If steps 1–3 produce no eligible work AND step 4 says not-complete, the orchestrator is deadlocked — some `blocked` block is gating remaining work via `depends_on`. Write a `DEADLOCK` row to `tick_log`, render the blocking chain into `BRINGUP_LOG.md`, escalate to user via a final commit message `bringup(<slug>): deadlock — <block> blocks <N> downstream`, and exit without rewake. User unblocks via `--redo` or `--skip`.

### Bottom-up DAG

The architecture worker emits a dependency DAG, not just a flat block list:

```
Linear, RMSNorm, RoPE (leaves)
   ↓
Attention, MLP                       (composite)
   ↓
DecoderLayer                          (integrator)
   ↓
Talker, CodePredictor, SpeakerEncoder (sub-models)
   ↓
End-to-end model                      (full model)
```

A block's TTNN scheduling waits until *all* `depends_on` blocks are `ttnn.status=done`. Reference phase ignores the DAG (any block can be referenced standalone).

### "No shortcuts" guard

After every `ttnn-worker` returns success, the tick runs a guard before marking `ttnn.status=done`:

1. **Static lint** of the block's `tt/<block>.py` for forbidden patterns:
   - `.cpu()` / `.numpy()` on activations in forward paths (allowed in tests).
   - `torch.nn.functional.*` / `torch.matmul` inside the forward path.
   - `# TODO: move to ttnn` markers.
2. **Traced-op assertion**: capture a traced execution of the block's forward path; assert the traced op list contains the expected ttnn kernels for the block's `kind` (e.g. `attention` requires `ttnn.matmul`, `ttnn.softmax`, `ttnn.transformer.scaled_dot_product_attention`-family; `norm` requires `ttnn.rms_norm` / `ttnn.layer_norm`).
3. **Reference cross-check**: diff against the `reference_impl` path in `ARCHITECTURE.md`. If the reference has a host-resident sub-op (e.g. a too-large Conv1d kept on PyTorch), that exact op is allowed in the new block. Otherwise any host-resident code → reject, force re-dispatch via debug-worker.

Escape hatch: setting `host_resident.allowed=true` with `justification` and `reference_link` in `.bringup_state.json` (manually, via `--skip`, or by the architecture worker) lets a block pass the guard without satisfying it. Logged.

### Failure handling

- **Reference-worker fail** → re-dispatch reference-worker with the failure trace appended to its prompt. `attempts++`.
- **TTNN-worker fail** (PCC < 0.99, hang, crash) → next tick dispatches `debug-worker` with `{block, last_pcc, last_error, tt_smi_state}`. Debug worker may edit `tt/<block>.py` and re-run.
- **Debug-worker may touch `reference/`** only if it then re-verifies the modified reference vs the official HuggingFace model. If the reference change makes the reference diverge from HF (PCC < 0.99 against HF), the reference change is rejected and the bug is pushed back into TTNN.
- **Max attempts hit** (default 10 per phase per block) → status `blocked`, last error captured in BRINGUP_LOG.md, orchestrator skips it but continues independent branches of the DAG.
- **Device hang detected** (worker returns hang signal) → orchestrator runs `tt-smi -r`, then before resuming new work, re-dispatches a smoke check on the last `ttnn.status=done` block on the DAG. If the smoke check fails, escalate to user — something deeper than the current block is broken.

---

## State schema (`.bringup_state.json`)

Lives at `models/demos/<model_slug>/.bringup_state.json`. Source of truth. `BRINGUP_LOG.md` is rendered from it each tick.

```json
{
  "schema_version": 1,
  "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "model_slug": "qwen3_tts",
  "device": "n150",
  "arch_name": "wormhole_b0",
  "started_at": "2026-05-27T14:00:00Z",
  "updated_at": "2026-05-27T15:42:11Z",
  "components": [
    {
      "name": "RMSNorm",
      "kind": "norm",
      "reference_impl": "models/common/rmsnorm.py",
      "depends_on": [],
      "host_resident": {"allowed": false, "justification": null, "reference_link": null},
      "reference":    {"status": "done", "pcc": 0.999998, "attempts": 1,
                       "artifacts": ["reference/functional.py::rmsnorm_forward",
                                     "reference/golden/rmsnorm.pt"]},
      "ttnn":         {"status": "done", "pcc": 0.999985, "attempts": 1,
                       "artifacts": ["tt/rms_norm.py"]},
      "debug":        {"status": "n/a"},
      "optimization": {"status": "pending"}
    },
    {
      "name": "Attention",
      "kind": "attention",
      "reference_impl": "models/demos/llama3_70b_galaxy/tt/llama_attention.py",
      "depends_on": ["RMSNorm", "Linear", "RoPE"],
      "host_resident": {"allowed": false, "justification": null, "reference_link": null},
      "reference":    {"status": "done", "pcc": 0.9999, "attempts": 1, "artifacts": ["..."]},
      "ttnn":         {"status": "failing", "pcc": 0.81, "attempts": 3,
                       "last_error": "QK-norm mismatch (suspected fp32 cast)",
                       "artifacts": ["tt/attention.py"]},
      "debug":        {"status": "in_progress", "attempts": 1,
                       "hypothesis": "K matrix amplified by k_norm"},
      "optimization": {"status": "blocked", "blocked_on": "ttnn"}
    }
  ],
  "locks": {
    "device": {"held_by": null, "held_since": null}
  },
  "tick_log": [
    {"tick": 1, "ts": "...", "action": "architecture", "result": "ok"},
    {"tick": 2, "ts": "...", "action": "reference[RMSNorm,MLP,Attention]", "result": "ok"},
    {"tick": 17, "ts": "...", "action": "ttnn[Attention]", "result": "fail pcc=0.81"}
  ],
  "config": {
    "max_parallel_reference": 4,
    "max_attempts_per_phase": 10,
    "tick_interval_sec": 60
  }
}
```

**Invariants:**

- A phase's `status` ∈ `{pending, in_progress, done, failing, blocked, n/a, skipped}`.
- `attempts` increments only on `fail`. Hitting `max_attempts_per_phase` flips status to `blocked`.
- `tick_log` is bounded to last 100 entries; full history in git.
- `locks.device.held_by` is intent metadata only — serialization is enforced by the tick scheduler.

---

## Device registry

A small registry maps user-facing device names to mesh topology. Everything else (`l1_small_size`, `trace_region_size`, dispatch core count) is seeded from the reference TTNN model picked in `ARCHITECTURE.md`.

```python
DEVICE_REGISTRY = {
    "n150": {"arch": "wormhole_b0", "mesh_shape": (1, 1), "num_devices": 1},
    "n300": {"arch": "wormhole_b0", "mesh_shape": (1, 2), "num_devices": 2},
    "p150": {"arch": "blackhole",   "mesh_shape": (1, 1), "num_devices": 1},
    "t3k":  {"arch": "wormhole_b0", "mesh_shape": (1, 8), "num_devices": 8},
    "tg":   {"arch": "wormhole_b0", "mesh_shape": (8, 4), "num_devices": 32},
}
```

When the architecture worker selects `models/demos/llama3_70b_galaxy/` as the reference for an Attention block on `t3k`, `lib/device.py` greps that folder's device setup for `l1_small_size`, `trace_region_size`, mesh init parameters, and writes them into `.bringup_state.json` under a `device_defaults` block. User can override.

---

## File layout

```
skills/orchestrator/
├── SKILL.md                        # /bringup user-facing entry. Triggers /loop internally.
├── SPEC.md                         # this document
├── tick.md                         # the per-tick worker prompt the loop re-enters
├── workers/
│   ├── architecture-worker.md
│   ├── reference-worker.md
│   ├── ttnn-worker.md
│   ├── debug-worker.md
│   └── optimization-worker.md
└── lib/
    ├── state.py                    # read/write .bringup_state.json, render BRINGUP_LOG.md
    ├── dag.py                      # candidate selection from depends_on
    ├── guard.py                    # static lint + traced-op assertion
    └── device.py                   # tt-smi -r helper, smoke-check dispatcher, device-defaults extractor
```

Per-model artifacts (written by the orchestrator into the model folder):

```
models/demos/<model_slug>/
├── ARCHITECTURE.md                 # written by architecture-worker (existing convention)
├── BRINGUP_LOG.md                  # rendered from .bringup_state.json
├── .bringup_state.json             # source of truth
├── reference/
│   ├── functional.py
│   └── golden/
└── tt/
```

The existing per-phase skills (`architecture`, `reference`, `ttnn`, `debug`, `optimization`) are **not modified**. The orchestrator is a thin layer above them.

---

## Invocation summary

**Start:**
```
/bringup Qwen/Qwen3-TTS-12Hz-1.7B-Base --device n150
```
(The skill invokes `/loop` internally; user does not type `/loop`.)

**Resume after `/clear`, crash, or reboot:**
```
/bringup --resume models/demos/qwen3_tts
```

**Inspect:**
```
cat models/demos/qwen3_tts/.bringup_state.json | jq '.components[]|{name, ttnn:.ttnn.status, pcc:.ttnn.pcc}'
cat models/demos/qwen3_tts/BRINGUP_LOG.md
git log --oneline -- models/demos/qwen3_tts/.bringup_state.json
```

**Manual nudges:**
```
/bringup --redo qwen3_tts:Attention:ttnn
/bringup --skip qwen3_tts:SpeechTokDecoder:ttnn --justify "Conv too large for L1; matches reference hybrid pattern"
```

**Stop:** any user input during a tick interrupts the loop. State is on disk; resume with the command above.

---

## Open questions / future work

- Multi-host orchestration (TG cluster with multiple boxes) is out of scope for v1.
- Auto-promoting a block from `ttnn.status=done` to `optimization` is in scope, but the optimization phase's success criterion is the existing tracy/trace-replay throughput target inherited from the reference TTNN model. We may want explicit per-model perf targets in a later schema bump.
- Web UI for monitoring is not in scope; `jq` + `git log` + `BRINGUP_LOG.md` are the inspection surface.
