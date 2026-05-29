---
name: bringup
description: Automate the full TTNN model bring-up pipeline (architecture → reference → ttnn → debug → optimization → real_weights → generation → perf) for any HuggingFace model on any supported Tenstorrent device — from HF id to working end-to-end demos with characterized perf. State persists across sessions; resumable after crash.
---

<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Bringup Orchestrator (`/bringup`)

## Overview

`/bringup` is the single entry point that drives an entire TTNN model
bring-up from a HuggingFace model id to working end-to-end demos with
characterized perf on a Tenstorrent device. It bootstraps persistent
state under `models/demos/<slug>/.bringup_state.json`, then hands off
to `/loop` which repeatedly invokes the per-tick orchestrator
(`tick.md`).

Each tick dispatches one piece of work and commits. The orchestrator
walks 8 phases per model in order:

1. **architecture** (per-model) — discover components + use_cases
2. **reference** (per-component) — PyTorch reference vs HF
3. **ttnn** (per-component) — TTNN implementation, PCC > 0.99 vs reference
4. **debug** (per-component, only on failures) — diagnose + fix
5. **optimization** (per-component) — block-level perf tuning
6. **real_weights** (per-component, NEW) — real HF weights + full-config PCC
7. **generation** (per-use-case, NEW) — AR loop + demo + e2e validation gate
8. **perf** (per-use-case, NEW) — pipeline-level perf (paged_update_cache
   + reusable trace + targeted tracy)

The user's job is to invoke `/bringup` once and (occasionally) nudge
with `--redo` or `--skip` when a block or use_case is stuck.

See `SPEC.md` for the original bringup design and `SPEC_post_bringup.md`
for the integration / generation / perf extension.

## Argument forms

1. **First-run** — `/bringup <hf_model_id> [--device <name>]`
   - Example: `/bringup Qwen/Qwen3-TTS-12Hz-1.7B-Base --device n150`
   - `--device` defaults to `n150`.
2. **Resume** — `/bringup --resume <model_path>`
   - Example: `/bringup --resume models/demos/qwen_qwen3_tts_12hz_1.7b_base`
3. **Redo** — `/bringup --redo <slug>:<block>:<phase>`
   - Example: `/bringup --redo qwen3_tts:Attention:ttnn`
4. **Skip** — `/bringup --skip <slug>:<block>:<phase> --justify "<text>" --reference-link <path>`
   - Example: `/bringup --skip qwen3_tts:SpeechTokDecoder:ttnn --justify "Conv too large for L1" --reference-link models/demos/qwen3_tts/tt/decoder_ref.py`

`<phase>` is one of: `reference`, `ttnn`, `debug`, `optimization`.

## Branch 1: First-run (positional `<hf_id>`)

Parse the positional `<hf_id>` and optional `--device <name>` (default
`n150`). Then:

```bash
cd /local/ttuser/ssinghal/tt-metal
source python_env/bin/activate
export PYTHONPATH=$(pwd) && export TT_METAL_HOME=$(pwd)

python <<'PY'
import os, sys
from skills.orchestrator.lib.state import bootstrap, save_state, render_log
from skills.orchestrator.lib.device import device_info

model_id = "<HF_MODEL_ID>"        # substitute from args
device   = "<DEVICE>"             # substitute from args (default n150)
arch     = device_info(device)["arch"]
slug     = model_id.lower().replace("/", "_").replace("-", "_")
path     = f"models/demos/{slug}"

if os.path.exists(f"{path}/.bringup_state.json"):
    sys.exit(f"State already exists at {path}. Use /bringup --resume {path}")

os.makedirs(path, exist_ok=True)
state = bootstrap(model_id, device, arch)          # state.bootstrap
save_state(f"{path}/.bringup_state.json", state)
with open(f"{path}/BRINGUP_LOG.md", "w") as f:
    f.write(render_log(state))
print(path)
PY
```

Capture `<MODEL_PATH>` from the printed last line, then commit:

```bash
git add "<MODEL_PATH>/.bringup_state.json" "<MODEL_PATH>/BRINGUP_LOG.md"
git commit -m "chore(<SLUG>): start bringup"
```

Print: `Bootstrapped <SLUG>. Entering /loop /bringup --resume <MODEL_PATH>.`

Then invoke `Skill(loop)` with argument
`/bringup --resume <MODEL_PATH>` so the orchestrator self-drives via
`ScheduleWakeup` ticks. Do not block on the loop — return immediately.

## Branch 2: Resume (`--resume <model_path>`)

```bash
python <<'PY'
from skills.orchestrator.lib.state import load_state, resume_normalize, save_state
state = load_state("<MODEL_PATH>/.bringup_state.json")    # state.load_state
resume_normalize(state)                                    # state.resume_normalize
save_state("<MODEL_PATH>/.bringup_state.json", state)
print("pending_smoke_check:", state.get("pending_smoke_check"))
PY
```

If `pending_smoke_check` is set: dispatch a single ttnn-worker `Agent`
call for that block (re-run its existing PCC test), pop the field, save
state, then fall through to the normal tick. See `tick.md` §Step 2
"Override" for the spec shape.

Otherwise: read `skills/orchestrator/tick.md` and invoke it via
`Agent(subagent_type="general-purpose", prompt=tick_md_contents + "\n\n## Args:\n" + model_path)`.
The tick subagent handles phase selection, worker dispatch, state
mutation, and commit. It does NOT schedule the next tick — subagents
have no `ScheduleWakeup`, and a subagent `CronCreate` would leak a
duplicate timer. Instead the tick reports its next-tick intent on its
last line as `NEXT_TICK: {"phase": ..., "block": ..., "worker": ...}`.

When the tick returns, YOU (the parent) own the wakeup:
- Parse the `NEXT_TICK:` line.
- If `phase == "done"` or `phase == "deadlock"`: do not reschedule;
  print the completion/deadlock summary and exit.
- Otherwise: `ScheduleWakeup(delaySeconds=<tick_interval_sec>,
  prompt="/bringup --resume <model_path>", reason="next: <phase>/<block>")`.
  (If invoked under `/loop`, follow the /loop skill's dynamic-mode
  wakeup contract instead — same effect, the loop owns the cadence.)

## Branch 3: Redo (`--redo <slug>:<block>:<phase>`)

```bash
python <<'PY'
from skills.orchestrator.lib.state import load_state, redo, save_state, render_log
path = "models/demos/<SLUG>/.bringup_state.json"
state = load_state(path)
redo(state, "<BLOCK>", "<PHASE>")                          # state.redo
save_state(path, state)
with open("models/demos/<SLUG>/BRINGUP_LOG.md", "w") as f:
    f.write(render_log(state))
PY

git add "models/demos/<SLUG>/.bringup_state.json" "models/demos/<SLUG>/BRINGUP_LOG.md"
git commit -m "chore(<SLUG>): redo <BLOCK>:<PHASE>"
```

EXIT. Do NOT enter `/loop`. The user must invoke
`/bringup --resume models/demos/<SLUG>` to continue.

## Branch 4: Skip (`--skip <slug>:<block>:<phase> --justify "<text>" --reference-link <path>`)

```bash
python <<'PY'
from skills.orchestrator.lib.state import load_state, skip, save_state, render_log
path = "models/demos/<SLUG>/.bringup_state.json"
state = load_state(path)
skip(state, "<BLOCK>", "<PHASE>", "<JUSTIFY>", "<REFERENCE_LINK>")  # state.skip
save_state(path, state)
with open("models/demos/<SLUG>/BRINGUP_LOG.md", "w") as f:
    f.write(render_log(state))
PY

git add "models/demos/<SLUG>/.bringup_state.json" "models/demos/<SLUG>/BRINGUP_LOG.md"
git commit -m "chore(<SLUG>): skip <BLOCK>:<PHASE> (host-resident, justified)"
```

`state.skip` requires non-empty `--justify` AND `--reference-link` —
the host-resident escape hatch must be auditable. EXIT after commit;
do NOT enter `/loop`.

## Inspection (read-only, no orchestrator state change)

The user can inspect any in-progress bringup without invoking `/bringup`:

- `cat models/demos/<slug>/.bringup_state.json | jq .`
- `cat models/demos/<slug>/BRINGUP_LOG.md`
- `git log --oneline -- models/demos/<slug>/.bringup_state.json`

## Failure modes

- **State file already exists on first-run.** Refuse with the message
  shown above; tell the user to use `--resume`.
- **Unknown device name.** `device_info` raises `UnknownDeviceError`.
  Halt and print the supported names: `n150`, `n300`, `p150`, `t3k`, `tg`.
- **SchemaError on resume.** `load_state` raises `SchemaError`. Do not
  enter `/loop`. Print the error and let the user inspect the state file.
- **Empty `--justify` or `--reference-link` on `--skip`.** `state.skip`
  raises `ValueError`. Refuse the nudge.
- **Device hang during a tick.** Handled inside `tick.md` via
  `tt_smi_reset()` and `pending_smoke_check`. Not the responsibility of
  this entry script.

## Glossary

- **slug** — `model_id.lower().replace("/", "_").replace("-", "_")`. The
  on-disk directory name under `models/demos/`.
- **phase** — one of `reference`, `ttnn`, `debug`, `optimization`,
  `real_weights` (per-component), or `generation`, `perf` (per-use-case).
  `architecture` is a one-shot pre-step that populates both
  `state.components` AND `state.use_cases`.
- **component** — a single TTNN block (e.g. `Attention`, `LayerNorm`).
  Lives under `state["components"]`.
- **use_case** — a distinct inference path the model exposes (e.g.
  `t2tt`, `asr`, `text_generation`). Discovered by the architecture
  worker from the HF model's class hierarchy. Lives under
  `state["use_cases"]`. Each has its own `generation` and `perf` phase.
- **tick** — one atomic orchestrator step: load → decide → dispatch →
  parse → save → commit → wakeup. Driven by `tick.md`.
- **host-resident** — a block that legitimately cannot run on-device
  (e.g. a Conv too large for L1). Allowed only via `--skip` with a
  justification and a reference-link.
- **hybrid boundary** — parts of a use_case pipeline that legitimately
  stay on HF host (e.g. tokenizer-bound char prep). Documented per
  use_case in its `hybrid_notes` field.
