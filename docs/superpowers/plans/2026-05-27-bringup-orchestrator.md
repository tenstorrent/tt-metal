# Bringup Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tick-based orchestrator at `skills/orchestrator/` that autonomously drives the full TTNN model bring-up pipeline (architecture → reference → ttnn → debug → optimization) for any HuggingFace model on any supported Tenstorrent device, with state on disk and resumability across crashes.

**Architecture:** Three layers. (1) A `/bringup` skill (SKILL.md) is the user entry; on first run it bootstraps state and internally invokes `/loop` so it self-paces via `ScheduleWakeup`. (2) Each loop tick re-enters a short-lived orchestrator that reads `.bringup_state.json`, picks one action from the §3 decision tree, dispatches worker subagent(s), writes state, commits, and schedules the next tick. (3) Worker subagents are thin markdown prompts that invoke the existing per-phase skills (`architecture`, `reference`, `ttnn`, `debug`, `optimization`). All decision logic lives in `skills/orchestrator/lib/` (pure Python, fully unit-tested).

**Tech Stack:** Python 3.10, pytest, Click (already a repo dependency) for arg parsing in helpers, no torch imports in the orchestrator itself (workers handle that). The existing five `skills/*/SKILL.md` files are not modified.

**Spec reference:** `skills/orchestrator/SPEC.md`

---

## File Structure

**Files this plan creates (orchestrator code):**

- `skills/orchestrator/__init__.py` — marks the directory a Python package so tests can import.
- `skills/orchestrator/lib/__init__.py` — same for the lib subpackage.
- `skills/orchestrator/lib/state.py` — JSON load/save, schema validation, `BRINGUP_LOG.md` renderer, `bootstrap()`, `redo()`, `skip()`, `demote_in_progress()`.
- `skills/orchestrator/lib/dag.py` — `eligible_blocks(state)` returns the next action per the §3 decision tree; `is_complete(state)`, `is_deadlocked(state)`.
- `skills/orchestrator/lib/device.py` — `DEVICE_REGISTRY`, `extract_device_defaults(reference_model_path)`, `reset_device()` (wraps `tt-smi -r`), `last_done_block(state)`.
- `skills/orchestrator/lib/guard.py` — `lint_block(tt_block_path)` (static check), `assert_traced_ops(traced_op_list, block_kind)`, `verify_no_shortcuts(component, state)`.
- `skills/orchestrator/lib/cli.py` — Click CLI entry: `bringup start|resume|redo|skip` for use from the SKILL.md `Bash` calls.

**Tests (TDD):**

- `skills/orchestrator/lib/tests/__init__.py`
- `skills/orchestrator/lib/tests/conftest.py` — shared fixtures (sample state dict, tmp model folder).
- `skills/orchestrator/lib/tests/test_state.py`
- `skills/orchestrator/lib/tests/test_dag.py`
- `skills/orchestrator/lib/tests/test_device.py`
- `skills/orchestrator/lib/tests/test_guard.py`
- `skills/orchestrator/lib/tests/test_cli.py`

**Worker prompts (markdown — no unit tests; linted manually by smoke):**

- `skills/orchestrator/workers/architecture-worker.md`
- `skills/orchestrator/workers/reference-worker.md`
- `skills/orchestrator/workers/ttnn-worker.md`
- `skills/orchestrator/workers/debug-worker.md`
- `skills/orchestrator/workers/optimization-worker.md`

**Top-level orchestrator markdown:**

- `skills/orchestrator/SKILL.md` — `/bringup` entry, arg parsing via `lib/cli.py`, internal `/loop` invocation.
- `skills/orchestrator/tick.md` — per-tick re-entry prompt that the loop fires.

**Files this plan does NOT modify:** the five existing `skills/*/SKILL.md` files, anything under `models/`, anything under `tt_metal/`. The orchestrator is purely additive.

**Per-model artifacts (written by the orchestrator at runtime, not by this plan):** `.bringup_state.json`, rendered `BRINGUP_LOG.md`. The plan's tests use temp dirs.

---

## Worker Result Contract (shared spec)

Every worker subagent returns a single JSON document on stdout (parsed by the tick). This contract is referenced by the worker-prompt tasks and by `lib/state.py`:

```json
{
  "block": "Attention",
  "phase": "ttnn",
  "status": "ok",
  "pcc": 0.998,
  "artifacts": ["models/demos/qwen3_tts/tt/attention.py"],
  "notes": "fused QKV; QK-norm applied pre-RoPE",
  "last_error": null,
  "hang_detected": false
}
```

`status` ∈ `{"ok", "fail", "blocked"}`. `pcc` is `null` for phases without a PCC notion (architecture, optimization). `hang_detected=true` instructs the tick to run `tt-smi -r` before the next dispatch.

---

### Task 1: Scaffolding

**Files:**
- Create: `skills/orchestrator/__init__.py`
- Create: `skills/orchestrator/lib/__init__.py`
- Create: `skills/orchestrator/lib/tests/__init__.py`
- Create: `skills/orchestrator/lib/tests/conftest.py`

- [ ] **Step 1: Create the empty `__init__.py` files**

```bash
touch skills/orchestrator/__init__.py
touch skills/orchestrator/lib/__init__.py
touch skills/orchestrator/lib/tests/__init__.py
```

- [ ] **Step 2: Write `conftest.py` with shared fixtures**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for orchestrator lib tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest


SAMPLE_STATE: dict[str, Any] = {
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
            "host_resident": {"allowed": False, "justification": None, "reference_link": None},
            "reference": {"status": "done", "pcc": 0.999998, "attempts": 1, "artifacts": []},
            "ttnn": {"status": "done", "pcc": 0.999985, "attempts": 1, "artifacts": []},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
        {
            "name": "Attention",
            "kind": "attention",
            "reference_impl": "models/demos/llama3_70b_galaxy/tt/llama_attention.py",
            "depends_on": ["RMSNorm"],
            "host_resident": {"allowed": False, "justification": None, "reference_link": None},
            "reference": {"status": "done", "pcc": 0.9999, "attempts": 1, "artifacts": []},
            "ttnn": {"status": "pending", "attempts": 0, "artifacts": []},
            "debug": {"status": "n/a"},
            "optimization": {"status": "pending"},
        },
    ],
    "locks": {"device": {"held_by": None, "held_since": None}},
    "tick_log": [],
    "config": {
        "max_parallel_reference": 4,
        "max_attempts_per_phase": 10,
        "tick_interval_sec": 60,
    },
}


@pytest.fixture
def sample_state() -> dict[str, Any]:
    return copy.deepcopy(SAMPLE_STATE)


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    """Returns a tmp model folder layout: tmp_path/models/demos/qwen3_tts/."""
    p = tmp_path / "models" / "demos" / "qwen3_tts"
    p.mkdir(parents=True)
    return p


@pytest.fixture
def tmp_state_file(tmp_model_dir: Path, sample_state: dict[str, Any]) -> Path:
    f = tmp_model_dir / ".bringup_state.json"
    f.write_text(json.dumps(sample_state, indent=2))
    return f
```

- [ ] **Step 3: Verify pytest discovers the tests dir**

Run: `export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate && pytest skills/orchestrator/lib/tests/ --collect-only -q`

Expected: `no tests collected` (no test files yet) and no import errors.

- [ ] **Step 4: Commit**

```bash
git add skills/orchestrator/__init__.py skills/orchestrator/lib/__init__.py skills/orchestrator/lib/tests/__init__.py skills/orchestrator/lib/tests/conftest.py
git commit -m "skills/orchestrator: scaffold package layout and shared fixtures"
```

---

### Task 2: `lib/state.py` — load / save with schema validation (RED → GREEN)

**Files:**
- Create: `skills/orchestrator/lib/tests/test_state.py`
- Create: `skills/orchestrator/lib/state.py`

- [ ] **Step 1: Write the failing tests**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills/orchestrator/lib/state.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skills.orchestrator.lib import state as state_mod


SCHEMA_VERSION = 1


def test_load_state_round_trip(tmp_state_file: Path, sample_state: dict):
    s = state_mod.load_state(tmp_state_file)
    assert s == sample_state


def test_save_state_writes_json(tmp_model_dir: Path, sample_state: dict):
    out = tmp_model_dir / "out.json"
    state_mod.save_state(out, sample_state)
    on_disk = json.loads(out.read_text())
    assert on_disk == sample_state


def test_load_state_rejects_wrong_schema_version(tmp_state_file: Path):
    bad = json.loads(tmp_state_file.read_text())
    bad["schema_version"] = 999
    tmp_state_file.write_text(json.dumps(bad))
    with pytest.raises(state_mod.SchemaError, match="schema_version"):
        state_mod.load_state(tmp_state_file)


def test_load_state_rejects_missing_required_field(tmp_state_file: Path):
    bad = json.loads(tmp_state_file.read_text())
    del bad["components"]
    tmp_state_file.write_text(json.dumps(bad))
    with pytest.raises(state_mod.SchemaError, match="components"):
        state_mod.load_state(tmp_state_file)


def test_load_state_rejects_unknown_status(tmp_state_file: Path):
    bad = json.loads(tmp_state_file.read_text())
    bad["components"][0]["ttnn"]["status"] = "not-a-real-status"
    tmp_state_file.write_text(json.dumps(bad))
    with pytest.raises(state_mod.SchemaError, match="status"):
        state_mod.load_state(tmp_state_file)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skills/orchestrator/lib/tests/test_state.py -v`

Expected: `ModuleNotFoundError: No module named 'skills.orchestrator.lib.state'` (RED).

- [ ] **Step 3: Implement `lib/state.py` — load/save + schema check**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bringup orchestrator state — JSON load/save and schema validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

VALID_STATUSES = {
    "pending",
    "in_progress",
    "done",
    "failing",
    "blocked",
    "n/a",
    "skipped",
}
VALID_PHASES = ("reference", "ttnn", "debug", "optimization")
REQUIRED_TOP_LEVEL = (
    "schema_version",
    "model_id",
    "model_slug",
    "device",
    "components",
    "locks",
    "tick_log",
    "config",
)


class SchemaError(ValueError):
    """Raised when .bringup_state.json fails schema validation."""


def _validate(state: dict[str, Any]) -> None:
    for k in REQUIRED_TOP_LEVEL:
        if k not in state:
            raise SchemaError(f"missing required field: {k}")
    if state["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema_version: got {state['schema_version']!r}, expected {SCHEMA_VERSION}"
        )
    for c in state["components"]:
        if "name" not in c:
            raise SchemaError("component missing name")
        for phase in VALID_PHASES:
            if phase not in c:
                raise SchemaError(f"component {c['name']} missing phase {phase}")
            st = c[phase].get("status")
            if st not in VALID_STATUSES:
                raise SchemaError(
                    f"component {c['name']} phase {phase} has invalid status: {st!r}"
                )


def load_state(path: Path) -> dict[str, Any]:
    state = json.loads(Path(path).read_text())
    _validate(state)
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    _validate(state)
    Path(path).write_text(json.dumps(state, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest skills/orchestrator/lib/tests/test_state.py -v`

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add skills/orchestrator/lib/state.py skills/orchestrator/lib/tests/test_state.py
git commit -m "skills/orchestrator: state.py load/save with schema validation"
```

---

### Task 3: `lib/state.py` — `BRINGUP_LOG.md` renderer (RED → GREEN)

**Files:**
- Modify: `skills/orchestrator/lib/tests/test_state.py` (append)
- Modify: `skills/orchestrator/lib/state.py` (append `render_bringup_log`)

- [ ] **Step 1: Append failing test**

```python
def test_render_bringup_log_contains_per_component_row(sample_state: dict):
    md = state_mod.render_bringup_log(sample_state)
    assert "Qwen3-TTS-12Hz-1.7B-Base" in md
    assert "| RMSNorm |" in md
    assert "| Attention |" in md
    # PCC visible for the ttnn-done row
    assert "0.999985" in md
    # Pending phase rendered as empty PCC, not 'None'
    assert "None" not in md


def test_render_bringup_log_marks_blocked_with_reason(sample_state: dict):
    sample_state["components"][1]["ttnn"]["status"] = "blocked"
    sample_state["components"][1]["ttnn"]["last_error"] = "QK-norm mismatch"
    md = state_mod.render_bringup_log(sample_state)
    assert "BLOCKED" in md
    assert "QK-norm mismatch" in md
```

- [ ] **Step 2: Run — verify RED**

Run: `pytest skills/orchestrator/lib/tests/test_state.py::test_render_bringup_log_contains_per_component_row -v`
Expected: `AttributeError: module ... has no attribute 'render_bringup_log'`

- [ ] **Step 3: Implement `render_bringup_log` in `state.py`**

Append to `state.py`:

```python
def render_bringup_log(state: dict[str, Any]) -> str:
    """Render BRINGUP_LOG.md from state. Pure function — no I/O."""
    lines: list[str] = []
    model_short = state["model_id"].split("/")[-1]
    lines.append(f"# BRINGUP LOG: {model_short}\n")
    lines.append(f"**Model:** [{state['model_id']}](https://huggingface.co/{state['model_id']})  ")
    lines.append(f"**Device:** {state['device']} ({state['arch_name']})  ")
    lines.append(f"**Updated:** {state['updated_at']}\n")
    lines.append("## Per-Block Status\n")
    lines.append("| Block | Reference | TTNN | PCC | Optimization | Notes |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for c in state["components"]:
        ref = c["reference"]["status"]
        ttnn = c["ttnn"]["status"]
        opt = c["optimization"]["status"]
        pcc = c["ttnn"].get("pcc")
        pcc_str = f"{pcc:.6f}" if isinstance(pcc, (int, float)) else ""
        notes = ""
        if ttnn == "blocked":
            notes = f"BLOCKED: {c['ttnn'].get('last_error', '')}"
        elif c["host_resident"]["allowed"]:
            notes = f"HOST-RESIDENT: {c['host_resident']['justification']}"
        lines.append(
            f"| {c['name']} | {ref} | {ttnn} | {pcc_str} | {opt} | {notes} |"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests — verify GREEN**

Run: `pytest skills/orchestrator/lib/tests/test_state.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add skills/orchestrator/lib/state.py skills/orchestrator/lib/tests/test_state.py
git commit -m "skills/orchestrator: render BRINGUP_LOG.md from state"
```

---

### Task 4: `lib/state.py` — `bootstrap()` + manual nudges (RED → GREEN)

**Files:**
- Modify: `skills/orchestrator/lib/tests/test_state.py` (append)
- Modify: `skills/orchestrator/lib/state.py` (append `bootstrap`, `redo`, `skip`, `demote_in_progress`)

- [ ] **Step 1: Append failing tests**

```python
def test_bootstrap_writes_skeleton_state(tmp_model_dir: Path):
    target = tmp_model_dir / ".bringup_state.json"
    state_mod.bootstrap(
        target,
        model_id="Foo/Bar",
        model_slug="bar",
        device="n150",
        arch_name="wormhole_b0",
    )
    s = state_mod.load_state(target)
    assert s["model_id"] == "Foo/Bar"
    assert s["components"] == []
    assert s["config"]["max_attempts_per_phase"] == 10
    assert s["config"]["max_parallel_reference"] == 4


def test_bootstrap_refuses_if_state_already_exists(tmp_state_file: Path):
    with pytest.raises(FileExistsError):
        state_mod.bootstrap(
            tmp_state_file,
            model_id="X/Y",
            model_slug="y",
            device="n150",
            arch_name="wormhole_b0",
        )


def test_redo_resets_one_phase(sample_state: dict):
    state_mod.redo(sample_state, block="Attention", phase="ttnn")
    a = next(c for c in sample_state["components"] if c["name"] == "Attention")
    assert a["ttnn"]["status"] == "pending"
    assert a["ttnn"]["attempts"] == 0


def test_skip_sets_host_resident_with_justification(sample_state: dict):
    state_mod.skip(
        sample_state,
        block="Attention",
        phase="ttnn",
        justification="conv too large for L1",
        reference_link="models/demos/qwen3_tts/tt/speech_decoder.py",
    )
    a = next(c for c in sample_state["components"] if c["name"] == "Attention")
    assert a["host_resident"]["allowed"] is True
    assert "conv too large" in a["host_resident"]["justification"]


def test_demote_in_progress(sample_state: dict):
    sample_state["components"][1]["ttnn"]["status"] = "in_progress"
    state_mod.demote_in_progress(sample_state)
    assert sample_state["components"][1]["ttnn"]["status"] == "pending"
```

- [ ] **Step 2: Run — verify RED**

Run: `pytest skills/orchestrator/lib/tests/test_state.py -v`
Expected: 5 new failures (`AttributeError`).

- [ ] **Step 3: Implement the four functions in `state.py`**

Append to `state.py`:

```python
import datetime as _dt


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _skeleton(model_id: str, model_slug: str, device: str, arch_name: str) -> dict[str, Any]:
    now = _utcnow_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "model_slug": model_slug,
        "device": device,
        "arch_name": arch_name,
        "started_at": now,
        "updated_at": now,
        "components": [],
        "locks": {"device": {"held_by": None, "held_since": None}},
        "tick_log": [],
        "config": {
            "max_parallel_reference": 4,
            "max_attempts_per_phase": 10,
            "tick_interval_sec": 60,
        },
    }


def bootstrap(
    path: Path,
    *,
    model_id: str,
    model_slug: str,
    device: str,
    arch_name: str,
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists — use --resume")
    save_state(path, _skeleton(model_id, model_slug, device, arch_name))


def _component(state: dict[str, Any], block: str) -> dict[str, Any]:
    for c in state["components"]:
        if c["name"] == block:
            return c
    raise KeyError(f"no such component: {block}")


def redo(state: dict[str, Any], *, block: str, phase: str) -> None:
    c = _component(state, block)
    if phase not in VALID_PHASES:
        raise SchemaError(f"unknown phase: {phase}")
    c[phase]["status"] = "pending"
    c[phase]["attempts"] = 0
    state["updated_at"] = _utcnow_iso()


def skip(
    state: dict[str, Any],
    *,
    block: str,
    phase: str,
    justification: str,
    reference_link: str,
) -> None:
    c = _component(state, block)
    if phase not in VALID_PHASES:
        raise SchemaError(f"unknown phase: {phase}")
    c["host_resident"] = {
        "allowed": True,
        "justification": justification,
        "reference_link": reference_link,
    }
    c[phase]["status"] = "skipped"
    state["updated_at"] = _utcnow_iso()


def demote_in_progress(state: dict[str, Any]) -> None:
    for c in state["components"]:
        for phase in VALID_PHASES:
            if c[phase].get("status") == "in_progress":
                c[phase]["status"] = "pending"
    state["updated_at"] = _utcnow_iso()
```

- [ ] **Step 4: Run tests — verify GREEN**

Run: `pytest skills/orchestrator/lib/tests/test_state.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add skills/orchestrator/lib/state.py skills/orchestrator/lib/tests/test_state.py
git commit -m "skills/orchestrator: state bootstrap, redo, skip, demote_in_progress"
```

---

### Task 5: `lib/dag.py` — eligibility + completion + deadlock (RED → GREEN)

**Files:**
- Create: `skills/orchestrator/lib/tests/test_dag.py`
- Create: `skills/orchestrator/lib/dag.py`

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills/orchestrator/lib/dag.py."""
from __future__ import annotations

import pytest

from skills.orchestrator.lib import dag as dag_mod


def test_eligible_when_architecture_pending(sample_state: dict):
    sample_state["components"] = []  # arch not yet run
    action = dag_mod.next_action(sample_state)
    assert action.kind == "architecture"


def test_eligible_reference_fans_out(sample_state: dict):
    # Mark everything ttnn-pending and reference-pending to force reference phase
    for c in sample_state["components"]:
        c["reference"]["status"] = "pending"
        c["reference"]["attempts"] = 0
        c["ttnn"]["status"] = "pending"
    action = dag_mod.next_action(sample_state)
    assert action.kind == "reference"
    # Both eligible — fan-out picks up to max_parallel_reference (default 4)
    assert set(action.blocks) == {"RMSNorm", "Attention"}


def test_ttnn_serialized_one_block(sample_state: dict):
    # Attention is ttnn-pending; RMSNorm ttnn-done; depends_on satisfied
    action = dag_mod.next_action(sample_state)
    assert action.kind == "ttnn"
    assert action.blocks == ["Attention"]


def test_ttnn_skipped_when_dep_not_done(sample_state: dict):
    # Make RMSNorm ttnn-pending so Attention's dep is unmet
    sample_state["components"][0]["ttnn"]["status"] = "pending"
    sample_state["components"][1]["ttnn"]["status"] = "pending"
    action = dag_mod.next_action(sample_state)
    assert action.kind == "ttnn"
    # Only RMSNorm is dispatchable (no unmet deps)
    assert action.blocks == ["RMSNorm"]


def test_debug_routed_when_ttnn_failing(sample_state: dict):
    sample_state["components"][1]["ttnn"]["status"] = "failing"
    sample_state["components"][1]["ttnn"]["pcc"] = 0.81
    action = dag_mod.next_action(sample_state)
    assert action.kind == "debug"
    assert action.blocks == ["Attention"]


def test_blocked_excluded(sample_state: dict):
    sample_state["components"][1]["ttnn"]["status"] = "blocked"
    # Nothing else pending → no ttnn action
    sample_state["components"][0]["optimization"]["status"] = "done"
    action = dag_mod.next_action(sample_state)
    # Attention is blocked → only optimization remains, but RMSNorm.optimization already done
    # So pipeline should be deadlocked: Attention blocked, downstream gated
    assert action.kind in ("deadlock", "done")


def test_is_complete(sample_state: dict):
    for c in sample_state["components"]:
        for phase in ("reference", "ttnn", "optimization"):
            c[phase]["status"] = "done"
    assert dag_mod.is_complete(sample_state) is True


def test_is_complete_accepts_skipped(sample_state: dict):
    for c in sample_state["components"]:
        for phase in ("reference", "ttnn", "optimization"):
            c[phase]["status"] = "done"
    sample_state["components"][1]["host_resident"]["allowed"] = True
    sample_state["components"][1]["ttnn"]["status"] = "skipped"
    assert dag_mod.is_complete(sample_state) is True


def test_deadlock_detection_via_blocked_dep(sample_state: dict):
    # RMSNorm blocked → Attention can never satisfy depends_on
    sample_state["components"][0]["ttnn"]["status"] = "blocked"
    sample_state["components"][1]["ttnn"]["status"] = "pending"
    assert dag_mod.is_deadlocked(sample_state) is True
```

- [ ] **Step 2: Run — verify RED**

Run: `pytest skills/orchestrator/lib/tests/test_dag.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `lib/dag.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bringup orchestrator decision tree — pure functions over state."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ActionKind = Literal["architecture", "reference", "ttnn", "debug", "optimization", "deadlock", "done"]


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    blocks: list[str] = field(default_factory=list)


def _dep_satisfied(component: dict[str, Any], state: dict[str, Any]) -> bool:
    if not component["depends_on"]:
        return True
    done_set = {c["name"] for c in state["components"] if c["ttnn"]["status"] == "done"
                or (c["host_resident"]["allowed"] and c["ttnn"]["status"] == "skipped")}
    return all(d in done_set for d in component["depends_on"])


def _phase_active(phase_obj: dict[str, Any]) -> bool:
    return phase_obj["status"] not in ("done", "blocked", "skipped", "n/a")


def is_complete(state: dict[str, Any]) -> bool:
    for c in state["components"]:
        ttnn_ok = c["ttnn"]["status"] == "done" or (
            c["host_resident"]["allowed"] and c["ttnn"]["status"] == "skipped"
        )
        opt_ok = c["optimization"]["status"] in ("done", "skipped", "n/a")
        if not (ttnn_ok and opt_ok):
            return False
    return True


def is_deadlocked(state: dict[str, Any]) -> bool:
    """No eligible action, not complete, and at least one block is blocked
    such that downstream depends_on cannot be satisfied."""
    if is_complete(state):
        return False
    action = next_action(state, _allow_deadlock_recursion=True)
    return action.kind == "deadlock"


def next_action(state: dict[str, Any], *, _allow_deadlock_recursion: bool = False) -> Action:
    # 1. Architecture phase first
    if not state["components"]:
        return Action(kind="architecture")

    cfg = state["config"]
    max_attempts = cfg["max_attempts_per_phase"]

    # 2. Reference fan-out
    ref_candidates = [
        c["name"]
        for c in state["components"]
        if c["reference"]["status"] in ("pending", "failing")
        and c["reference"].get("attempts", 0) < max_attempts
    ]
    if ref_candidates:
        n = cfg["max_parallel_reference"]
        return Action(kind="reference", blocks=ref_candidates[:n])

    # 3. Device-touching phases — drain a single FIFO queue, debug first
    failing_ttnn = [
        c["name"]
        for c in state["components"]
        if c["ttnn"]["status"] == "failing"
        and c["ttnn"].get("attempts", 0) < max_attempts
        and _dep_satisfied(c, state)
    ]
    if failing_ttnn:
        return Action(kind="debug", blocks=failing_ttnn[:1])

    pending_ttnn = [
        c["name"]
        for c in state["components"]
        if c["ttnn"]["status"] == "pending"
        and c["reference"]["status"] == "done"
        and c["ttnn"].get("attempts", 0) < max_attempts
        and _dep_satisfied(c, state)
    ]
    if pending_ttnn:
        return Action(kind="ttnn", blocks=pending_ttnn[:1])

    pending_opt = [
        c["name"]
        for c in state["components"]
        if c["optimization"]["status"] == "pending"
        and (c["ttnn"]["status"] == "done"
             or (c["host_resident"]["allowed"] and c["ttnn"]["status"] == "skipped"))
    ]
    if pending_opt:
        return Action(kind="optimization", blocks=pending_opt[:1])

    # 4. Pipeline complete?
    if is_complete(state):
        return Action(kind="done")

    # 5. Deadlock
    return Action(kind="deadlock")
```

- [ ] **Step 4: Run tests — verify GREEN**

Run: `pytest skills/orchestrator/lib/tests/test_dag.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add skills/orchestrator/lib/dag.py skills/orchestrator/lib/tests/test_dag.py
git commit -m "skills/orchestrator: dag.py — next_action, is_complete, is_deadlocked"
```

---

### Task 6: `lib/device.py` — registry + defaults extraction (RED → GREEN)

**Files:**
- Create: `skills/orchestrator/lib/tests/test_device.py`
- Create: `skills/orchestrator/lib/device.py`

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills/orchestrator/lib/device.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from skills.orchestrator.lib import device as dev


def test_registry_has_known_devices():
    assert "n150" in dev.DEVICE_REGISTRY
    assert dev.DEVICE_REGISTRY["n150"]["arch"] == "wormhole_b0"
    assert dev.DEVICE_REGISTRY["t3k"]["num_devices"] == 8


def test_extract_device_defaults_finds_l1_small(tmp_path: Path):
    ref = tmp_path / "tt"
    ref.mkdir()
    (ref / "model_config.py").write_text(
        "L1_SMALL_SIZE = 16384\nTRACE_REGION_SIZE = 50_000_000\n"
    )
    defaults = dev.extract_device_defaults(ref.parent)
    assert defaults["l1_small_size"] == 16384
    assert defaults["trace_region_size"] == 50_000_000


def test_extract_device_defaults_returns_empty_when_not_found(tmp_path: Path):
    (tmp_path / "tt").mkdir()
    defaults = dev.extract_device_defaults(tmp_path)
    assert defaults == {}


def test_last_done_block_returns_most_recent(sample_state: dict):
    # RMSNorm is the only ttnn-done block in the sample
    assert dev.last_done_block(sample_state) == "RMSNorm"


def test_last_done_block_returns_none_when_no_done(sample_state: dict):
    sample_state["components"][0]["ttnn"]["status"] = "pending"
    assert dev.last_done_block(sample_state) is None
```

- [ ] **Step 2: Run — verify RED**

Run: `pytest skills/orchestrator/lib/tests/test_device.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `lib/device.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device helpers: registry, defaults extraction, tt-smi reset, smoke selection."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


DEVICE_REGISTRY: dict[str, dict[str, Any]] = {
    "n150": {"arch": "wormhole_b0", "mesh_shape": (1, 1), "num_devices": 1},
    "n300": {"arch": "wormhole_b0", "mesh_shape": (1, 2), "num_devices": 2},
    "p150": {"arch": "blackhole",   "mesh_shape": (1, 1), "num_devices": 1},
    "t3k":  {"arch": "wormhole_b0", "mesh_shape": (1, 8), "num_devices": 8},
    "tg":   {"arch": "wormhole_b0", "mesh_shape": (8, 4), "num_devices": 32},
}

_INT_PATTERNS = {
    "l1_small_size": re.compile(r"L1_SMALL_SIZE\s*=\s*([0-9_]+)"),
    "trace_region_size": re.compile(r"TRACE_REGION_SIZE\s*=\s*([0-9_]+)"),
    "num_command_queues": re.compile(r"NUM_COMMAND_QUEUES\s*=\s*([0-9_]+)"),
}


def extract_device_defaults(reference_model_root: Path) -> dict[str, int]:
    """Grep the reference model's tt/ folder for device-config constants."""
    tt_dir = Path(reference_model_root) / "tt"
    if not tt_dir.is_dir():
        return {}
    out: dict[str, int] = {}
    for py in tt_dir.rglob("*.py"):
        text = py.read_text(errors="ignore")
        for key, pat in _INT_PATTERNS.items():
            if key in out:
                continue
            m = pat.search(text)
            if m:
                out[key] = int(m.group(1).replace("_", ""))
    return out


def reset_device() -> subprocess.CompletedProcess[str]:
    """Run tt-smi -r. Returns the CompletedProcess; caller decides on failure."""
    return subprocess.run(
        ["tt-smi", "-r"], check=False, capture_output=True, text=True
    )


def last_done_block(state: dict[str, Any]) -> str | None:
    """Return the name of the most recent (by component order) ttnn-done block."""
    done = [c["name"] for c in state["components"] if c["ttnn"]["status"] == "done"]
    return done[-1] if done else None
```

- [ ] **Step 4: Run tests — verify GREEN**

Run: `pytest skills/orchestrator/lib/tests/test_device.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add skills/orchestrator/lib/device.py skills/orchestrator/lib/tests/test_device.py
git commit -m "skills/orchestrator: device registry, defaults extraction, smoke selector"
```

---

### Task 7: `lib/guard.py` — static lint (RED → GREEN)

**Files:**
- Create: `skills/orchestrator/lib/tests/test_guard.py`
- Create: `skills/orchestrator/lib/guard.py`

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills/orchestrator/lib/guard.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from skills.orchestrator.lib import guard


GOOD_TTNN = '''
import ttnn

def forward(x, weights):
    return ttnn.rms_norm(x, weight=weights["gamma"], epsilon=1e-6)
'''

BAD_CPU_CALL = '''
import ttnn

def forward(x, weights):
    h = ttnn.matmul(x, weights["w"])
    return h.cpu()  # shortcut
'''

BAD_TORCH_MATMUL = '''
import torch
import ttnn

def forward(x, weights):
    return torch.matmul(x, weights["w"])
'''

BAD_TODO_MARKER = '''
import ttnn

def forward(x, weights):
    # TODO: move to ttnn
    return x
'''


def test_lint_accepts_clean_block(tmp_path: Path):
    f = tmp_path / "rms_norm.py"
    f.write_text(GOOD_TTNN)
    issues = guard.lint_block(f)
    assert issues == []


def test_lint_rejects_cpu_call(tmp_path: Path):
    f = tmp_path / "bad_cpu.py"
    f.write_text(BAD_CPU_CALL)
    issues = guard.lint_block(f)
    assert any("cpu()" in i for i in issues)


def test_lint_rejects_torch_matmul(tmp_path: Path):
    f = tmp_path / "bad_matmul.py"
    f.write_text(BAD_TORCH_MATMUL)
    issues = guard.lint_block(f)
    assert any("torch.matmul" in i for i in issues)


def test_lint_rejects_todo_marker(tmp_path: Path):
    f = tmp_path / "bad_todo.py"
    f.write_text(BAD_TODO_MARKER)
    issues = guard.lint_block(f)
    assert any("TODO" in i for i in issues)
```

- [ ] **Step 2: Run — verify RED**

Run: `pytest skills/orchestrator/lib/tests/test_guard.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement static lint in `guard.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""No-shortcuts guard: static lint + traced-op assertion + host-resident check."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# (regex, human description) — applied to non-test files only
_FORBIDDEN_PATTERNS = [
    (re.compile(r"\.cpu\(\)"), ".cpu() in forward path"),
    (re.compile(r"\.numpy\(\)"), ".numpy() in forward path"),
    (re.compile(r"\btorch\.matmul\b"), "torch.matmul in forward path"),
    (re.compile(r"\btorch\.nn\.functional\.\w+"), "torch.nn.functional in forward path"),
    (re.compile(r"#\s*TODO[:\s].*move to ttnn", re.IGNORECASE), "TODO move-to-ttnn marker"),
    (re.compile(r"#\s*TODO[:\s].*ttnn", re.IGNORECASE), "TODO ttnn-related marker"),
]


def lint_block(tt_block_path: Path) -> list[str]:
    """Static lint — returns a list of issue strings. Empty list = clean."""
    text = Path(tt_block_path).read_text(errors="ignore")
    issues: list[str] = []
    for pat, desc in _FORBIDDEN_PATTERNS:
        m = pat.search(text)
        if m:
            issues.append(f"{tt_block_path}: {desc} (found {m.group(0)!r})")
    return issues
```

- [ ] **Step 4: Run tests — verify GREEN**

Run: `pytest skills/orchestrator/lib/tests/test_guard.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add skills/orchestrator/lib/guard.py skills/orchestrator/lib/tests/test_guard.py
git commit -m "skills/orchestrator: guard.py static lint for forbidden shortcuts"
```

---

### Task 8: `lib/guard.py` — traced-op assertion + host-resident check (RED → GREEN)

**Files:**
- Modify: `skills/orchestrator/lib/tests/test_guard.py` (append)
- Modify: `skills/orchestrator/lib/guard.py` (append)

- [ ] **Step 1: Append failing tests**

```python
def test_assert_traced_ops_passes_for_norm():
    traced = ["ttnn.embedding", "ttnn.rms_norm"]
    guard.assert_traced_ops(traced, block_kind="norm")  # no raise


def test_assert_traced_ops_fails_when_kernel_missing():
    traced = ["ttnn.embedding"]
    with pytest.raises(guard.GuardError, match="rms_norm|layer_norm"):
        guard.assert_traced_ops(traced, block_kind="norm")


def test_assert_traced_ops_attention_requires_matmul_and_softmax():
    traced = ["ttnn.matmul", "ttnn.softmax", "ttnn.matmul"]
    guard.assert_traced_ops(traced, block_kind="attention")


def test_assert_traced_ops_attention_missing_softmax_fails():
    with pytest.raises(guard.GuardError, match="softmax"):
        guard.assert_traced_ops(["ttnn.matmul"], block_kind="attention")


def test_verify_no_shortcuts_passes_when_host_resident_allowed(sample_state: dict, tmp_path: Path):
    sample_state["components"][1]["host_resident"]["allowed"] = True
    sample_state["components"][1]["host_resident"]["justification"] = "conv too large"
    # Even with a "bad" file, host_resident.allowed bypasses the lint
    bad = tmp_path / "attention.py"
    bad.write_text("def forward(x):\n    return x.cpu()\n")
    issues = guard.verify_no_shortcuts(
        sample_state["components"][1],
        tt_block_path=bad,
        traced_ops=[],
    )
    assert issues == []
```

- [ ] **Step 2: Run — verify RED**

Run: `pytest skills/orchestrator/lib/tests/test_guard.py -v`
Expected: 5 new failures.

- [ ] **Step 3: Append to `guard.py`**

```python
class GuardError(AssertionError):
    """Raised by assert_traced_ops when required kernels are missing."""


# block_kind → set of (kernel, count) requirements. A list means OR.
_REQUIRED_OPS: dict[str, list[list[str]]] = {
    "norm": [["ttnn.rms_norm", "ttnn.layer_norm"]],
    "attention": [["ttnn.matmul"], ["ttnn.softmax"]],
    "mlp": [["ttnn.matmul"]],
    "embedding": [["ttnn.embedding"]],
    "rope": [["ttnn.experimental.rotary_embedding_llama", "ttnn.rope"]],
    "linear": [["ttnn.matmul", "ttnn.linear"]],
}


def assert_traced_ops(traced_op_list: list[str], *, block_kind: str) -> None:
    reqs = _REQUIRED_OPS.get(block_kind)
    if reqs is None:
        # No expectations for unknown kinds (e.g. composite blocks)
        return
    seen = set(traced_op_list)
    for requirement_alts in reqs:
        if not any(alt in seen for alt in requirement_alts):
            raise GuardError(
                f"block_kind={block_kind!r} requires one of {requirement_alts}, "
                f"none found in traced ops: {sorted(seen)}"
            )


def verify_no_shortcuts(
    component: dict[str, Any],
    *,
    tt_block_path: Path,
    traced_ops: list[str],
) -> list[str]:
    """Full guard check. Returns list of issue strings (empty = clean)."""
    if component["host_resident"]["allowed"]:
        return []
    issues = lint_block(tt_block_path)
    try:
        assert_traced_ops(traced_ops, block_kind=component["kind"])
    except GuardError as e:
        issues.append(str(e))
    return issues
```

- [ ] **Step 4: Run tests — verify GREEN**

Run: `pytest skills/orchestrator/lib/tests/test_guard.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add skills/orchestrator/lib/guard.py skills/orchestrator/lib/tests/test_guard.py
git commit -m "skills/orchestrator: guard.py traced-op assertion + host-resident bypass"
```

---

### Task 9: `lib/cli.py` — Click entry for `start|resume|redo|skip` (RED → GREEN)

**Files:**
- Create: `skills/orchestrator/lib/tests/test_cli.py`
- Create: `skills/orchestrator/lib/cli.py`

- [ ] **Step 1: Write failing tests**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for skills/orchestrator/lib/cli.py."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from skills.orchestrator.lib.cli import cli


def test_start_creates_state_file(tmp_model_dir: Path):
    runner = CliRunner()
    target = tmp_model_dir / ".bringup_state.json"
    result = runner.invoke(
        cli,
        ["start", "--model-id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
         "--slug", "qwen3_tts", "--device", "n150",
         "--out", str(target)],
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    state = json.loads(target.read_text())
    assert state["model_id"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert state["device"] == "n150"
    assert state["arch_name"] == "wormhole_b0"  # filled from registry


def test_start_refuses_existing(tmp_state_file: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["start", "--model-id", "X/Y", "--slug", "y", "--device", "n150",
         "--out", str(tmp_state_file)],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_redo_resets_phase(tmp_state_file: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["redo", "--state", str(tmp_state_file), "--block", "Attention", "--phase", "ttnn"],
    )
    assert result.exit_code == 0, result.output
    s = json.loads(tmp_state_file.read_text())
    a = next(c for c in s["components"] if c["name"] == "Attention")
    assert a["ttnn"]["status"] == "pending"


def test_skip_records_justification(tmp_state_file: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli, ["skip", "--state", str(tmp_state_file), "--block", "Attention",
              "--phase", "ttnn", "--justify", "conv too large for L1",
              "--reference-link", "models/demos/qwen3_tts/tt/speech_decoder.py"],
    )
    assert result.exit_code == 0
    s = json.loads(tmp_state_file.read_text())
    a = next(c for c in s["components"] if c["name"] == "Attention")
    assert a["host_resident"]["allowed"] is True


def test_resume_demotes_in_progress(tmp_state_file: Path):
    s = json.loads(tmp_state_file.read_text())
    s["components"][1]["ttnn"]["status"] = "in_progress"
    tmp_state_file.write_text(json.dumps(s))
    runner = CliRunner()
    result = runner.invoke(cli, ["resume", "--state", str(tmp_state_file)])
    assert result.exit_code == 0, result.output
    s2 = json.loads(tmp_state_file.read_text())
    assert s2["components"][1]["ttnn"]["status"] == "pending"
```

- [ ] **Step 2: Run — verify RED**

Run: `pytest skills/orchestrator/lib/tests/test_cli.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `lib/cli.py`**

```python
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CLI entry for the bringup orchestrator. Called by skills/orchestrator/SKILL.md."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from skills.orchestrator.lib import state as state_mod
from skills.orchestrator.lib.device import DEVICE_REGISTRY


@click.group()
def cli() -> None:
    """Bringup orchestrator helper CLI."""


@cli.command()
@click.option("--model-id", required=True, help="HuggingFace model id, e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base")
@click.option("--slug", required=True, help="Short folder name under models/demos/")
@click.option("--device", required=True, type=click.Choice(list(DEVICE_REGISTRY)))
@click.option("--out", required=True, type=click.Path(dir_okay=False))
def start(model_id: str, slug: str, device: str, out: str) -> None:
    """Write skeleton .bringup_state.json. Refuses if file exists."""
    arch = DEVICE_REGISTRY[device]["arch"]
    try:
        state_mod.bootstrap(
            Path(out), model_id=model_id, model_slug=slug, device=device, arch_name=arch,
        )
    except FileExistsError as e:
        click.echo(str(e), err=True)
        sys.exit(2)
    click.echo(f"bootstrapped {out}")


@cli.command()
@click.option("--state", "state_path", required=True, type=click.Path(exists=True, dir_okay=False))
def resume(state_path: str) -> None:
    """Validate state, demote in_progress → pending."""
    p = Path(state_path)
    s = state_mod.load_state(p)
    state_mod.demote_in_progress(s)
    state_mod.save_state(p, s)
    click.echo(f"resumed {state_path}")


@cli.command()
@click.option("--state", "state_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--block", required=True)
@click.option("--phase", required=True, type=click.Choice(list(state_mod.VALID_PHASES)))
def redo(state_path: str, block: str, phase: str) -> None:
    p = Path(state_path)
    s = state_mod.load_state(p)
    state_mod.redo(s, block=block, phase=phase)
    state_mod.save_state(p, s)
    click.echo(f"redo {block}:{phase}")


@cli.command()
@click.option("--state", "state_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--block", required=True)
@click.option("--phase", required=True, type=click.Choice(list(state_mod.VALID_PHASES)))
@click.option("--justify", required=True, help="Why this block stays host-resident.")
@click.option("--reference-link", required=True, help="Path/URL to the reference impl that does the same.")
def skip(state_path: str, block: str, phase: str, justify: str, reference_link: str) -> None:
    p = Path(state_path)
    s = state_mod.load_state(p)
    state_mod.skip(s, block=block, phase=phase, justification=justify, reference_link=reference_link)
    state_mod.save_state(p, s)
    click.echo(f"skip {block}:{phase}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests — verify GREEN**

Run: `pytest skills/orchestrator/lib/tests/test_cli.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the full lib test suite as a sanity check**

Run: `pytest skills/orchestrator/lib/tests/ -v`
Expected: ~31 passed, no failures.

- [ ] **Step 6: Commit**

```bash
git add skills/orchestrator/lib/cli.py skills/orchestrator/lib/tests/test_cli.py
git commit -m "skills/orchestrator: cli.py start/resume/redo/skip Click commands"
```

---

### Task 10: Architecture worker prompt

**Files:**
- Create: `skills/orchestrator/workers/architecture-worker.md`

- [ ] **Step 1: Write the worker prompt**

```markdown
# Architecture Worker

You are a sub-agent dispatched by the bringup orchestrator. Run **once** per bring-up, before any other phase.

## Inputs (read from the prompt passed to you)
- `model_id` — HuggingFace model id, e.g. `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `model_slug` — short folder name under `models/demos/`
- `device` — one of `n150 | n300 | p150 | t3k | tg`
- `arch_name` — e.g. `wormhole_b0`

## What you do

1. **Invoke the `architecture` skill** via the Skill tool.
2. Follow it end-to-end. Produce `models/demos/<model_slug>/ARCHITECTURE.md` per its conventions (full Component Inventory, weight mapping, reference implementations).
3. Pick reference TTNN implementations **conditioned on the target device**. For `t3k`/`tg`, prefer `models/demos/llama3_70b_galaxy/`. For single-chip Wormhole, prefer `models/demos/llama3_subdevices/` or similar. Document the choice per block.
4. **Emit a sidecar file** at `models/demos/<model_slug>/architecture_inventory.json` matching this schema:

```json
{
  "components": [
    {
      "name": "RMSNorm",
      "kind": "norm",
      "reference_impl": "models/common/rmsnorm.py",
      "depends_on": []
    },
    {
      "name": "Attention",
      "kind": "attention",
      "reference_impl": "models/demos/llama3_70b_galaxy/tt/llama_attention.py",
      "depends_on": ["RMSNorm", "Linear", "RoPE"]
    }
  ]
}
```

`kind` ∈ `{norm, attention, mlp, embedding, rope, linear, decoder_layer, composite}` (composite for sub-models and the full model — guard expectations relaxed for these).

`depends_on` enforces bottom-up bring-up. List small primitives first (Linear, RMSNorm, RoPE), then composite blocks (Attention, MLP), then integrators (DecoderLayer, Talker, full model).

## Result Format

Return exactly one JSON document on stdout:

```json
{
  "block": null,
  "phase": "architecture",
  "status": "ok",
  "pcc": null,
  "artifacts": [
    "models/demos/<slug>/ARCHITECTURE.md",
    "models/demos/<slug>/architecture_inventory.json"
  ],
  "notes": "<one-sentence summary>",
  "last_error": null,
  "hang_detected": false
}
```

On failure, set `status="fail"` and put the failure reason in `last_error`. Do not retry — the orchestrator handles retries.
```

- [ ] **Step 2: Sanity-check the markdown lints clean (no unclosed code fences)**

Run: `grep -c '^\`\`\`' skills/orchestrator/workers/architecture-worker.md`
Expected: an even number (matched fences).

- [ ] **Step 3: Commit**

```bash
git add skills/orchestrator/workers/architecture-worker.md
git commit -m "skills/orchestrator: architecture worker prompt"
```

---

### Task 11: Reference worker prompt

**Files:**
- Create: `skills/orchestrator/workers/reference-worker.md`

- [ ] **Step 1: Write the prompt**

```markdown
# Reference Worker

You are a sub-agent dispatched by the bringup orchestrator. You may run **in parallel with other reference workers** (one per block). You do **not** touch the TT device.

## Inputs
- `model_slug`
- `block` — single block name from `architecture_inventory.json`
- `reference_impl` — pointer to the chosen reference (informational; you implement, not copy)
- `model_id` — for HuggingFace cross-checks

## What you do

1. **Invoke `superpowers:test-driven-development`** to set up the TDD scaffold for this block.
2. **Invoke the `reference` skill.** Implement the block in `models/demos/<slug>/reference/functional.py` per its conventions.
3. Generate golden tensors in `models/demos/<slug>/reference/golden/<block>.pt` from the **official HuggingFace** model — not your own implementation.
4. PCC vs HF must be > 0.99. If lower, set `status="fail"` with the PCC and stop — debug-worker will be dispatched.

## Result Format

```json
{
  "block": "<block name>",
  "phase": "reference",
  "status": "ok",
  "pcc": <float>,
  "artifacts": [
    "models/demos/<slug>/reference/functional.py",
    "models/demos/<slug>/reference/golden/<block>.pt"
  ],
  "notes": "<one-sentence summary>",
  "last_error": null,
  "hang_detected": false
}
```

Failure: `status="fail"`, include the failure trace in `last_error`. Do not retry.
```

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/workers/reference-worker.md
git commit -m "skills/orchestrator: reference worker prompt"
```

---

### Task 12: TTNN worker prompt

**Files:**
- Create: `skills/orchestrator/workers/ttnn-worker.md`

- [ ] **Step 1: Write the prompt**

```markdown
# TTNN Worker

You are a sub-agent dispatched by the bringup orchestrator. You **hold the device lock**. No other worker is running on the device concurrently.

## Inputs
- `model_slug`
- `block` — single block name
- `kind` — block kind (norm | attention | mlp | embedding | rope | linear | decoder_layer | composite)
- `device` — n150 | n300 | p150 | t3k | tg
- `device_defaults` — dict pulled from the reference TTNN model (l1_small_size etc.)

## What you do

1. **Invoke the `ttnn` skill.** Implement the block in `models/demos/<slug>/tt/<block_snake>.py`.
2. Run the PCC test against `models/demos/<slug>/reference/golden/<block>.pt`. PCC must be > 0.99.
3. **Capture the traced op list** for the forward path and return it under `traced_ops`. (Use `ttnn.tracing` / `tracy` per the ttnn skill's "Profiler-driven workflow" section.)
4. If a device hang is detected, set `hang_detected=true` and **stop immediately** — do not run `tt-smi -r` yourself; the orchestrator will.

## Result Format

```json
{
  "block": "<block name>",
  "phase": "ttnn",
  "status": "ok",
  "pcc": <float>,
  "artifacts": ["models/demos/<slug>/tt/<block_snake>.py"],
  "traced_ops": ["ttnn.matmul", "ttnn.softmax", "ttnn.matmul"],
  "notes": "<one-sentence summary>",
  "last_error": null,
  "hang_detected": false
}
```

If PCC < 0.99: `status="fail"`, include the actual PCC. The orchestrator routes you to debug-worker next.
If hang: `status="fail"`, `hang_detected=true`, do not include `traced_ops`.
```

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/workers/ttnn-worker.md
git commit -m "skills/orchestrator: ttnn worker prompt"
```

---

### Task 13: Debug worker prompt

**Files:**
- Create: `skills/orchestrator/workers/debug-worker.md`

- [ ] **Step 1: Write the prompt**

```markdown
# Debug Worker

You are a sub-agent dispatched by the orchestrator after a TTNN failure. You **hold the device lock**.

## Inputs
- `model_slug`
- `block` — block whose TTNN PCC dropped below 0.99 (or hung)
- `last_pcc` — most recent PCC value
- `last_error` — error message or hang signal
- `debug_attempts` — how many times this block has been debug'd

## What you do

1. **Invoke `superpowers:systematic-debugging` FIRST.** Form a hypothesis before touching code.
2. **Invoke the `debug` skill.** Diagnose the root cause.
3. You may edit `models/demos/<slug>/tt/<block_snake>.py`.
4. You may edit `models/demos/<slug>/reference/functional.py` **only if** you can prove the reference change does not change the model. Concretely: re-run reference vs the official HuggingFace model and verify PCC vs HF remains > 0.99. If the reference change pulls reference away from HF, **reject your own change** and push the fix back into TTNN.
5. After the fix, re-run the TTNN PCC test. PCC > 0.99 is required to claim success.

## Result Format

```json
{
  "block": "<block name>",
  "phase": "debug",
  "status": "ok",
  "pcc": <float>,
  "artifacts": [
    "models/demos/<slug>/tt/<block_snake>.py",
    "models/demos/<slug>/reference/functional.py"
  ],
  "hypothesis": "<one-line root cause>",
  "reference_touched": false,
  "reference_hf_pcc": null,
  "notes": "<what you changed>",
  "last_error": null,
  "hang_detected": false
}
```

If `reference_touched=true`, you MUST set `reference_hf_pcc` to the post-change PCC vs official HF. The orchestrator rejects any debug result where `reference_touched=true` and `reference_hf_pcc < 0.99`.

If you cannot find a fix in this attempt, `status="fail"`. The orchestrator will retry (or escalate after max attempts).
```

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/workers/debug-worker.md
git commit -m "skills/orchestrator: debug worker prompt"
```

---

### Task 14: Optimization worker prompt

**Files:**
- Create: `skills/orchestrator/workers/optimization-worker.md`

- [ ] **Step 1: Write the prompt**

```markdown
# Optimization Worker

You are a sub-agent dispatched by the orchestrator once a block (or the full model) has TTNN PCC > 0.99 and all `depends_on` are TTNN-done. You **hold the device lock**.

## Inputs
- `model_slug`
- `block` — block to optimize (or `"full_model"`)
- `device`
- `reference_impl` — pointer to the reference TTNN model whose perf targets we inherit

## What you do

1. **Invoke the `optimization` skill** end-to-end (profiler-driven loop: measure → bucket → attack → verify).
2. Stay within the PCC > 0.99 acceptance bar throughout. If an optimization drops PCC, revert it.
3. The success target is whatever throughput / latency the reference TTNN model achieves on the same device, **or** a project-specific target set in `models/demos/<slug>/PERF_TARGET.md` if it exists.

## Result Format

```json
{
  "block": "<block name or 'full_model'>",
  "phase": "optimization",
  "status": "ok",
  "pcc": <float>,
  "perf": {"tok_per_sec": <float>, "ms_per_frame": <float>},
  "artifacts": ["models/demos/<slug>/tt/<block_snake>.py"],
  "notes": "<one-sentence summary of wins>",
  "last_error": null,
  "hang_detected": false
}
```

Hang handling: same as ttnn-worker.
```

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/workers/optimization-worker.md
git commit -m "skills/orchestrator: optimization worker prompt"
```

---

### Task 15: `tick.md` — per-tick re-entry prompt

**Files:**
- Create: `skills/orchestrator/tick.md`

- [ ] **Step 1: Write the tick prompt**

```markdown
# Bringup Orchestrator Tick

You are the orchestrator running ONE tick. Be brief; do not over-explain.

## Inputs
- `model_path` — absolute path to the model folder (contains `.bringup_state.json`)

## Tick procedure

1. **Load state.**
   ```bash
   python -m skills.orchestrator.lib.cli resume --state "<model_path>/.bringup_state.json"
   ```
   (This validates the schema and demotes any `in_progress` rows.)

2. **Pick next action** via the dag:
   ```python
   from skills.orchestrator.lib import state as st, dag
   s = st.load_state(model_path / ".bringup_state.json")
   action = dag.next_action(s)
   ```

3. **Branch on `action.kind`:**

   - **`done`** → exit, do NOT call ScheduleWakeup.
   - **`deadlock`** → write a deadlock row to `tick_log`, render `BRINGUP_LOG.md`, commit `bringup(<slug>): deadlock — <block> blocks downstream`, exit, do NOT reschedule.
   - **`architecture`** → dispatch one `architecture-worker` Agent. Parse its JSON result. On `ok`, load the emitted `architecture_inventory.json` and populate `state.components[]`. Increment `attempts` only on `fail`.
   - **`reference`** → dispatch up to `state.config.max_parallel_reference` `reference-worker` Agents in parallel (one per block in `action.blocks`). Wait for all. Update each block's `reference.status` from the worker result.
   - **`ttnn`** → dispatch ONE `ttnn-worker` Agent for `action.blocks[0]`. On `ok`, run `guard.verify_no_shortcuts(component, tt_block_path, worker_result["traced_ops"])`. If guard issues, mark `ttnn.status="failing"`, record issues in `last_error`. Else mark `ttnn.status="done"`.
   - **`debug`** → dispatch ONE `debug-worker` Agent. On `ok`, set the TTNN status to `done`. On `fail`, increment `debug.attempts`; if `attempts >= max_attempts_per_phase`, set `ttnn.status="blocked"`.
   - **`optimization`** → dispatch ONE `optimization-worker` Agent. Set `optimization.status` from the result.

4. **Hang recovery.** If any worker returned `hang_detected=true`:
   ```bash
   tt-smi -r
   ```
   Then dispatch a smoke-check via `ttnn-worker` on `device.last_done_block(state)`. If the smoke check fails, set `status="blocked"` on the block being attempted AND on the smoke-check block, write `tick_log` row `"SMOKE_FAILED"`, escalate to user in commit message, do NOT reschedule.

5. **Append `tick_log` row**, render `BRINGUP_LOG.md`, save state.

6. **Commit:**
   ```bash
   git add <model_path>/.bringup_state.json <model_path>/BRINGUP_LOG.md <touched files>
   git commit -m "bringup(<slug>): <phase> <block> <status>"
   ```

7. **Schedule the next tick:**
   ```
   ScheduleWakeup(
     delaySeconds=state.config.tick_interval_sec,
     prompt="/bringup --resume <model_path>",
     reason="<next phase preview>"
   )
   ```
   (Skip this call on `done`, `deadlock`, or `SMOKE_FAILED`.)

## Worker dispatch contract

When dispatching a worker, use the `Agent` tool with:
- `subagent_type="general-purpose"`
- a prompt that starts with: `Read skills/orchestrator/workers/<phase>-worker.md and follow it. Inputs: <inputs JSON>`

Workers MUST return a single JSON document; if the JSON is malformed, treat it as `status="fail"`, `last_error="malformed worker result"`.
```

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/tick.md
git commit -m "skills/orchestrator: tick.md per-tick orchestrator prompt"
```

---

### Task 16: `SKILL.md` — `/bringup` user entry

**Files:**
- Create: `skills/orchestrator/SKILL.md`

- [ ] **Step 1: Write the skill entry**

````markdown
---
name: bringup
description: Autonomously bring up a HuggingFace model in TTNN, end-to-end. Architecture → reference → ttnn → debug → optimization, with PCC > 0.99 gates and full resumability. Use when starting a new model bring-up or resuming an existing one.
---

# /bringup — Autonomous Model Bringup

## Purpose

Drive the full TTNN bring-up pipeline for any HuggingFace model on any supported Tenstorrent device. State lives on disk in `.bringup_state.json`; the pipeline is resumable across `/clear`, crashes, and reboots.

## Usage

```
/bringup <model_id> --device <n150|n300|p150|t3k|tg>      # start fresh
/bringup --resume <model_path>                            # resume after interruption
/bringup --redo <slug>:<block>:<phase>                    # reset one phase cell, re-dispatch
/bringup --skip <slug>:<block>:<phase> --justify "<text>" # mark host-resident, with reason
```

## What this skill does (first run)

1. Resolve `<model_id>` → `model_slug` (the part after `/`, lowercased, dashes → underscores).
2. Create `models/demos/<slug>/` if it does not exist.
3. Bootstrap state:
   ```bash
   python -m skills.orchestrator.lib.cli start \
       --model-id "<model_id>" --slug "<slug>" \
       --device "<device>" \
       --out "models/demos/<slug>/.bringup_state.json"
   ```
4. Commit: `chore(<slug>): start bringup`.
5. **Invoke `/loop` internally** via the Skill tool with the sub-prompt:
   ```
   /bringup --resume models/demos/<slug>
   ```
   `/loop` enters dynamic mode; the user does not type `/loop`.
6. The first tick runs `tick.md`, which dispatches the architecture worker.

## What this skill does (`--resume`)

1. Validate the state file exists and `schema_version` matches.
2. Demote any `in_progress` rows to `pending` via `cli resume`.
3. **Smoke check** the most recent `ttnn.status=done` block by dispatching `ttnn-worker` on it. If it fails, escalate to user (something deeper than the current bringup is broken).
4. Read `tick.md` and run one tick.

## What this skill does (`--redo` / `--skip`)

Single CLI call, single commit, done. No `/loop`.

## Inspect

```bash
cat models/demos/<slug>/.bringup_state.json | jq '.components[]|{name, ttnn:.ttnn.status, pcc:.ttnn.pcc}'
cat models/demos/<slug>/BRINGUP_LOG.md
git log --oneline -- models/demos/<slug>/.bringup_state.json
```

## Stop

Any user input during a tick interrupts the loop. State is on disk; resume with `/bringup --resume <model_path>`.

## Implementation notes

- All decision logic lives in `skills/orchestrator/lib/` (pure Python, fully unit-tested).
- Worker prompts are in `skills/orchestrator/workers/*.md` and invoke the existing `architecture`, `reference`, `ttnn`, `debug`, `optimization` skills.
- The five existing per-phase skills are not modified by this orchestrator.
- See `skills/orchestrator/SPEC.md` for the design rationale and `skills/orchestrator/tick.md` for the per-tick procedure.
````

- [ ] **Step 2: Commit**

```bash
git add skills/orchestrator/SKILL.md
git commit -m "skills/orchestrator: SKILL.md /bringup user entry"
```

---

### Task 17: Permissions wiring for unattended runs

**Files:**
- Modify: `.claude/settings.local.json` (read first, then surgical edit) — or `.claude/settings.json` if it exists.

- [ ] **Step 1: Read the current local settings**

```bash
cat .claude/settings.local.json 2>/dev/null || echo "{}"
```

- [ ] **Step 2: Add permissions for the orchestrator's Bash calls**

The orchestrator needs these to run unattended:
- `python -m skills.orchestrator.lib.cli *`
- `tt-smi -r`
- `git add models/demos/*/.bringup_state.json`, `git add models/demos/*/BRINGUP_LOG.md`, `git commit -m bringup*`

Use the `update-config` skill (or hand-edit) to add them to the `permissions.allow` array. Example block to add:

```json
{
  "permissions": {
    "allow": [
      "Bash(python -m skills.orchestrator.lib.cli:*)",
      "Bash(tt-smi -r)",
      "Bash(git add models/demos/*/.bringup_state.json:*)",
      "Bash(git add models/demos/*/BRINGUP_LOG.md:*)",
      "Bash(git commit -m \"bringup*\")"
    ]
  }
}
```

- [ ] **Step 3: Verify**

Run: `cat .claude/settings.local.json | jq '.permissions.allow'`
Expected: the new entries present.

- [ ] **Step 4: Commit**

```bash
git add .claude/settings.local.json
git commit -m "skills/orchestrator: settings.local.json permissions for unattended runs"
```

---

### Task 18: End-to-end smoke test on qwen3_tts (`--resume` mode, non-mutating)

**Goal:** Verify the orchestrator can load real qwen3_tts state, decide what's next, and dispatch one worker, **without actually running a model phase** — done by injecting a `dry-run` env var that makes the tick skip the `Agent` dispatch and instead print the planned action.

**Files:**
- Modify: `skills/orchestrator/tick.md` (add a `BRINGUP_DRY_RUN` short-circuit clause)
- Modify: `skills/orchestrator/lib/cli.py` (add `tick --dry-run` subcommand for the smoke)
- Modify: `skills/orchestrator/lib/tests/test_cli.py` (test the dry-run path)

- [ ] **Step 1: Write a dry-run test (RED)**

Append to `test_cli.py`:

```python
def test_tick_dry_run_reports_next_action(tmp_state_file: Path, capsys):
    runner = CliRunner()
    result = runner.invoke(cli, ["tick", "--state", str(tmp_state_file), "--dry-run"])
    assert result.exit_code == 0
    # Sample state has Attention ttnn-pending with deps satisfied → next is ttnn[Attention]
    assert "ttnn" in result.output
    assert "Attention" in result.output
```

- [ ] **Step 2: Run — verify RED**

Run: `pytest skills/orchestrator/lib/tests/test_cli.py::test_tick_dry_run_reports_next_action -v`
Expected: `Usage: cli ...` error (no such subcommand).

- [ ] **Step 3: Add the `tick` subcommand to `cli.py`**

Append to `cli.py`:

```python
@cli.command()
@click.option("--state", "state_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--dry-run", is_flag=True, help="Print the next action without dispatching any worker.")
def tick(state_path: str, dry_run: bool) -> None:
    """One orchestrator tick. With --dry-run, only prints the planned action."""
    from skills.orchestrator.lib import dag as dag_mod

    s = state_mod.load_state(Path(state_path))
    state_mod.demote_in_progress(s)
    action = dag_mod.next_action(s)
    if dry_run:
        click.echo(f"next_action: kind={action.kind} blocks={action.blocks}")
        return
    # Non-dry tick is implemented by tick.md prompt; the CLI does not dispatch agents itself.
    click.echo(
        "non-dry tick must be driven by skills/orchestrator/tick.md (Agent tool not available from CLI)",
        err=True,
    )
    sys.exit(3)
```

- [ ] **Step 4: Run — verify GREEN**

Run: `pytest skills/orchestrator/lib/tests/test_cli.py::test_tick_dry_run_reports_next_action -v`
Expected: 1 passed.

- [ ] **Step 5: Live smoke against qwen3_tts (read-only)**

First create a synthetic `.bringup_state.json` for qwen3_tts that reflects current BRINGUP_LOG.md (since the orchestrator did not produce it):

```bash
# Construct a state file by hand from BRINGUP_LOG.md current state.
# Place at: /tmp/qwen3_tts_smoke_state.json
# Use cli to validate it:
python -m skills.orchestrator.lib.cli resume --state /tmp/qwen3_tts_smoke_state.json
python -m skills.orchestrator.lib.cli tick --state /tmp/qwen3_tts_smoke_state.json --dry-run
```

Expected output: a single line `next_action: kind=optimization blocks=[...]` or similar — i.e. the orchestrator agrees with the real next step on qwen3_tts.

(If the synthetic state contradicts reality, that's diagnostic, not a failure of this task. The point of the smoke is that the orchestrator runs end-to-end without crashing.)

- [ ] **Step 6: Commit**

```bash
git add skills/orchestrator/lib/cli.py skills/orchestrator/lib/tests/test_cli.py
git commit -m "skills/orchestrator: tick --dry-run for smoke testing"
```

---

### Task 19: Final lint + full-suite verification + memory note

**Files:**
- Modify: `/local/ttuser/.claude/projects/-local-ttuser-ssinghal-tt-metal/memory/MEMORY.md`
- Create: `/local/ttuser/.claude/projects/-local-ttuser-ssinghal-tt-metal/memory/feedback_bringup_orchestrator.md`

- [ ] **Step 1: Run the full lib test suite**

Run: `pytest skills/orchestrator/lib/tests/ -v`
Expected: ~32 passed, 0 failed.

- [ ] **Step 2: Verify the markdown structure**

Run:
```bash
ls skills/orchestrator/
ls skills/orchestrator/workers/
ls skills/orchestrator/lib/
```
Expected: SKILL.md, SPEC.md, tick.md, workers/ (5 files), lib/ (5 .py + tests/).

- [ ] **Step 3: Add a feedback memory so future sessions know the invocation surface**

Write `feedback_bringup_orchestrator.md`:

```markdown
---
name: bringup-orchestrator-usage
description: How to invoke the autonomous model bring-up orchestrator and resume it after interruption
metadata:
  type: feedback
---

The `/bringup` skill at `skills/orchestrator/` autonomously drives full TTNN model bring-up.

**Why:** The pipeline (architecture → reference → ttnn → debug → optimization) is well-defined per-block, and most of the per-phase decisions are mechanical given the existing five per-phase skills.

**How to apply:**
- Start: `/bringup <hf_id> --device <n150|n300|p150|t3k|tg>` — internally invokes `/loop`, user does not type `/loop`.
- Resume: `/bringup --resume models/demos/<slug>` (after `/clear`, crash, or reboot).
- Nudge: `/bringup --redo <slug>:<block>:<phase>` or `/bringup --skip ... --justify "..."`.
- State source of truth: `models/demos/<slug>/.bringup_state.json`. `BRINGUP_LOG.md` is rendered from it.
- TTNN/debug/optimization phases are serial (device contention); reference fans out.
- Spec: `skills/orchestrator/SPEC.md`.
```

- [ ] **Step 4: Append the index line to MEMORY.md**

Append to `MEMORY.md`:
```
- [Bringup orchestrator usage](feedback_bringup_orchestrator.md) — /bringup <hf_id> --device <dev>; resumable via JSON state; reference fans out, ttnn+ serial
```

- [ ] **Step 5: Commit**

```bash
git add skills/orchestrator/
git commit -m "skills/orchestrator: implementation complete (lib + workers + entry)"
```

(Memory files are outside the repo — no commit needed.)

---

## Self-review notes

- **Spec coverage:** every section of `skills/orchestrator/SPEC.md` is implemented by at least one task above (state schema → Tasks 2-4; phase rules → Task 5; device registry → Task 6; no-shortcuts guard → Tasks 7-8; CLI surface → Task 9; workers → Tasks 10-14; tick → Task 15; SKILL.md → Task 16; permissions → Task 17; smoke → Task 18).
- **Bottom-up DAG** is enforced in `dag.py::_dep_satisfied` (Task 5) and the architecture worker is responsible for emitting the DAG (Task 10).
- **Resumability** is exercised by `cli resume` (Task 9) and the smoke (Task 18). `demote_in_progress` is the safety net for mid-tick crashes.
- **Deadlock detection** is in `dag.is_deadlocked` (Task 5) and surfaced by the tick (Task 15).
- **No placeholders.** Every code step has runnable code; every test step has an assertion.
- **One ambiguity in the spec** — the `optimization.status` field with `"blocked_on": "ttnn"` example in the SPEC was clarified to mean a separate `blocked_on` key (not a value). The state-schema implementation in Task 2/4 keeps the simpler shape (`status` is one of the seven enum values; no separate `blocked_on` key needed because the DAG infers blocking from `depends_on`).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-27-bringup-orchestrator.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
