# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""State load/save with schema validation for the bringup orchestrator.

The state file is the source of truth for orchestrator progress; see
`skills/orchestrator/SPEC.md` §State schema. This module provides:

- `SchemaError`: raised on missing or invalid required keys.
- `load_state(path)`: read + validate JSON.
- `save_state(path, state)`: validate + atomic write.
- `bootstrap(model_id, device, arch_name)`: build a fresh skeleton state.
- `resume_normalize(state)`: make a state safe to resume after a crash.
- `render_log(state)`: format the state as a BRINGUP_LOG.md markdown document.
- `redo(state, block, phase)`: reset a phase cell to pending for retry.
- `skip(state, block, phase, justify, reference_link)`: mark a block as
  host-resident-allowed with an audited justification.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

# Top-level required keys per SPEC.md §State schema. Deeper validation of nested
# structures (components, locks contents, etc.) is intentionally deferred to
# future tasks.
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_slug",
        "device",
        "arch_name",
        "started_at",
        "updated_at",
        "components",
        "locks",
        "tick_log",
        "config",
    }
)

_SCHEMA_VERSION = 1

# The four phase keys present on each component (per SPEC.md). Used by
# resume_normalize to demote in_progress workers whose owning session is gone.
# Public because downstream tasks (render_log, redo, skip) consume the same
# canonical phase ordering.
PHASE_NAMES = ("reference", "ttnn", "debug", "optimization")

# Default tunables baked into a fresh state by bootstrap(). Public because
# downstream tasks read these as the canonical defaults. The dict is flat and
# holds only primitives, so a shallow copy in bootstrap is sufficient.
DEFAULT_CONFIG = {
    "max_parallel_reference": 4,
    "max_attempts_per_phase": 10,
    "tick_interval_sec": 60,
}

# SPEC.md bounds the on-disk tick_log to the last N entries; full history
# lives in git via per-tick commits. Any mutator that appends to tick_log
# must call _trim_tick_log afterwards.
_MAX_TICK_LOG_ENTRIES = 100


class SchemaError(Exception):
    """Raised when a state dict fails schema validation."""


def _validate(state: dict) -> None:
    """Validate top-level required keys and schema_version.

    - Raises ``SchemaError`` listing all missing required keys (sorted, so the
      error message is deterministic).
    - Raises ``SchemaError`` if ``schema_version`` is not ``_SCHEMA_VERSION``.
    - Emits a ``UserWarning`` (does not raise) for unknown top-level keys.
    """
    missing = sorted(_REQUIRED_KEYS - state.keys())
    if missing:
        raise SchemaError(f"missing required keys: {missing}")

    version = state["schema_version"]
    if version != _SCHEMA_VERSION:
        raise SchemaError(f"schema_version must be {_SCHEMA_VERSION}, got {version!r}")

    extras = set(state.keys()) - _REQUIRED_KEYS
    if extras:
        warnings.warn(
            f"unknown top-level state fields: {sorted(extras)}",
            UserWarning,
            stacklevel=2,
        )


def load_state(path: PathLike) -> dict:
    """Load a state file from JSON and validate it.

    Raises ``FileNotFoundError`` if the file does not exist (default open
    behavior) and ``SchemaError`` if the loaded dict fails validation.
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        state = json.load(f)
    _validate(state)
    return state


def save_state(path: PathLike, state: dict) -> None:
    """Validate ``state`` and atomically write it as JSON to ``path``.

    Atomicity is achieved by writing to ``<path>.tmp`` first, then
    ``os.replace`` onto the final path. Parent directories are created
    if they do not exist. If validation fails the target file is not
    created or modified.
    """
    _validate(state)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Construct tmp via parent + name (not Path.with_suffix, which raises
    # ValueError on paths with no suffix or trailing dots).
    tmp = p.parent / (p.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
    except BaseException:
        # On any failure during the JSON write, remove the partial tmp file
        # so it cannot be mistaken for a valid state later. The final path
        # is untouched because os.replace has not run yet.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    os.replace(tmp, p)


def _utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 with a trailing 'Z' (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def bootstrap(model_id: str, device: str, arch_name: str) -> dict:
    """Create a fresh state dict for a new model bring-up.

    The returned dict passes ``_validate()``. The caller is responsible for
    persisting it (e.g. via ``save_state``); ``bootstrap`` itself does no I/O.

    ``model_slug`` is derived from ``model_id`` by lowercasing and replacing
    ``/`` and ``-`` with ``_``. Other characters (including ``.``) are kept
    so versioned model identifiers like ``...-1.7B-Base`` remain readable.
    """
    slug = model_id.lower().replace("/", "_").replace("-", "_")
    # Single timestamp reused for both fields so they are structurally equal
    # at bootstrap time. updated_at advances when ticks mutate state.
    now = _utc_now_iso()
    return {
        "schema_version": _SCHEMA_VERSION,
        "model_id": model_id,
        "model_slug": slug,
        "device": device,
        "arch_name": arch_name,
        "started_at": now,
        "updated_at": now,
        "components": [],
        "locks": {"device": {"held_by": None, "held_since": None}},
        "tick_log": [],
        "config": dict(DEFAULT_CONFIG),
    }


def resume_normalize(state: dict) -> dict:
    """Make ``state`` safe to resume after a session crash or ``/clear``.

    The worker that owned any ``in_progress`` phase is, by definition, gone
    by the time we're resuming — its work cannot be assumed complete. Demote
    every such phase to ``pending`` so the next tick re-dispatches it. Also
    drop any held device lock for the same reason.

    Mutates ``state`` in place and returns it (so callers can chain
    ``state = resume_normalize(load_state(path))``).
    """
    # Top-level keys (components, locks, locks.device) are guaranteed by
    # _validate / the canonical schema, so they're accessed directly.
    # Per-component phase keys are NOT enforced by _validate, so they keep
    # the defensive guard — a component may legitimately omit phases it has
    # not reached yet.
    for component in state["components"]:
        for phase in PHASE_NAMES:
            phase_dict = component.get(phase)
            if isinstance(phase_dict, dict) and phase_dict.get("status") == "in_progress":
                phase_dict["status"] = "pending"

    device_lock = state["locks"]["device"]
    if device_lock.get("held_by") is not None:
        device_lock["held_by"] = None
        device_lock["held_since"] = None

    return state


# ---------------------------------------------------------------------------
# render_log: state -> BRINGUP_LOG.md markdown
# ---------------------------------------------------------------------------

# Em dash used to render missing/None scalar cells (status, pcc) in the table.
# A plain "-" would conflict with markdown's table-alignment row syntax and
# arguably look like a typo; the em dash reads as "intentionally empty".
_EM_DASH = "—"


def _escape_cell(text: str) -> str:
    """Escape characters that break a markdown table cell.

    - ``|`` becomes ``\\|`` so it doesn't terminate the cell.
    - Newlines collapse to a single space so the row stays on one line.
    """
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt_pcc(pcc) -> str:
    """Render a PCC value with 6 decimal places, or em dash if absent."""
    if pcc is None:
        return _EM_DASH
    return f"{pcc:.6f}"


def _fmt_notes(phase_dict: dict) -> str:
    """Pick the notes cell for a phase row.

    For failing/blocked phases, prefer ``last_error`` (the operator wants to
    see *why* it's stuck). For everything else fall back to ``notes``. Returns
    empty string when neither is present; callers feed the result through
    ``_escape_cell``.
    """
    status = phase_dict.get("status")
    if status in ("failing", "blocked"):
        text = phase_dict.get("last_error") or phase_dict.get("notes") or ""
    else:
        text = phase_dict.get("notes") or ""
    return text


def render_log(state: dict) -> str:
    """Render ``state`` as a BRINGUP_LOG.md markdown document.

    Pure function: no I/O, no mutation of ``state``. The output ends with a
    single trailing newline so writing it directly to a file produces a
    POSIX-clean file with no trailing blank lines.

    Layout (top to bottom): header block, ``## Block Status`` table with one
    row per ``(component, phase)`` pair iterated in component order then
    ``PHASE_NAMES`` order, ``## Recent Ticks`` (last 10 tick_log entries,
    chronological), and ``## Host-Resident Exceptions`` (components flagged
    with ``host_resident.allowed = True``, or ``_None._`` if none).
    """
    lines = []

    # --- Header ---
    lines.append(f"# BRINGUP LOG: {state['model_id']}")
    lines.append("")
    lines.append(f"**Model:** `{state['model_id']}`")
    lines.append(f"**Slug:** `{state['model_slug']}`")
    lines.append(f"**Target Device:** {state['device']} ({state['arch_name']})")
    lines.append(f"**Started:** {state['started_at']}")
    lines.append(f"**Updated:** {state['updated_at']}")
    lines.append("")

    # --- Block Status table ---
    lines.append("## Block Status")
    lines.append("")
    lines.append("| Block | Phase | Status | PCC | Attempts | Notes |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for component in state["components"]:
        name = component.get("name", "")
        for phase in PHASE_NAMES:
            phase_dict = component.get(phase)
            if not isinstance(phase_dict, dict):
                status_cell = _EM_DASH
                pcc_cell = _EM_DASH
                attempts_cell = "0"
                notes_cell = ""
            else:
                status_cell = phase_dict.get("status") or _EM_DASH
                pcc_cell = _fmt_pcc(phase_dict.get("pcc"))
                attempts_cell = str(phase_dict.get("attempts", 0))
                notes_cell = _escape_cell(_fmt_notes(phase_dict))
            lines.append(f"| {name} | {phase} | {status_cell} | {pcc_cell} | {attempts_cell} | {notes_cell} |")
    lines.append("")

    # --- Recent Ticks ---
    lines.append("## Recent Ticks")
    lines.append("")
    tick_log = state.get("tick_log") or []
    if not tick_log:
        lines.append("_No ticks yet._")
    else:
        for entry in tick_log[-10:]:
            tick = entry.get("tick", "?")
            ts = entry.get("ts", "")
            action = _escape_cell(str(entry.get("action", "")))
            result = _escape_cell(str(entry.get("result", "")))
            if ts:
                lines.append(f"- tick {tick} ({ts}): {action} — {result}")
            else:
                lines.append(f"- tick {tick}: {action} — {result}")
    lines.append("")

    # --- Host-Resident Exceptions ---
    lines.append("## Host-Resident Exceptions")
    lines.append("")
    exceptions = []
    for component in state["components"]:
        hr = component.get("host_resident") or {}
        if hr.get("allowed") is True:
            exceptions.append(component)
    if not exceptions:
        lines.append("_None._")
    else:
        for component in exceptions:
            hr = component["host_resident"]
            name = component.get("name", "")
            justification = hr.get("justification", "") or ""
            ref = hr.get("reference_link", "") or ""
            lines.append(f"- **{name}**: {justification} (ref: {ref})")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# redo / skip: manual nudges that mutate state and append a tick_log entry
# ---------------------------------------------------------------------------


def _next_tick_number(state: dict) -> int:
    """Return the next monotonic tick number for a manual nudge entry.

    Computed as ``max(existing tick numbers, default 0) + 1`` so manual
    nudges never collide with real orchestrator tick numbers. The ``action``
    string ("redo[...]" / "skip[...]") disambiguates nudges from tick work.
    """
    return max((entry.get("tick", 0) for entry in state["tick_log"]), default=0) + 1


def _trim_tick_log(state: dict) -> None:
    """Trim tick_log to the last _MAX_TICK_LOG_ENTRIES entries.

    Called after any mutation that appends a tick_log entry. The SPEC
    bounds the on-disk log to the last N entries; full history lives
    in git via per-tick commits.
    """
    log = state["tick_log"]
    if len(log) > _MAX_TICK_LOG_ENTRIES:
        state["tick_log"] = log[-_MAX_TICK_LOG_ENTRIES:]


def _format_action(verb: str, block: str, phase: str) -> str:
    """Format a tick_log action string.

    Convention: ``"<verb>[<block>:<phase>]"``. Verb is the orchestrator
    operation (``redo``, ``skip``, ``architecture``, ``reference``, ``ttnn``,
    ``debug``, ``optimization``). Block and phase are not escaped — callers
    are expected to use canonical block/phase names that do not contain
    ``[``, ``]``, or ``:`` characters.
    """
    return f"{verb}[{block}:{phase}]"


def _find_component(state: dict, block: str) -> dict:
    """Return the component dict whose ``name`` matches ``block``.

    Linear scan — the components list is short (one entry per architectural
    block) so a dict index would be more bookkeeping than it's worth.

    Raises ``KeyError`` if no component matches.
    """
    for component in state["components"]:
        if component.get("name") == block:
            return component
    raise KeyError(f"no component named {block!r}")


def redo(state: dict, block: str, phase: str) -> dict:
    """Reset a phase cell back to pending so the orchestrator retries it.

    Mutates ``state`` in place. Sets the named phase to
    ``{"status": "pending", "attempts": 0}`` — a clean slate that drops any
    prior ``pcc``, ``last_error``, or ``notes`` from the failed attempt.
    Appends a tick_log entry recording the nudge and advances
    ``updated_at``.

    Raises:
        ValueError: if ``phase`` is not one of ``PHASE_NAMES``.
        KeyError: if no component in state matches ``block``.

    Returns:
        The same state dict (mutated), for chaining.
    """
    if phase not in PHASE_NAMES:
        raise ValueError(f"phase must be one of {PHASE_NAMES}, got {phase!r}")

    component = _find_component(state, block)

    # Clean slate — see docstring. We deliberately do NOT preserve old fields
    # so the next attempt starts from a known-empty baseline.
    component[phase] = {"status": "pending", "attempts": 0}

    now = _utc_now_iso()
    state["tick_log"].append(
        {
            "tick": _next_tick_number(state),
            "ts": now,
            "action": _format_action("redo", block, phase),
            "result": "ok",
        }
    )
    _trim_tick_log(state)
    state["updated_at"] = now
    return state


def skip(state: dict, block: str, phase: str, justify: str, reference_link: str) -> dict:
    """Mark a block as host-resident-allowed so the no-shortcuts guard passes.

    Mutates ``state`` in place. Sets ``component["host_resident"]`` to
    ``{"allowed": True, "justification": justify, "reference_link":
    reference_link}`` and sets the named phase to ``{"status": "skipped"}``
    (replacing any prior phase fields — the phase is now an explicit skip).
    Appends a tick_log entry recording the nudge and advances
    ``updated_at``.

    Raises:
        ValueError: if ``phase`` is not in ``PHASE_NAMES``, or if either
            ``justify`` or ``reference_link`` is empty. The escape hatch
            must be auditable, so empty strings are rejected.
        KeyError: if no component matches ``block``.

    Returns:
        The same state dict (mutated), for chaining.
    """
    if phase not in PHASE_NAMES:
        raise ValueError(f"phase must be one of {PHASE_NAMES}, got {phase!r}")
    if not justify:
        raise ValueError("justify must be non-empty")
    if not reference_link:
        raise ValueError("reference_link must be non-empty")

    component = _find_component(state, block)

    component["host_resident"] = {
        "allowed": True,
        "justification": justify,
        "reference_link": reference_link,
    }
    component[phase] = {"status": "skipped"}

    now = _utc_now_iso()
    state["tick_log"].append(
        {
            "tick": _next_tick_number(state),
            "ts": now,
            "action": _format_action("skip", block, phase),
            "result": f"host_resident: {justify}",
        }
    )
    _trim_tick_log(state)
    state["updated_at"] = now
    return state
