"""G4: mechanical action library.

A small set of category-agnostic, deterministic actions the executor
tries BEFORE invoking the LLM. Each action is a binary toggle: "apply
this change, re-run the gate, did the verdict change?" If yes, keep
the change; if no, revert.

The point is to catch trivial causes (stale cache, trace enabled,
recent broken LLM edit) without spending an LLM iteration on them.
No action contains category- or model-specific knowledge. Toggles
that don't apply to a category are cheap no-ops (the demo still runs;
the verdict just doesn't change).

Actions implemented
-------------------

* :class:`InvalidateCache`     -- delete the TT-native weight cache.
* :class:`RevertLastEdits`     -- ``git checkout -- <files>`` for the
                                  files touched in the last LLM iter.
* :class:`SetEnvVar`           -- set an env var for the next demo
                                  run (e.g. ``TT_PLANNER_NO_TRACE=1``,
                                  ``TT_TRANSFORMERS_DEBUG=1``).
* :class:`RunDeviceReset`      -- ``tt-smi -r`` (delegated to the cli's
                                  existing helper).

The executor (:mod:`.executor`) is responsible for sequencing actions
and observing the verdict after each.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


SYMPTOM_COMPILE_OR_CACHE = "compile_or_cache"
SYMPTOM_TRACE_CORRUPTION = "trace_corruption"
SYMPTOM_OP_SYNTH_ISSUE = "op_synth_issue"
SYMPTOM_NUMERICAL_PRECISION = "numerical_precision"
SYMPTOM_SAMPLING_ISSUE = "sampling_issue"
SYMPTOM_ATTENTION_PATH = "attention_path"
SYMPTOM_WEIGHT_CONVERSION = "weight_conversion"
SYMPTOM_RECENT_EDIT_DIRTY = "recent_edit_dirty"
SYMPTOM_STAGNANT_NO_SIGNAL = "stagnant_no_signal"


@dataclass
class ActionResult:
    """Outcome of an :class:`Action` execution."""

    action_name: str
    applied: bool
    notes: List[str] = field(default_factory=list)
    state_changed: bool = False
    rollback: Optional[Callable[[], None]] = None

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)


class Action:
    """Base class. Subclasses implement :meth:`apply`.

    ``symptom_classes`` declares which kinds of failures this action
    is meant to address. The executor's selector filters the action
    list by intersecting ``symptom_classes`` with the symptoms it
    inferred from the latest gate verdict. An action with an EMPTY
    ``symptom_classes`` matches any symptom (universal fallback).
    """

    name: str = "Action"
    symptom_classes: Tuple[str, ...] = ()

    def apply(self, ctx: Dict[str, Any]) -> ActionResult:
        raise NotImplementedError

    def applies_to(self, symptoms: Sequence[str]) -> bool:
        """Return True iff ANY of this action's symptom_classes is in
        ``symptoms``, or this action has no declared symptoms (matches
        all)."""
        if not self.symptom_classes:
            return True
        return any(s in symptoms for s in self.symptom_classes)


@dataclass
class InvalidateCache(Action):
    """Delete the TT-native weight cache for ``model_id``.

    Delegates to :func:`scripts.tt_hw_planner.cli._invalidate_tt_weight_cache`
    -- the existing helper that the legacy PCC-repair loop already uses.
    No new logic here; this is just the :class:`Action`-shaped adapter
    so the executor can sequence cache invalidation alongside other
    mechanical toggles.
    """

    model_id: str
    workspace_root: Path
    name: str = "InvalidateCache"

    symptom_classes: Tuple[str, ...] = (
        SYMPTOM_COMPILE_OR_CACHE,
        SYMPTOM_WEIGHT_CONVERSION,
        SYMPTOM_RECENT_EDIT_DIRTY,
        SYMPTOM_STAGNANT_NO_SIGNAL,
    )

    def apply(self, ctx: Dict[str, Any]) -> ActionResult:
        res = ActionResult(action_name=self.name, applied=False)
        try:
            from scripts.tt_hw_planner.cli import _invalidate_tt_weight_cache

            target = _invalidate_tt_weight_cache(self.model_id)
        except Exception as exc:
            res.add_note(f"cli helper failed: {type(exc).__name__}: {exc}")
            return res
        if target is None:
            res.add_note("no cache to invalidate (or deletion failed)")
            return res
        res.applied = True
        res.state_changed = True
        res.add_note(f"invalidated cache at {target}; next run rebuilds")
        return res


@dataclass
class RevertLastEdits(Action):
    """``git checkout -- <files>`` the files touched in the last LLM
    iteration. Used by the executor to bisect over a multi-file LLM
    commit and identify the minimal-helpful-subset."""

    files: List[str]
    workspace_root: Path
    name: str = "RevertLastEdits"

    symptom_classes: Tuple[str, ...] = (
        SYMPTOM_RECENT_EDIT_DIRTY,
        SYMPTOM_STAGNANT_NO_SIGNAL,
    )

    def apply(self, ctx: Dict[str, Any]) -> ActionResult:
        res = ActionResult(action_name=self.name, applied=False)
        if not self.files:
            res.add_note("no files to revert")
            return res
        try:
            subprocess.run(
                ["git", "checkout", "--", *self.files],
                cwd=str(self.workspace_root),
                capture_output=True,
                check=True,
                timeout=30,
            )
            res.applied = True
            res.state_changed = True
            res.add_note(f"reverted {len(self.files)} file(s)")
        except subprocess.CalledProcessError as exc:
            res.add_note(f"git checkout failed: {exc.stderr.decode(errors='replace')[:200]}")
        except Exception as exc:
            res.add_note(f"revert failed: {type(exc).__name__}: {exc}")
        return res


@dataclass
class SetEnvVar(Action):
    """Set an environment variable for the next demo subprocess. The
    action records the previous value (if any) and rollback restores
    it. The actual export into the demo subprocess happens in the
    executor's ``env_overrides`` dict, NOT in ``os.environ`` -- so
    parallel uses don't leak.

    ``symptom_classes`` defaults to empty (universal fallback) but
    callers should set it -- the variable's name is the strongest
    signal we have about which symptom it targets, e.g.
    ``TT_PLANNER_NO_TRACE`` -> ``trace_corruption``,
    ``TT_PLANNER_DISABLE_OP_SYNTH`` -> ``op_synth_issue``.
    """

    var: str
    value: str
    name: str = ""
    symptom_classes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"SetEnvVar({self.var}={self.value})"

        if not self.symptom_classes:
            self.symptom_classes = _infer_symptoms_for_env(self.var)

    def apply(self, ctx: Dict[str, Any]) -> ActionResult:
        env_overrides = ctx.setdefault("env_overrides", {})
        prev = env_overrides.get(self.var, None)
        env_overrides[self.var] = self.value
        res = ActionResult(
            action_name=self.name,
            applied=True,
            state_changed=True,
        )
        res.add_note(f"will export {self.var}={self.value} for next demo run")

        def _rollback() -> None:
            if prev is None:
                env_overrides.pop(self.var, None)
            else:
                env_overrides[self.var] = prev

        res.rollback = _rollback
        return res


def _infer_symptoms_for_env(var: str) -> Tuple[str, ...]:
    """Best-effort env-var-name -> symptom-class mapping. Returns an
    empty tuple for variables we don't recognise (which makes the
    action match all symptoms -- universal fallback)."""
    var_u = var.upper()
    if "NO_TRACE" in var_u or "DISABLE_TRACE" in var_u:
        return (SYMPTOM_TRACE_CORRUPTION,)
    if "OP_SYNTH" in var_u:
        return (SYMPTOM_OP_SYNTH_ISSUE,)
    if "ATTN" in var_u or "ATTENTION" in var_u or "FLEX" in var_u or "PAGED" in var_u:
        return (SYMPTOM_ATTENTION_PATH,)
    if "DI_DT" in var_u:
        return (SYMPTOM_NUMERICAL_PRECISION,)
    if "DEBUG" in var_u or "VERBOSE" in var_u:
        return ()
    return ()


@dataclass
class RunDeviceReset(Action):
    """Run ``tt-smi -r`` to reset the device. Used when the demo
    hangs or the device is in a bad state."""

    name: str = "RunDeviceReset"

    symptom_classes: Tuple[str, ...] = (SYMPTOM_STAGNANT_NO_SIGNAL,)

    def apply(self, ctx: Dict[str, Any]) -> ActionResult:
        res = ActionResult(action_name=self.name, applied=False)
        try:
            from scripts.tt_hw_planner.cli import _run_tt_smi_reset

            ok = _run_tt_smi_reset(reason="agentic.RunDeviceReset")
            res.applied = bool(ok)
            res.state_changed = bool(ok)
            res.add_note("tt-smi -r invoked" if ok else "tt-smi reset failed")
        except Exception as exc:
            res.add_note(f"reset error: {type(exc).__name__}: {exc}")
        return res


def infer_symptoms(
    *,
    verdict_result: Any = None,
    text_evidence: Any = None,
    convergence: Any = None,
    recent_edits: Optional[Sequence[str]] = None,
) -> List[str]:
    """Derive the symptom-class set from a gate verdict.

    This is the heuristic that connects the (category-agnostic)
    :class:`ValidationResult` -- which only carries scalars like
    ``mismatch_ratio`` and ``max_repeat_ratio`` -- to the symptom
    classes the action library is tagged with.

    The inference is INTENTIONALLY OVER-INCLUSIVE: it's cheaper to
    try a not-quite-matching action than to skip a matching one.
    Two strong principles:

      1. ``stagnant_no_signal`` is added whenever convergence flags
         stagnant -- this is the "do something different" signal,
         not a substantive diagnosis.
      2. Symptoms based on text-evidence shape (collapse position,
         prefix_match_count, regime_shifts) only fire when text
         evidence is present. Non-text categories rely on the
         coarser ``mismatch_ratio`` / convergence signals.
    """
    syms: List[str] = []

    if convergence is not None and getattr(convergence, "stagnant", False):
        syms.append(SYMPTOM_STAGNANT_NO_SIGNAL)

    if text_evidence is not None:
        collapse_pos = getattr(text_evidence, "collapse_position", None)
        prefix_match = getattr(text_evidence, "prefix_match_count", 0)
        regime_shifts = getattr(text_evidence, "regime_shifts", None) or []
        kinds = {getattr(r, "kind", "") for r in regime_shifts}

        if collapse_pos is not None and prefix_match >= 5:
            syms.append(SYMPTOM_NUMERICAL_PRECISION)
            if "repetition" in kinds:
                syms.append(SYMPTOM_SAMPLING_ISSUE)

        elif prefix_match == 0 and collapse_pos is not None:
            syms.append(SYMPTOM_WEIGHT_CONVERSION)
            syms.append(SYMPTOM_ATTENTION_PATH)

    if verdict_result is not None:
        mr = float(getattr(verdict_result, "mismatch_ratio", 0.0))
        rr = float(getattr(verdict_result, "max_repeat_ratio", 0.0))
        if mr >= 0.95:
            syms.append(SYMPTOM_WEIGHT_CONVERSION)
            syms.append(SYMPTOM_COMPILE_OR_CACHE)
        if rr >= 0.5:
            syms.append(SYMPTOM_SAMPLING_ISSUE)
            syms.append(SYMPTOM_NUMERICAL_PRECISION)

    if recent_edits and convergence is not None and getattr(convergence, "stagnant", False):
        syms.append(SYMPTOM_RECENT_EDIT_DIRTY)

    return syms


def filter_actions_by_symptoms(
    actions: Sequence[Action],
    symptoms: Sequence[str],
) -> List[Action]:
    """Return the subset of ``actions`` whose ``symptom_classes``
    intersects ``symptoms``, preserving order. Actions with empty
    ``symptom_classes`` are universal and always pass through.
    """
    return [a for a in actions if a.applies_to(symptoms)]


def default_mechanical_actions(
    *,
    model_id: str,
    workspace_root: Path,
    recent_edits: Optional[List[str]] = None,
) -> List[Action]:
    """Return the default-order list of cheap toggles tried before
    LLM. Order matters: cheapest / most-likely-to-be-the-bug first.

    The toggles are model-agnostic. Many will be no-ops on most
    models (e.g. disabling trace doesn't help if the model never
    used trace); that's fine.
    """
    actions: List[Action] = []

    # AUDIT FIX #6 (2026-05-31): wire RevertLastEdits when recent_edits
    # is provided. The action existed in actions.py but was never added
    # to the default list — so executor never picked it. This is the
    # "build broke from LLM edit" recovery path that the per-component
    # iterate loop relies on after a failed build iteration.
    if recent_edits:
        actions.append(RevertLastEdits(files=list(recent_edits), workspace_root=workspace_root))

    actions.append(InvalidateCache(model_id=model_id, workspace_root=workspace_root))

    actions.append(SetEnvVar("TT_PLANNER_NO_TRACE", "1"))

    actions.append(SetEnvVar("TT_PLANNER_DISABLE_OP_SYNTH", "1"))

    actions.append(SetEnvVar("ATTN_IMPLEMENTATION", "eager"))

    actions.append(SetEnvVar("TT_METAL_DISABLE_DI_DT_WORKAROUND", "0"))

    return actions


__all__ = [
    "Action",
    "ActionResult",
    "InvalidateCache",
    "RevertLastEdits",
    "RunDeviceReset",
    "SetEnvVar",
    "default_mechanical_actions",
    "SYMPTOM_COMPILE_OR_CACHE",
    "SYMPTOM_TRACE_CORRUPTION",
    "SYMPTOM_OP_SYNTH_ISSUE",
    "SYMPTOM_NUMERICAL_PRECISION",
    "SYMPTOM_SAMPLING_ISSUE",
    "SYMPTOM_ATTENTION_PATH",
    "SYMPTOM_WEIGHT_CONVERSION",
    "SYMPTOM_RECENT_EDIT_DIRTY",
    "SYMPTOM_STAGNANT_NO_SIGNAL",
    "infer_symptoms",
    "filter_actions_by_symptoms",
]
