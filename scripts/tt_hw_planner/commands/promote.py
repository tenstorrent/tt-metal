from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _reset_cc_fallbacks(demo_dir: Path) -> None:
    """Clear per-component fallback list + attempt counters so a promote rerun
    gets a full attempt budget on every non-graduated component. Keeps
    last_failure_* and best_pcc so the LLM still sees prior context."""
    state_path = demo_dir / ".bringup_cc_state.json"
    if not state_path.is_file():
        return
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return
    prev_fallback = list(state.get("fallback") or [])
    prior_attempts = state.get("attempts") or {}
    if not prev_fallback and not prior_attempts:
        return
    state["fallback"] = []
    state["attempts"] = {}
    state["consecutive_same_class"] = {}
    try:
        state_path.write_text(json.dumps(state, indent=2))
    except Exception:
        return
    if prev_fallback:
        print(f"  [promote] reset {len(prev_fallback)} CPU-fallback components: {', '.join(prev_fallback)}")
    if prior_attempts:
        print(f"  [promote] cleared attempts for {len(prior_attempts)} components (fresh budget)")


def _latest_worktree_demo(model_id: str):
    """Find the newest active worktree holding this model's demo, point BRINGUP_ROOT there, and return its demo_dir (or None)."""
    from ..bringup_loop import find_demo_dir

    if os.environ.get("TT_HW_PLANNER_BRINGUP_CWD"):
        return None
    try:
        from .. import worktree as _wt

        sessions = sorted(
            (s for s in _wt.list_active() if s.model_id == model_id),
            key=lambda s: s.created_ts,
            reverse=True,
        )
    except Exception:
        return None
    for s in sessions:
        d = find_demo_dir(model_id, repo_root=s.path)
        if d is not None:
            os.environ["TT_HW_PLANNER_BRINGUP_CWD"] = str(s.path)
            return d
    return None


def cmd_promote(args) -> int:
    from .._cli_helpers.bringup_cc import _emit_stop_summary, _reset_summary

    _reset_summary()
    model_id = getattr(args, "model_id", "") or ""
    stop_reason = "promote ended"
    try:
        rc = _cmd_promote_impl(args)
        stop_reason = {
            0: "promote completed (gate can_stop)",
            1: "promote incomplete — components still not graduated (budget/attempts capped)",
            2: "promote setup failed — no scaffolded demo found / model not prepared",
        }.get(rc, f"ended with rc={rc}")
        return rc
    except Exception as exc:
        stop_reason = f"aborted by exception: {type(exc).__name__}: {exc}"
        raise
    finally:
        if model_id:
            try:
                _emit_stop_summary(model_id, stop_reason)
            except Exception:
                pass


def _cmd_promote_impl(args) -> int:
    from ..cli import (
        _API_KEY_ENV_VAR,
        _PROVIDER_LABEL,
        _auto_iteration_blockers,
        _check_agent_ready,
        _emit_and_verify_runnable_demo,
        _enforce_memory_fit_or_abort,
        _print_bringup_summary,
        _prompt_for_api_key,
        _quiet_framework_logging,
        _resolve_tiered_model_aliases,
        cmd_bringup,
        iter_loop_kwargs_from,
    )

    _quiet_framework_logging()

    MODEL = args.model_id
    BOX = args.box
    sep = "=" * 78

    def banner(title: str) -> None:
        print()
        print(sep)
        print(f"  {title}")
        print(sep)

    if getattr(args, "regen_demo_only", False):
        banner(f"REGEN-DEMO-ONLY for {MODEL}")
        ok, _ = _emit_and_verify_runnable_demo(MODEL, sep=sep)
        return 0 if ok else 1

    banner(f"PROMOTE  resume bring-up: replace CPU fallback with native TTNN for {MODEL}")

    from ..bringup_loop import find_demo_dir

    demo_dir = find_demo_dir(MODEL)
    if demo_dir is None:
        demo_dir = _latest_worktree_demo(MODEL)
        if demo_dir is not None:
            banner("PROMOTE  no demo in current tree — resuming from latest worktree")
            print(f"  worktree: {os.environ.get('TT_HW_PLANNER_BRINGUP_CWD')}")
    if demo_dir is None:
        print(
            f"ERROR: no scaffolded demo for {MODEL} (not in this tree, and no active "
            f"worktree holds it). Run bring-up first:\n"
            f"    python -m scripts.tt_hw_planner up {MODEL} --box {BOX} --execute\n",
            file=sys.stderr,
        )
        return 2

    _reset_cc_fallbacks(demo_dir)

    if (getattr(args, "engine", "cc") or "cc") == "cc":
        from .._cli_helpers.bringup_cc import run_bringup_cc

        banner(f"PROMOTE (cc engine) — harness loop on the per-component gate for {MODEL}")
        _cc_rc = run_bringup_cc(
            model_id=MODEL,
            demo_dir=demo_dir,
            agent_bin=(getattr(args, "auto_agent_bin", None) or "claude"),
            mesh=getattr(args, "mesh", None),
            max_attempts=getattr(args, "auto_max_attempts_per_component", 2),
        )
        return _cc_rc
