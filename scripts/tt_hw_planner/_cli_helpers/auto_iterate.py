from __future__ import annotations
from ..discovery import safe_relative_to_root, BRINGUP_ROOT

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _brain_sync_graduated_to_main_tree(
    *,
    MODEL: str,
    demo_dir: Path,
    graduated_this_run: List[str],
    banner_fn,
) -> None:
    """Brain G8 worktree → main-tree sync. Wraps the
    agentic.persistence primitive with the local-context-handling
    (demo subpath, _safe_id) so the call site is a 1-liner.

    Safe to call from ANY success-exit point in the loop. No-ops
    cleanly when not in a worktree, or when graduated_this_run is
    empty. Errors are non-fatal.
    """
    if not graduated_this_run:
        return
    try:
        from ..agentic.persistence import sync_graduated_to_main_tree
        from ..bringup_loop import _safe_id

        # B2-FIX: compute the demo path RELATIVE TO the worktree root
        # directly. `safe_relative_to_root` returns the absolute path
        # on miss (when demo_dir doesn't sit under BRINGUP_ROOT), which
        # would then combine via `worktree_root / abs_path` and
        # silently target the absolute path, not the main tree.
        worktree_root = Path.cwd()
        try:
            demo_subpath = demo_dir.resolve().relative_to(worktree_root.resolve())
        except ValueError:
            # demo_dir is outside the worktree — caller can't meaningfully
            # sync to main tree.
            print(f"  [brain G8 sync] demo_dir ({demo_dir}) is outside " f"worktree ({worktree_root}); skipping sync")
            return
        result = sync_graduated_to_main_tree(
            worktree_root=worktree_root,
            demo_subpath=str(demo_subpath),
            graduated_components=graduated_this_run,
            safe_id_fn=_safe_id,
        )
        if result.synced:
            banner_fn(
                f"SYNC (brain G8): wrote {len(result.synced)} graduated "
                f"stub(s) to main tree → {result.main_tree_path}"
            )
            for note in result.notes:
                print(f"    {note}")
        elif result.notes:
            for note in result.notes:
                print(f"  [brain G8 sync] {note}")
    except Exception as exc:
        print(
            f"  [brain G8] worktree sync non-fatal: " f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _brain_handle_phantom_failures(
    *,
    MODEL: str,
    demo_dir: Path,
    final_failed: List[str],
    banner_fn,
    allow_kill_stale: bool,
    allow_device_reset: bool,
) -> Optional[Dict[str, Any]]:
    """Brain G8 phantom-failure handler.

    For each failed component, ask the brain whether the failure is a
    stale-decomposed-parent test (the parent was split into children
    that carry the real work; the parent's old standalone test is a
    phantom). If yes, archive the file. If any phantoms were archived,
    re-run pytest and return the updated ``{"rc": int, "report": dict}``.
    Returns ``None`` when nothing was archived (caller keeps current state).

    Called from BOTH the early-exit path (everything at cap → final
    pytest → if any failure is a phantom, clean up) AND the
    fall-through path (loop ended normally → final pytest → same
    check). Without dual call sites, runs that exit early skip the
    brain check entirely — exactly the SAM2 2026-05-30 symptom.
    """
    if not final_failed:
        return None
    try:
        from ..agentic.stale_tests import archive_stale_test, detect_stale_decomposed_test
        from ..bringup_loop import _safe_id
        from ..cli import _list_component_pcc_tests, _parse_pytest_report, _run_focused_pytest, _scope_report_to_demo
        from ..overlay_manager import load_no_emit_tests

        no_emit = load_no_emit_tests(MODEL)
        archived_phantoms: List[str] = []
        for failed_comp in final_failed:
            verdict = detect_stale_decomposed_test(
                component=failed_comp,
                no_emit_tests=no_emit,
            )
            if verdict.is_stale and verdict.action == "archive_test":
                archived_path = archive_stale_test(
                    demo_dir=demo_dir,
                    component=failed_comp,
                    safe_id=_safe_id(failed_comp),
                )
                if archived_path is not None:
                    archived_phantoms.append(failed_comp)
                    print(f"  [brain G8] phantom failure detected: {verdict.reason}")
                    print(f"    archived: {archived_path.name}")
        if not archived_phantoms:
            return None
        banner_fn(
            f"PHANTOM-CLEANUP (brain G8): archived {len(archived_phantoms)} "
            f"stale decomposed-parent test(s) — re-running final pytest"
        )
        final_tests = _list_component_pcc_tests(demo_dir)
        if not final_tests:
            # G1-FIX: every PCC test was archived (all were phantoms).
            # The pre-archive `final_failed` is stale — those failures
            # were on tests that no longer exist. Synthesize a clean
            # success report so the caller stops trusting stale state.
            print(
                "  [brain G8] all remaining PCC tests were phantoms — " "nothing left to re-run; reporting clean state"
            )
            return {
                "rc": 0,
                "report": {
                    "all_passed": True,
                    "passed_components": [],
                    "failed_components": [],
                    "skipped_components": [],
                },
            }
        rc = _run_focused_pytest(
            model_id=MODEL,
            test_files=final_tests,
            allow_kill_stale=allow_kill_stale,
            allow_device_reset=allow_device_reset,
        )
        report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
        return {"rc": rc, "report": report}
    except Exception as exc:
        print(
            f"  [brain G8] stale-test detection non-fatal: " f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def _run_auto_iterate_loop(
    *,
    MODEL: str,
    BOX: str,
    mesh: Optional[str],
    dtype: Optional[str],
    batch: int,
    max_seq_len: int,
    max_generated_tokens: int,
    accuracy: bool,
    no_trace: bool,
    no_paged_attention: bool,
    no_instruct: bool,
    download_first: bool,
    strict: bool,
    demo_dir: Path,
    provider: str,
    agent_bin: str,
    model: str,
    max_iters: int,
    sep: str,
    target_components: Optional[List[str]] = None,
    strict_native: bool = False,
    agent_timeout_s: int = 600,
    allow_kill_stale: bool = True,
    allow_device_reset: bool = True,
    max_attempts_per_component: int = 2,
    allow_partial_cpu: bool = False,
    model_light: Optional[str] = None,
    model_heavy: Optional[str] = None,
    parallel_agents: int = 1,
    only_component: Optional[str] = None,
) -> int:
    from ..cli import (
        REPO_ROOT,
        _LAST_PYTEST_STAGES,
        _TORCH_WRAPPER_PATTERNS,
        _agent_complexity_timeout,
        _auto_iteration_blockers,
        _build_cross_component_context_block,
        _classify_components,
        _classify_failure,
        _clear_responses_dir,
        _component_metadata,
        _component_stub_path,
        _detect_no_hardware_failure,
        _emit_and_verify_runnable_demo,
        _exemplar_block,
        _extract_pcc_from_failure,
        _extract_shape_probes_from_report,
        _failure_class_severity,
        _failure_signature,
        _find_exemplar,
        _find_handoff_path,
        _format_agentic_affordances_block,
        _format_attempt_history_block,
        _format_compute_split,
        _format_escalated_edit_scope_block,
        _format_failure_block_for_component,
        _format_no_hardware_diagnostic_banner,
        _format_op_split,
        _format_shape_probe_block,
        _full_hf_reference_source,
        _invoke_agent,
        _list_component_pcc_tests,
        _load_attempt_history,
        _native_directive,
        _format_captured_shape_contract_block,
        _numerical_constraints_block,
        _only_pcc_threshold_failures,
        _op_synth_manifest,
        _op_synth_prompt_blocks,
        _parse_pytest_report,
        _partial_cpu_components,
        _pick_agent_model_for_iter,
        _print_bringup_summary,
        _read_file_excerpt,
        _read_test_source,
        _rewrite_components_to_stable_fallback,
        _run_focused_pytest,
        _run_tt_smi_reset,
        _runtime_fallback_details,
        _safe_id,
        _scope_report_to_demo,
        _strategy_directive_for_failure,
        _stub_forward_body_excerpt,
        _stub_has_graduated_from_autofill,
        _stub_source_excerpt,
        _stub_uses_torch_wrapper,
        _torch_ref_summary,
        _ungraduated_breakdown,
        _write_attempt_log,
        cmd_prepare,
        find_demo_dir,
        hashlib,
        run_bringup_loop,
    )

    def banner(title: str) -> None:
        print()
        print(sep)
        print(f"  {title}")
        print(sep)

    last_failures = ""
    last_failure_details = ""
    last_failed_components: List[str] = []
    last_failed_tests: List[str] = []

    last_shape_probes: List[Dict[str, str]] = []
    seed_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
    repeat_error_counts: Dict[str, int] = {}
    consecutive_empty_agent_calls = 0
    max_consecutive_empty = 3
    attempts_per_component: Dict[str, int] = {}

    consecutive_same_class_attempts: Dict[str, int] = {}
    last_failure_class_per_component: Dict[str, str] = {}
    last_pcc_per_component: Dict[str, Optional[float]] = {}

    best_pcc_per_component: Dict[str, float] = {}

    # Per-component PCC history for the agentic-engine plateau detector
    # (G8 convergence). Appended on every PCC-fail iter; consumed by
    # is_stagnant() to decide whether to apply a mechanical action
    # (G4) before the next LLM call.
    pcc_history_per_component: Dict[str, List[float]] = {}

    # Per-component set of G4 mechanical actions already tried this
    # run. We try each action at most once per component to avoid
    # re-applying the same toggle on repeated plateaus.
    tried_actions_per_component: Dict[str, set] = {}

    # Per-component count of brain-granted cap extensions this run.
    # The brain (agentic.convergence.should_extend_component_cap) uses
    # this to enforce its own per-component cap-on-cap-extensions.
    cap_extensions_used_per_component: Dict[str, int] = {}

    # Per-component set tracking which components have already had
    # auto-decompose attempted this run. Prevents re-running the
    # expensive decompose subprocess on every iter for the same
    # parent component.
    decomposition_auto_attempted: set = set()

    hard_total_attempt_cap: int = max(3, max_attempts_per_component * 2)
    permanently_skipped: List[str] = []
    graduated_this_run: List[str] = []

    from .sweep_cache import ValidationSweepCache
    from ..overlay_manager import load_persistent_skips, load_no_emit_tests, persist_skip

    _SWEEP_CACHE = ValidationSweepCache()

    # Apply any pending decomposition plan BEFORE loading the component
    # list. cmd_decompose --write-plan writes the plan; this consumer
    # picks it up on the next run, adds children as NEW components to
    # bringup_status.json, and marks the parent no_emit. Idempotent —
    # already-applied plans are archived and not re-applied.
    try:
        from ..decomposition_consumer import consume_decomposition_plan

        _decomp_added, _decomp_notes = consume_decomposition_plan(model_id=MODEL, demo_dir=demo_dir)
        for _line in _decomp_notes:
            print(f"  {_line}")
        if _decomp_added:
            banner(f"DECOMPOSITION CONSUMER: added {_decomp_added} child component(s) " f"from decomposition_plan.json")
    except Exception as _decomp_exc:
        print(
            f"  [decomposition-consume] non-fatal error: {type(_decomp_exc).__name__}: {_decomp_exc}", file=sys.stderr
        )

    _persistent_skips = load_persistent_skips(MODEL)
    if _persistent_skips:
        # Only KERNEL_MISSING entries are persisted now. They stay
        # permanently excluded until the TTNN op lands (cleared via
        # `overlay-clear-skips`). Legacy entries with other categories
        # are treated as retryable — they're tooling-debt verdicts from
        # an older tool version and should be re-attempted on this run.
        _permanent_skips = {
            name: entry
            for name, entry in _persistent_skips.items()
            if (entry.get("category") or "").upper() == "KERNEL_MISSING"
        }
        _retryable_skips = {name: entry for name, entry in _persistent_skips.items() if name not in _permanent_skips}
        permanently_skipped.extend(sorted(_permanent_skips.keys()))
        print(
            f"  [persistent-skips] loaded {len(_permanent_skips)} kernel-missing skip(s) "
            f"from prior runs: {', '.join(sorted(_permanent_skips.keys())) or '(none)'}"
        )
        if _retryable_skips:
            _retry_desc = ", ".join(
                f"{name}={(entry.get('category') or '?')}" for name, entry in sorted(_retryable_skips.items())
            )
            print(
                f"  [persistent-skips] {len(_retryable_skips)} legacy non-kernel-missing skip(s) "
                f"will be re-attempted this run ({_retry_desc})"
            )
        print(
            f"  [persistent-skips] to clear kernel-missing skips and re-attempt them, run: "
            f"`python -m scripts.tt_hw_planner overlay-clear-skips {MODEL}`"
        )

    _no_emit_tests = load_no_emit_tests(MODEL)
    if _no_emit_tests:
        _new_no_emit = sorted(c for c in _no_emit_tests.keys() if c not in permanently_skipped)
        permanently_skipped.extend(_new_no_emit)
        print(
            f"  [no-emit-tests] loaded {len(_no_emit_tests)} component(s) flagged as "
            f"structurally untestable (Phase 2 ModuleList drops): "
            f"{', '.join(sorted(_no_emit_tests.keys()))}"
        )
        if _new_no_emit:
            print(f"  [no-emit-tests] excluding {len(_new_no_emit)} of these from the " f"candidate pool for this run")

    unverified_native_this_run: set = set()
    verified_fail: set = set()

    validated_this_run: set = set(seed_report.get("passed_components", []) or [])

    skipped_components_this_run: set = set(seed_report.get("skipped_components", []) or [])

    _seed_harness_markers = (
        "HF reference forward",
        "_make_arg_for()",
        "synthetic inputs from _make_arg_for",
        "incompatible with this submodule's expected shapes",
        "the synthetic inputs",
    )
    _seed_per_skipped = seed_report.get("per_skipped", {}) if isinstance(seed_report, dict) else {}
    if isinstance(_seed_per_skipped, dict):
        _seed_harness_components: Dict[str, List[str]] = {}
        for entry in _seed_per_skipped.values():
            if not isinstance(entry, dict):
                continue
            comp = str(entry.get("component") or "").strip()
            if not comp:
                continue
            reason = str(entry.get("reason") or "").strip()
            if not reason:
                continue
            if any(mark in reason for mark in _seed_harness_markers):
                _seed_harness_components.setdefault(comp, []).append(reason)
        for comp, reasons in _seed_harness_components.items():
            stub_path = demo_dir / "_stubs" / f"{_safe_id(comp)}.py"
            try:
                is_native = _stub_has_graduated_from_autofill(stub_path)
            except Exception:
                is_native = False
            if not is_native:
                continue
            if comp in graduated_this_run or comp in unverified_native_this_run or comp in permanently_skipped:
                continue
            unverified_native_this_run.add(comp)
            skipped_components_this_run.discard(comp)
            verified_fail.discard(comp)
            reason_blob = "; ".join(reasons) or "(no skip reason captured)"
            # Seed-phase UNVERIFIED NATIVE = harness can't run PCC on
            # this stub (scaffolder bug). Classify as TOOL_BUG so it's
            # not silently bucketed with workload-COLD components.
            persist_skip(MODEL, comp, reason_blob, category="TOOL_BUG")
            print(
                f"  seed-phase: `{comp}` UNVERIFIED NATIVE — stub is "
                f"native ttnn but PCC could not be measured because the "
                f"auto-generated _make_arg_for() inputs are shape-"
                f"incompatible with this HF submodule. NOT counted as "
                f"graduated (no PCC proof). NOT retried (test scaffold "
                f"is the bug). Bring-up complete will be blocked until "
                f"the per-component PCC test is hand-fixed. "
                f"Reason: {reason_blob}"
            )
    try:
        env_cap = int(os.environ.get("TT_PLANNER_MAX_ATTEMPTS_PER_COMPONENT", "") or 0)
        if env_cap > 0:
            max_attempts_per_component = env_cap
    except Exception:
        pass
    if max_attempts_per_component < 1:
        max_attempts_per_component = 1

    hard_total_attempt_cap = max(3, max_attempts_per_component * 2)
    try:
        env_hard = int(os.environ.get("TT_PLANNER_HARD_TOTAL_ATTEMPT_CAP", "") or 0)
        if env_hard > 0:
            hard_total_attempt_cap = max(hard_total_attempt_cap, env_hard)
    except Exception:
        pass

    PCC_STUCK_THRESHOLD = 0.5
    PCC_STUCK_EXTRA_ATTEMPTS = 2

    def _component_complexity_bonus(comp: str) -> int:
        """Return the EXTRA consecutive-same-class attempts a component
        gets due to its op-level complexity. Range: 0..4 inclusive.

        Returns 0 for components without an op-synth manifest (we have
        no reliable signal). Returns positive bonuses for complex
        components based on palette size and llm_gaps count."""
        manifest = _op_synth_manifest(demo_dir, comp)
        if not manifest:
            return 0
        palette = manifest.get("palette") or []
        gaps = manifest.get("llm_gaps") or []
        palette_size = len(palette) if isinstance(palette, list) else 0
        gaps_size = len(gaps) if isinstance(gaps, list) else 0

        if palette_size <= 10:
            palette_bonus = 0
        elif palette_size <= 30:
            palette_bonus = 1
        elif palette_size <= 80:
            palette_bonus = 2
        else:
            palette_bonus = 3

        gap_bonus = min(2, gaps_size)
        return min(4, palette_bonus + gap_bonus)

    def _effective_attempt_cap(comp: str) -> int:
        """Per-component cap on the consecutive-same-class counter.

        Three additive sources of cap:
          1. `max_attempts_per_component` (base, e.g. 2)
          2. Capability-5 complexity bonus from op-synth manifest
             (0..4 extra attempts for big/novel components)
          3. Tier-2 PCC-stuck bonus (+`PCC_STUCK_EXTRA_ATTEMPTS` when
             last failure was PCC_ONLY with PCC >= 0.5 — structurally
             correct, numerically close)

        The total is clamped at `hard_total_attempt_cap` so the
        absolute safety ceiling is never breached.
        """
        base = max_attempts_per_component
        complexity = _component_complexity_bonus(comp)
        base = base + complexity
        last_class = last_failure_class_per_component.get(comp, "")
        last_pcc = last_pcc_per_component.get(comp)
        if last_class == "PCC_ONLY" and last_pcc is not None and last_pcc >= PCC_STUCK_THRESHOLD:
            return min(base + PCC_STUCK_EXTRA_ATTEMPTS, hard_total_attempt_cap)

        return min(base, hard_total_attempt_cap)

    def _is_at_cap(comp: str) -> bool:
        """Combined cap check used at every target-pick / fallback-on-exhaustion
        site. A component is "at cap" when EITHER:
          - it has accumulated `_effective_attempt_cap(comp)` consecutive
            failures with the SAME failure class AND no PCC improvement
            (i.e. the LLM is no longer making forward progress on it), OR
          - the absolute total attempts have hit `hard_total_attempt_cap`
            (safety net against runaway loops where the failure class
            oscillates).

        Pre-Tier-1#2 behavior was "total >= max_attempts_per_component"
        (the consecutive-same-class counter equals total attempts when no
        progress is ever detected, so the Tier-1#2 rule was a strict
        relaxation). Tier-2 C further relaxes the per-component cap for
        the structural-but-stuck PCC regime (PCC >= 0.5) — see
        `_effective_attempt_cap` — while keeping the hard ceiling intact.
        """
        if attempts_per_component.get(comp, 0) >= hard_total_attempt_cap:
            return True
        return consecutive_same_class_attempts.get(comp, 0) >= _effective_attempt_cap(comp)

    def _attempts_display(comp: str) -> str:
        """Return the "K/N (total=T)" string used in user-facing log lines.

        N reflects the effective cap (so the "high-PCC stuck" relaxation
        is visible to the user immediately, rather than appearing as the
        component "exceeding its cap" by 2 attempts)."""
        total = attempts_per_component.get(comp, 0)
        consec = consecutive_same_class_attempts.get(comp, 0)
        cap = _effective_attempt_cap(comp)
        if total == consec:
            return f"{consec}/{cap}"
        return f"{consec}/{cap} (total={total}/{hard_total_attempt_cap})"

    def _record_failure_for_component(
        comp: str,
        failure_class: str,
        pcc_value: Optional[float],
    ) -> bool:
        """Update the per-component progress trackers based on the latest
        pytest failure observation. Returns True if this iteration represents
        STRONG progress, in which case the consecutive-same-class counter
        resets to 1; False otherwise, counter increments by 1.

        STRONG progress (2026-05-22 23:00 UTC tightening — was previously
        "any class change OR PCC improvement"; that rule let the LLM cycle
        SHAPE -> CRASH -> SHAPE indefinitely because each step "shifted
        class", resetting the counter and bypassing the per-component cap):
          - First failure observed for this component (no prev class), OR
          - failure_class IMPROVED (lower severity than prev), OR
          - SAME class is PCC_ONLY AND PCC strictly improved by > 0.001.

        Regressions (worse severity) and SAME-class non-PCC failures count
        as NO progress (counter increments). This is the rule that makes
        `consecutive_same_class_attempts` actually bound LLM thrash.

        Environmental failure classes (DEVICE_NEEDS_RESET) are ignored —
        they reflect host/device hygiene, not code state, and the retry
        will run identical code, so they should neither count as progress
        nor as a same-class repeat.
        """
        if failure_class == "DEVICE_NEEDS_RESET":
            return False

        if attempts_per_component.get(comp, 0) == 0:
            last_failure_class_per_component[comp] = failure_class
            last_pcc_per_component[comp] = pcc_value
            return False
        prev_class = last_failure_class_per_component.get(comp, "")
        prev_pcc = last_pcc_per_component.get(comp)
        is_progress = False
        if not prev_class:
            is_progress = True
        elif (
            failure_class != prev_class
            and failure_class not in ("OTHER", "UNKNOWN")
            and prev_class not in ("OTHER", "UNKNOWN")
            and _failure_class_severity(failure_class) < _failure_class_severity(prev_class)
        ):
            is_progress = True
        elif (
            failure_class == "PCC_ONLY"
            and prev_class == "PCC_ONLY"
            and pcc_value is not None
            and prev_pcc is not None
            and pcc_value > prev_pcc + 0.001
        ):
            is_progress = True
        if is_progress:
            consecutive_same_class_attempts[comp] = 1
        else:
            consecutive_same_class_attempts[comp] = consecutive_same_class_attempts.get(comp, 0) + 1
        last_failure_class_per_component[comp] = failure_class
        last_pcc_per_component[comp] = pcc_value
        # G8 plateau-detector feed: record EVERY PCC observation in a
        # per-component history. mismatch_ratio = 1.0 - pcc, so the
        # is_stagnant() heuristic interprets a stagnant history as
        # "no convergence delta over the last window."
        if pcc_value is not None:
            pcc_history_per_component.setdefault(comp, []).append(1.0 - float(pcc_value))

        return is_progress

    systemic_failure_counts: Dict[tuple, set] = {}
    systemic_pattern_seen: set = set()
    SYSTEMIC_THRESHOLD = 3

    def _record_systemic_failure(
        comp: str,
        failure_class: str,
        signature: str,
    ) -> Optional[tuple]:
        """Record a per-iter failure observation in the systemic-pattern
        tracker. Returns the (failure_class, signature) key when this
        is the first iter at which the systemic threshold has been
        crossed for that key (i.e. >=`SYSTEMIC_THRESHOLD` DISTINCT
        components have now failed with the same signature). Returns
        None otherwise.

        The caller is responsible for actually emitting the SYSTEMIC
        PATTERN banner + adjusting the next iter's prompt — this
        helper only tracks state and reports the threshold-crossing
        event ONCE per key per session."""
        if failure_class in ("DEVICE_NEEDS_RESET", ""):
            return None
        if not signature:
            return None
        key = (failure_class, signature)
        bucket = systemic_failure_counts.setdefault(key, set())
        bucket.add(comp)
        if len(bucket) >= SYSTEMIC_THRESHOLD and key not in systemic_pattern_seen:
            systemic_pattern_seen.add(key)
            return key
        return None

    def _format_systemic_pattern_block(failure_class: str, signature: str) -> str:
        """Render the SYSTEMIC PATTERN section for the LLM prompt when
        the threshold has been crossed for (failure_class, signature).
        Lists the affected components and asks the LLM to investigate
        the harness instead of rewriting more stubs."""
        key = (failure_class, signature)
        comps = sorted(systemic_failure_counts.get(key, set()))
        if not comps:
            return ""
        lines = [
            "SYSTEMIC PATTERN DETECTED (PRIORITY: read this first):",
            f"  {len(comps)} distinct components have now failed with the SAME",
            f"  failure class `{failure_class}` and the SAME signature:",
            f"  ",
            f"    components: {', '.join(comps)}",
            f"    signature : {signature[:200]}{'...' if len(signature) > 200 else ''}",
            f"  ",
            "  When N>=3 distinct components fail with byte-identical",
            "  failure signatures, the bug is almost never in the stubs",
            "  individually — it is in a path SHARED by all of them.",
            "  Likely suspects, in order of frequency:",
            "    1. The test harness's `_ttnn_to_torch_mesh_safe` (or",
            "       similar) drain helper. Look in `tests/pcc/conftest.py`",
            "       and the per-test PCC files. Check it handles dict /",
            "       tuple / list outputs and calls",
            "       `ttnn.synchronize_device(device)` before drain.",
            "    2. The device fixture configuration",
            "       (`l1_small_size`, `dispatch_core_type`, mesh layout).",
            "    3. A shared op-synth helper template applied across many",
            "       components (look under `_op_helpers/` or",
            "       `_stubs/_shared.py`).",
            "  ",
            "  Action for this iter: BEFORE writing any per-component code,",
            "  use Read+Grep tools to locate the shared path that produces",
            "  this signature, edit the SHARED file to fix it, and only",
            "  then move on to any remaining per-component cleanup. You",
            "  are explicitly allowed to edit harness/conftest/op_helper",
            "  files for this iteration.",
            "",
        ]
        return "\n".join(lines)

    def _snapshot_native_stub(comp: str) -> None:
        """Snapshot a freshly-graduated component's stub so we can roll back
        to it if a future iteration silently regresses it."""
        safe = _safe_id(comp)
        stub_path = demo_dir / "_stubs" / f"{safe}.py"
        snap_path = stub_path.with_suffix(".py.last_good_native")
        if not stub_path.is_file():
            return
        try:
            snap_path.write_text(stub_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as exc:
            print(
                f"  could not snapshot native stub for `{comp}` to " f"{snap_path.name}: {exc}",
                file=sys.stderr,
            )

    def _snapshot_preiter_native_stub(comp: str) -> Optional[Path]:
        """Snapshot the PRE-ITER state of an already-native stub before the
        LLM is invoked on it.

        Fixes a real bug observed in the 2026-05-22 sam2-hiera-small run:
        `decoder_head` entered the session with an existing native TTNN port
        (scored PCC=0.671 — imperfect but exercises the device path). It had
        no `.last_good_native` snapshot because that snapshot is taken only
        on a CPU->native graduation event, which never happened during this
        session. When the LLM failed to fix the PCC bug across 2 attempts
        and the cap-rule fired, `_skip_component_to_fallback` restored from
        `.py.bak` — which was the stale op-synth torch-wrapper template
        from the bring-up scaffold. That silently rolled back the native
        code, throwing away whatever worked.

        Net effect of NOT having this snapshot: cap-out on an
        already-native component makes things STRICTLY WORSE than the
        session start. With this snapshot in place, the worst-case
        cap-out path is "rollback to the native code we came in with",
        which is the right behavior.

        Returns the snapshot Path on success, None if no snapshot was
        taken (stub missing, stub is torch-wrapper, or a snapshot already
        exists)."""
        safe = _safe_id(comp)
        stub_path = demo_dir / "_stubs" / f"{safe}.py"
        snap_path = stub_path.with_suffix(".py.preiter_native")
        if not stub_path.is_file():
            return None

        if _stub_uses_torch_wrapper(stub_path):
            return None

        if snap_path.is_file():
            return snap_path
        try:
            snap_path.write_text(stub_path.read_text(encoding="utf-8"), encoding="utf-8")
            return snap_path
        except Exception as exc:
            print(
                f"  could not snapshot pre-iter native stub for `{comp}` " f"to {snap_path.name}: {exc}",
                file=sys.stderr,
            )
            return None

    def _restore_native_snapshot(comp: str) -> bool:
        """Restore a previously-snapshotted native stub. Returns True if the
        restore succeeded."""
        safe = _safe_id(comp)
        stub_path = demo_dir / "_stubs" / f"{safe}.py"
        snap_path = stub_path.with_suffix(".py.last_good_native")
        if not snap_path.is_file():
            return False
        try:
            stub_path.write_text(snap_path.read_text(encoding="utf-8"), encoding="utf-8")
            return True
        except Exception as exc:
            print(
                f"  could not restore native snapshot for `{comp}` " f"from {snap_path.name}: {exc}",
                file=sys.stderr,
            )
            return False

    def _snapshot_best_native_stub(comp: str, pcc: Optional[float]) -> None:
        """Persist the highest-PCC native stub body we've seen for this
        component during the current session, to `.py.best_native`.

        Captures BOTH cases the audit identified:

        1. An already-native component (preiter snapshot exists at e.g.
           PCC=0.67) is improved by the LLM to PCC=0.88 but still fails.
           Pre-fix, cap-out restored preiter (0.67), throwing away the
           in-session improvement. With `.best_native` (now 0.88), cap-
           out restores the better version.

        2. An op-synth component starts as a torch-wrapper `__call__`
           (so no preiter snapshot is taken), the agent replaces the
           body with pure ttnn ops, PCC fails. Pre-fix, cap-out fell
           through to `.py.bak` (the autofill torch wrapper),
           discarding the new native `__call__`. With `.best_native`,
           cap-out restores the structural-native body the agent wrote,
           which the next attempt can iterate on.

        Snapshot rules:
          - SKIP if stub is torch-wrapper (no native code to save).
          - WRITE if no prior `.best_native` exists yet (any native
            body is better than `.bak`'s torch wrapper or no snapshot).
          - WRITE if PCC strictly improves over our recorded best.
          - SKIP if PCC is None (we have no quality signal, and we
            don't want to overwrite a known-good 0.88 with an
            unmeasured stub from a different iter).
        """
        safe = _safe_id(comp)
        stub_path = demo_dir / "_stubs" / f"{safe}.py"
        if not stub_path.is_file():
            return
        if _stub_uses_torch_wrapper(stub_path):
            return
        snap_path = stub_path.with_suffix(".py.best_native")
        prior = best_pcc_per_component.get(comp)
        if snap_path.is_file() and prior is not None and pcc is not None and pcc <= prior:
            return
        try:
            snap_path.write_text(stub_path.read_text(encoding="utf-8"), encoding="utf-8")
            if pcc is not None:
                if prior is None or pcc > prior:
                    best_pcc_per_component[comp] = pcc
        except Exception as exc:
            print(
                f"  could not snapshot best-native stub for `{comp}` " f"to {snap_path.name}: {exc}",
                file=sys.stderr,
            )

    def _skip_component_to_fallback(comp: str, reason: str) -> None:
        """Mark a component as permanently-skipped for this run: roll the
        stub back to the best known floor and record the skip so the loop
        won't target it again.

        WIRING #10 NOTE (2026-05-31): The PATH-AGNOSTIC CORE of this
        function (classify failure + persist KERNEL_MISSING if
        warranted) is consolidated in
        ``failure_classifier.classify_and_persist_skip``;
        ``_cli_helpers/runtime_repair.py`` exposes
        ``_classify_and_persist_skip`` as a thin shim around it.
        (``_cli_helpers/pcc_repair.py`` was deleted 2026-05-31 — the
        whole-model retry loop was duplication of Path 1's
        per-component iterate flow; Path 2 now escalates here via
        ``_maybe_escalate_pcc_fail`` in cli.py.) The Path-1-specific
        parts of THIS function (stub rollback from
        .last_good_native/.best_native/.preiter_native/.bak
        snapshots, permanently_skipped list update, decomposition
        auto-spawn) remain here.

        BUG/LAYER-7: If the agent's failure trace matches a kernel-missing
        signal, ALSO persist to missing_kernels.json so the final
        categorization can put this component in KERNEL_MISSING (allowed
        past the gate) instead of HOT_STUCK (blocks demo emission). This
        distinguishes "TTNN dev work needed" from "tool effort gap".

        Restoration priority (highest to lowest):
        1. `.py.last_good_native` — captured at the most recent
           successful graduation (PCC test passed). Strongest possible
           signal that we have a working native body.
        2. `.py.best_native` — captured during the session whenever the
           LLM produced a native body with a new best PCC (even if it
           did not reach the 0.99 graduation threshold). Closes the
           "in-session improvement is lost on cap-out" gap (e.g. PCC
           climbed 0.67 -> 0.88 but still failed; pre-fix we threw
           0.88 away and restored 0.67).
        3. `.py.preiter_native` — native code we entered the session
           with. Used when no in-session improvement was seen.
        4. `.py.bak` — the original bring-up torch-wrapper template
           (CPU fallback). Used when we never had native code at all.
        5. Synthesized stable CPU-fallback stub — final safety net if
           all snapshots are missing."""
        if comp in permanently_skipped:
            return
        permanently_skipped.append(comp)
        verified_fail.discard(comp)
        # S2-FIX: also drop from graduated_this_run so a regressed
        # component doesn't continue to inflate the brain's "momentum"
        # proxy (`should_extend_budget`, `should_extend_component_cap`).
        # Without this, a stale graduation could grant cap / budget
        # extensions on a run that has actually regressed.
        if comp in graduated_this_run:
            graduated_this_run[:] = [g for g in graduated_this_run if g != comp]
        # Classify the failure (`failure_classifier.py`). Only a verified
        # KERNEL_MISSING verdict writes a persistent skip-list entry; the
        # rest are diagnostic — the loop logs them and the component
        # stays in the active queue, retried on the next session.
        skip_category = "ITERATION_BUDGET"
        skip_reason = reason
        try:
            from ..failure_classifier import classify_failure, skip_category_for_verdict

            scan_text = (last_failures or "") + "\n" + (last_failure_details or "")
            verdict = classify_failure(
                reason=reason,
                failure_text=scan_text,
            )
            skip_category = skip_category_for_verdict(verdict)
            skip_reason = f"{reason} — {verdict.class_name} ({verdict.confidence} confidence): {verdict.reason}"
            if verdict.missing_op:
                skip_reason += f" [op={verdict.missing_op}]"
            persistence_note = "persisted" if skip_category == "KERNEL_MISSING" else "transient, retry next run"
            print(
                f"  [failure-class] `{comp}` -> {verdict.class_name} "
                f"({verdict.confidence}); category={skip_category} ({persistence_note})"
            )
            # Auto-invoke decomposition for components whose failure
            # verdict warrants it. The HF load is expensive — to avoid
            # bloating the auto-loop process, we spawn the existing
            # `decompose --write-plan` CLI in a subprocess. The
            # `consume_decomposition_plan` consumer at loop start
            # (auto_iterate.py:158-173) picks up the plan on the NEXT
            # iteration and adds the children as NEW components.
            #
            # Tracked in decomposition_auto_attempted so we don't keep
            # re-running the subprocess on every iter for the same
            # parent. Empty children (leaf module) is also "attempted."
            from ..component_decomposer import failure_class_warrants_decomposition

            if failure_class_warrants_decomposition(verdict.class_name) and comp not in decomposition_auto_attempted:
                decomposition_auto_attempted.add(comp)
                try:
                    _decomp_proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.tt_hw_planner",
                            "decompose",
                            MODEL,
                            comp,
                            "--write-plan",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    if _decomp_proc.returncode == 0:
                        print(
                            f"  [decompose-auto] `{comp}` decomposed "
                            f"(verdict={verdict.class_name}); children queued "
                            f"in decomposition_plan.json — consumer will add "
                            f"them as NEW components on the next iter."
                        )
                    elif _decomp_proc.returncode == 1:
                        # rc=1 = leaf module, no non-trivial children.
                        # Decomposition genuinely exhausted; don't print
                        # the manual hint since auto-attempt confirmed
                        # nothing to decompose.
                        print(
                            f"  [decompose-auto] `{comp}` is primitive — "
                            f"no non-trivial children to spawn (verdict="
                            f"{verdict.class_name} stands)"
                        )
                    else:
                        print(
                            f"  [decompose-auto] `{comp}` subprocess failed "
                            f"(rc={_decomp_proc.returncode}): "
                            f"{(_decomp_proc.stderr or '')[:200]}",
                            file=sys.stderr,
                        )
                except subprocess.TimeoutExpired:
                    print(
                        f"  [decompose-auto] `{comp}` timed out after 180s "
                        f"(HF load too slow); falling back to manual hint: "
                        f"python -m scripts.tt_hw_planner decompose {MODEL} {comp}",
                        file=sys.stderr,
                    )
                except Exception as _decomp_exc:
                    print(
                        f"  [decompose-auto] `{comp}` error: "
                        f"{type(_decomp_exc).__name__}: {_decomp_exc}; "
                        f"falling back to manual hint",
                        file=sys.stderr,
                    )
        except Exception as _classify_exc:
            print(f"  [failure-class] classifier raised on `{comp}`: {_classify_exc}", file=sys.stderr)
        # Persist the verdict to the skip-list. If the classifier raised,
        # skip_category stays at the safe "COLD" default and skip_reason
        # stays at the raw input reason — degraded but not crashed.
        try:
            from ..overlay_manager import persist_skip as _persist_skip_cat

            _persist_skip_cat(MODEL, comp, reason=skip_reason, category=skip_category)
        except Exception as _persist_exc:
            # In-memory permanently_skipped is already updated so this
            # run is fine; the loss is next-run visibility into the
            # skip. Surface to stderr so the user knows their skip-list
            # didn't persist.
            print(
                f"  [persist-skip] failed to write skip-list entry for `{comp}`: " f"{_persist_exc}",
                file=sys.stderr,
            )
        safe = _safe_id(comp)
        stub_path = demo_dir / "_stubs" / f"{safe}.py"
        last_good_path = stub_path.with_suffix(".py.last_good_native")
        best_native_path = stub_path.with_suffix(".py.best_native")
        preiter_native_path = stub_path.with_suffix(".py.preiter_native")
        bak_path = stub_path.with_suffix(".py.bak")
        restored_from = None
        if last_good_path.is_file():
            try:
                stub_path.write_text(
                    last_good_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                restored_from = (
                    f"{safe_relative_to_root(last_good_path)} " f"(last graduated native — strongest signal)"
                )
            except Exception:
                restored_from = None
        if restored_from is None and best_native_path.is_file():
            try:
                stub_path.write_text(
                    best_native_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                best_pcc_str = (
                    f" (best in-session PCC={best_pcc_per_component[comp]:.4f})"
                    if comp in best_pcc_per_component
                    else ""
                )
                restored_from = (
                    f"{safe_relative_to_root(best_native_path)} "
                    f"(best in-session native{best_pcc_str} — "
                    f"better than .preiter_native and .bak)"
                )
            except Exception:
                restored_from = None
        if restored_from is None and preiter_native_path.is_file():
            try:
                stub_path.write_text(
                    preiter_native_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                restored_from = (
                    f"{safe_relative_to_root(preiter_native_path)} "
                    f"(pre-iter native — better than .bak torch wrapper)"
                )
            except Exception:
                restored_from = None
        if restored_from is None and bak_path.is_file():
            try:
                stub_path.write_text(bak_path.read_text())
                restored_from = str(safe_relative_to_root(bak_path))
            except Exception:
                restored_from = None
        if restored_from is None:
            try:
                _rewrite_components_to_stable_fallback(demo_dir, [comp])
                restored_from = "stable CPU-fallback stub"
            except Exception as exc:
                print(
                    f"  could not restore CPU fallback for `{comp}`: {exc}",
                    file=sys.stderr,
                )
        print(
            f"  Component `{comp}` left on CPU fallback after "
            f"{attempts_per_component.get(comp, 0)}/{max_attempts_per_component} "
            f"failed attempts (restored from {restored_from or 'unknown source'}). "
            f"Reason: {reason}"
        )

    seed_has_demo_data = bool(seed_report.get("failed_tests") or seed_report.get("failed_components"))
    if seed_has_demo_data and not bool(seed_report.get("all_passed", False)):
        last_failures = str(seed_report.get("summary", "(no failure summary)"))
        last_failure_details = str(seed_report.get("details", "(no failure traceback parsed)"))
        last_failed_components = list(seed_report.get("failed_components", []))
        last_failed_tests = list(seed_report.get("failed_tests", []))
        for c in last_failed_components:
            verified_fail.add(c)
            stale_snap = demo_dir / "_stubs" / f"{_safe_id(c)}.py.last_good_native"
            if stale_snap.is_file():
                try:
                    stale_snap.unlink()
                    print(
                        f"  cleared stale native snapshot for `{c}` "
                        f"(seed pytest shows it's currently failing): "
                        f"{safe_relative_to_root(stale_snap)}"
                    )
                except Exception as exc:
                    print(
                        f"  could not clear stale snapshot {stale_snap}: {exc}",
                        file=sys.stderr,
                    )
        failure_class = _classify_failure(last_failures, last_failure_details)
        signature = _failure_signature(last_failures, last_failure_details)
        for comp in set(last_failed_components):
            k = f"{comp}|{failure_class}|{signature}"
            repeat_error_counts[k] = repeat_error_counts.get(k, 0) + 1
    if target_components and not last_failed_components:
        last_failed_components = list(target_components)
        if not last_failures:
            last_failures = (
                f"Targeting {len(target_components)} NEW component(s) still on CPU "
                f"fallback (torch reference). Replace each with a native ttnn implementation."
            )
            last_failure_details = (
                "Each listed component is currently implemented as a torch-reference "
                "fallback wrapper inside _stubs/<comp>.py. The component's smoke/PCC test "
                "passes via that fallback, but the model is not yet running natively on "
                "the device for those ops. The goal is to replace each fallback with a "
                "real ttnn implementation that still passes the PCC test."
            )
    device_reset_only_skips = 0
    max_device_reset_only_skips = 2

    try:
        from ..bringup_loop import _emit_pcc_template as _bl_emit_pcc_template

        _smoke_to_pcc_done: List[str] = []
        try:
            _bs_data = json.loads((demo_dir / "bringup_status.json").read_text())
        except Exception:
            _bs_data = {}
        for c in _bs_data.get("components", []) or []:
            if c.get("status") != "NEW":
                continue
            name = str(c.get("name", "")).strip()
            if not name:
                continue
            safe = _safe_id(name)
            stub_p = demo_dir / "_stubs" / f"{safe}.py"
            test_p = demo_dir / "tests" / "pcc" / f"test_{safe}.py"
            if not (stub_p.is_file() and test_p.is_file()):
                continue
            try:
                head = test_p.read_text(errors="ignore")[:600]
            except Exception:
                continue
            if "Phase-1 SMOKE test" not in head:
                continue
            if not _stub_has_graduated_from_autofill(stub_p):
                continue
            try:
                _bl_emit_pcc_template(
                    demo_dir=demo_dir,
                    component_name=name,
                    model_id=MODEL,
                    hf_reference=c.get("hf_reference") or "",
                    new_shape=c.get("new_shape") or {},
                    repo_root=REPO_ROOT,
                    overwrite=True,
                )
                _smoke_to_pcc_done.append(name)
            except Exception as _exc:
                print(
                    f"  WARNING: SMOKE->PCC upgrade for `{name}` failed: {_exc}",
                    file=sys.stderr,
                )
        if _smoke_to_pcc_done:
            banner(
                f"PRE-FLIGHT SMOKE->PCC upgrade: forced {len(_smoke_to_pcc_done)} "
                f"native stub(s) to use Phase-2 PCC tests so they're actually "
                f"PCC-validated (was Phase-1 SMOKE = trivially passing)"
            )
            for n in _smoke_to_pcc_done:
                print(f"  - {n}")
    except Exception as _exc:
        print(
            f"  WARNING: SMOKE->PCC upgrade pass failed (continuing): {_exc}",
            file=sys.stderr,
        )

    try:
        from ..capture_inputs import capture_real_inputs, upgrade_all_tests_in_demo

        _cap_status_path = demo_dir / "bringup_status.json"
        if _cap_status_path.is_file():
            _cap_components = [
                str(c.get("name", "")).strip()
                for c in json.loads(_cap_status_path.read_text()).get("components", [])
                if c.get("status") == "NEW" and str(c.get("name", "")).strip()
            ]
        else:
            _cap_components = []
        if _cap_components:
            banner(
                f"PRE-FLIGHT capture-inputs: running HF model once to record "
                f"REAL per-component IO tensors so PCC tests don't have to "
                f"fabricate them"
            )
            _cap_results = capture_real_inputs(
                model_id=MODEL,
                demo_dir=demo_dir,
                components=_cap_components,
                verbose=True,
            )
            _captured_n = sum(1 for info in _cap_results.values() if info.get("status") == "captured")
            print(
                f"  captured {_captured_n}/{len(_cap_components)} components; " f"patching PCC tests to load them ..."
            )
            _ups = upgrade_all_tests_in_demo(demo_dir)
            _ups_n = sum(1 for _n, m in _ups if m)
            if _ups_n:
                print(f"  upgraded {_ups_n} test file(s) to prefer captured inputs.")
            else:
                print("  test files already upgraded (idempotent).")
        else:
            print("  (no NEW components yet — skipping capture-inputs.)")
    except Exception as _cap_exc:
        print(
            f"  pre-flight capture-inputs failed (continuing with synthetic " f"inputs): {_cap_exc}",
            file=sys.stderr,
        )

    try:
        _preflight_status_path = demo_dir / "bringup_status.json"
        if _preflight_status_path.is_file():
            _preflight_new_components = [
                str(c.get("name", "")).strip()
                for c in json.loads(_preflight_status_path.read_text()).get("components", [])
                if c.get("status") == "NEW" and str(c.get("name", "")).strip()
            ]
        else:
            _preflight_new_components = []
    except Exception:
        _preflight_new_components = []

    # G7 (agentic.learnings): before pre-flight, try to short-circuit
    # any component that already has a learned fix on disk for THIS
    # HF architecture. apply_fix() git-applies the diff to the stub;
    # if the diff is stale (e.g. patch context no longer matches), we
    # silently fall through to the normal LLM iteration. Direct reuse
    # of the existing engine.
    try:
        from ..agentic.executor import _load_hf_config as _load_hf
        from ..agentic.learnings import apply_fix, compute_arch_signature, lookup_fix

        _arch_sig_preflight = compute_arch_signature(_load_hf(MODEL))
        if _arch_sig_preflight:
            for _comp in _preflight_new_components:
                _stub_p = demo_dir / "_stubs" / f"{_safe_id(_comp)}.py"
                if _stub_has_graduated_from_autofill(_stub_p):
                    continue  # already native; nothing to apply
                _fix = lookup_fix(arch_signature=_arch_sig_preflight, first_diverging_qn=_comp)
                if _fix is None:
                    continue
                _ok, _msg = apply_fix(fix=_fix, workspace_root=BRINGUP_ROOT())
                if _ok:
                    print(
                        f"  [agentic:G7] applied learned fix for `{_comp}` "
                        f"from prior run (arch_sig={_arch_sig_preflight[:12]}); "
                        f"will be PCC-validated in pre-flight."
                    )
                else:
                    print(
                        f"  [agentic:G7] learned fix for `{_comp}` did not " f"apply cleanly: {_msg[:200]}",
                        file=sys.stderr,
                    )
    except Exception as _g7_lookup_exc:
        print(
            f"  [agentic:G7] non-fatal during preflight lookup: " f"{type(_g7_lookup_exc).__name__}: {_g7_lookup_exc}",
            file=sys.stderr,
        )

    _preflight_native_components: List[str] = []
    for comp in _preflight_new_components:
        if comp in graduated_this_run or comp in unverified_native_this_run or comp in permanently_skipped:
            continue
        stub_path = demo_dir / "_stubs" / f"{_safe_id(comp)}.py"
        try:
            if _stub_has_graduated_from_autofill(stub_path):
                _preflight_native_components.append(comp)
        except Exception:
            pass

    _preflight_test_files: List[str] = []
    for comp in _preflight_native_components:
        tp = demo_dir / "tests" / "pcc" / f"test_{_safe_id(comp)}.py"
        if tp.is_file():
            _preflight_test_files.append(str(safe_relative_to_root(tp)))

    if _preflight_test_files:
        banner(
            f"AUTO-ITERATE pre-flight: {len(_preflight_test_files)} "
            f"stub(s) already native on disk — running pytest to "
            f"classify BEFORE invoking the LLM (saves wasted iterations)"
        )
        for comp in _preflight_native_components:
            print(f"  - {comp}  ({safe_relative_to_root(demo_dir)}/_stubs/{_safe_id(comp)}.py)")
        _preflight_rc = 0
        try:
            _preflight_rc = _run_focused_pytest(
                model_id=MODEL,
                test_files=_preflight_test_files,
                allow_kill_stale=allow_kill_stale,
                allow_device_reset=allow_device_reset,
            )
            _preflight_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
        except Exception as exc:
            print(
                f"  pre-flight pytest crashed ({exc}); falling back to " f"normal LLM loop on these components.",
                file=sys.stderr,
            )
            _preflight_report = {}

        if _preflight_report:
            _no_hw_msg = _detect_no_hardware_failure(_preflight_report)
            if _no_hw_msg is not None:
                for _line in _format_no_hardware_diagnostic_banner(_no_hw_msg):
                    print(_line, file=sys.stderr)
                return 2

        if _preflight_rc == 124 or (isinstance(_preflight_rc, int) and _preflight_rc < 0):
            _pf_stage_map: Dict[str, str] = dict(_LAST_PYTEST_STAGES)
            _pf_ordered_components: List[str] = []
            _pf_last_stage_of: Dict[str, str] = {}
            for _tname, _tstage in _pf_stage_map.items():
                _m = re.search(r"test_([A-Za-z0-9_]+)\.py", _tname)
                if not _m:
                    continue
                _cmp = _m.group(1)
                if _cmp in _preflight_native_components:
                    _pf_ordered_components.append(_cmp)
                    _pf_last_stage_of[_cmp] = _tstage

            _seen_cmp: set = set()
            _pf_ordered_dedup: List[str] = []
            for _cmp in _pf_ordered_components:
                if _cmp not in _seen_cmp:
                    _seen_cmp.add(_cmp)
                    _pf_ordered_dedup.append(_cmp)
            if _pf_ordered_dedup:
                _pf_target = _pf_ordered_dedup[-1]
                verified_fail.add(_pf_target)
                last_failed_components = [_pf_target]

                _record_failure_for_component(_pf_target, "HANG", None)
                _pf_target_test = demo_dir / "tests" / "pcc" / f"test_{_safe_id(_pf_target)}.py"
                try:
                    last_failed_tests = [str(safe_relative_to_root(_pf_target_test))]
                except Exception:
                    last_failed_tests = [str(_pf_target_test)]
                _pf_budget_s = int(os.environ.get("TT_PLANNER_PYTEST_TIMEOUT_S", "600"))
                last_failures = (
                    f"pre-flight pytest WALL-CLOCK BUDGET EXHAUSTED "
                    f"at {_pf_budget_s}s — `{_pf_target}` was the "
                    f"test mid-execution at the moment of the "
                    f"SIGKILL (last stage="
                    f"`{_pf_last_stage_of.get(_pf_target, '?')}`). "
                    f"Treat as a HANG in the `{_pf_target}` stub."
                )
                last_failure_details = (
                    "Last [bringup] stage line(s) printed before "
                    "the pre-flight pytest hang (sequential pytest "
                    "order, last entry is the culprit):\n"
                    + "\n".join(f"  - {k}: stage={v}" for k, v in _pf_stage_map.items())
                )
                print(
                    f"  pre-flight HANG: targeting `{_pf_target}` "
                    f"in iter 1 (was the test mid-execution at "
                    f"SIGKILL time; the other "
                    f"{len(_preflight_native_components) - 1} "
                    f"component(s) couldn't be classified because "
                    f"pytest never got to them).",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  pre-flight pytest timed out before ANY "
                    f"[bringup] stage line was printed — hang is "
                    f"earlier than torch reference build (likely "
                    f"module import / scaffold load). Cannot infer "
                    f"culprit; iter 1 will fall back to alphabetical "
                    f"targeting.",
                    file=sys.stderr,
                )

        if _preflight_report:
            _pf_passed = set(_preflight_report.get("passed_components", []) or [])
            _pf_failed = set(_preflight_report.get("failed_components", []) or [])
            _pf_skipped = set(_preflight_report.get("skipped_components", []) or [])
            _pf_per_skipped = _preflight_report.get("per_skipped", {}) if isinstance(_preflight_report, dict) else {}
            _pf_harness_markers = (
                "HF reference forward",
                "_make_arg_for()",
                "synthetic inputs from _make_arg_for",
                "incompatible with this submodule's expected shapes",
                "the synthetic inputs",
            )
            for comp in _preflight_native_components:
                if comp in _pf_passed and comp not in _pf_failed and comp not in _pf_skipped:
                    if comp not in graduated_this_run and comp not in permanently_skipped:
                        graduated_this_run.append(comp)
                        validated_this_run.add(comp)
                        verified_fail.discard(comp)
                        _snapshot_native_stub(comp)
                        print(
                            f"  pre-flight: `{comp}` GRADUATED — PCC test "
                            f"PASSED on the existing native stub. No LLM "
                            f"iteration needed."
                        )
                elif comp in _pf_skipped and comp not in _pf_failed:
                    _pf_reasons: List[str] = []
                    if isinstance(_pf_per_skipped, dict):
                        for entry in _pf_per_skipped.values():
                            if isinstance(entry, dict) and entry.get("component") == comp:
                                r = str(entry.get("reason") or "").strip()
                                if r:
                                    _pf_reasons.append(r)
                    if any(any(m in r for m in _pf_harness_markers) for r in _pf_reasons):
                        if (
                            comp not in graduated_this_run
                            and comp not in unverified_native_this_run
                            and comp not in permanently_skipped
                        ):
                            unverified_native_this_run.add(comp)
                            skipped_components_this_run.discard(comp)
                            verified_fail.discard(comp)
                            _pf_reason_blob = "; ".join(_pf_reasons) or "(no skip reason captured)"
                            # Pre-flight UNVERIFIED NATIVE = harness
                            # can't run PCC. Classify as TOOL_BUG.
                            persist_skip(MODEL, comp, _pf_reason_blob, category="TOOL_BUG")
                            print(
                                f"  pre-flight: `{comp}` UNVERIFIED NATIVE "
                                f"— stub is native ttnn but PCC could not "
                                f"be measured (test harness incompatibility)"
                                f". Not retried, blocks bring-up-complete. "
                                f"Reason: {_pf_reason_blob}"
                            )
                elif comp in _pf_failed:
                    verified_fail.add(comp)
                    print(
                        f"  pre-flight: `{comp}` FAILED PCC on the existing "
                        f"native stub — LLM loop will retry to fix the real "
                        f"failure."
                    )

        try:
            _pf_sep = "=" * 78
            print()
            print(_pf_sep)
            print(f"  Pre-flight compute split (baseline before iter 1):")
            for _line in _format_compute_split(MODEL, label="components ", indent="  "):
                print(_line)
            for _line in _format_op_split(
                MODEL,
                label="operations  ",
                indent="  ",
                show_per_component=False,
            ):
                print(_line)
            print(_pf_sep)
        except Exception as _pf_split_exc:
            print(
                f"  (pre-flight compute split unavailable: " f"{_pf_split_exc})",
                file=sys.stderr,
            )

    # Brain (G8) owns the run-level "should we extend the budget?"
    # decision via agentic.convergence.should_extend_budget. The loop
    # just consults the brain at budget exhaustion and executes the
    # verdict. All policy (residue threshold, momentum gate, bump size,
    # per-run cap) lives in the brain module so it tunes in ONE place.
    budget_extensions_used = 0
    it = 0
    while True:
        it += 1
        if it > max_iters:
            _ext_ungrad, _ext_smoke = _auto_iteration_blockers(MODEL)
            _ext_pending = sorted((set(_ext_ungrad) | set(_ext_smoke)) - set(permanently_skipped))
            from ..agentic.convergence import should_extend_budget as _brain_should_extend

            _verdict = _brain_should_extend(
                pending_components=_ext_pending,
                pcc_history_per_component=pcc_history_per_component,
                graduated_this_run=graduated_this_run,
                max_iters=max_iters,
                extensions_used=budget_extensions_used,
            )
            if _verdict.extend:
                banner(
                    f"AUTO-EXTEND (brain G8): max_iters {max_iters} → "
                    f"{max_iters + _verdict.bump} — {_verdict.reason}"
                )
                max_iters += _verdict.bump
                budget_extensions_used += 1
                # Decrement so the bump translates to N real body iters,
                # not N-1. Without this, the exhaustion-check iter
                # "consumes" one slot from the bump.
                it -= 1
                continue
            # Brain declined to extend — leave trace so users see why.
            if _ext_pending:
                print(f"  [brain G8] declined to extend budget: {_verdict.reason}")
            break
        seed_failure_class = _classify_failure(last_failures, last_failure_details)
        if (
            seed_failure_class == "DEVICE_NEEDS_RESET"
            and allow_device_reset
            and device_reset_only_skips < max_device_reset_only_skips
        ):
            banner(
                f"AUTO-ITERATE {it}/{max_iters}: seed failure is DEVICE_NEEDS_RESET — "
                f"skipping LLM, running `tt-smi -r` + re-running pytest only"
            )
            print(
                "  The previous iteration crashed inside the UMD sysmem-mapping "
                "path (`Proceeding could lead to undefined behavior`). This is a "
                "device-state problem, not a code defect — invoking the LLM would "
                "burn the per-iteration budget on a stub that isn't broken. Resetting "
                "the device and re-running the same pytest directly."
            )
            if not _run_tt_smi_reset(context=f"iter-{it}:skip-llm-device-reset"):
                print(
                    "  device reset did not succeed; falling through to LLM as last resort.",
                    file=sys.stderr,
                )
            else:
                tests_to_rerun = list(last_failed_tests) if last_failed_tests else []
                if not tests_to_rerun:
                    tests_to_rerun = _list_component_pcc_tests(demo_dir)
                if tests_to_rerun:
                    rerun_rc = _run_focused_pytest(
                        model_id=MODEL,
                        test_files=tests_to_rerun,
                        allow_kill_stale=allow_kill_stale,
                        allow_device_reset=allow_device_reset,
                    )
                else:
                    rerun_rc = 0
                device_reset_only_skips += 1
                rerun_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
                if rerun_rc == 0 and bool(rerun_report.get("all_passed", False)):
                    ungrad_after, smoke_after = _auto_iteration_blockers(MODEL)
                    if not ungrad_after and not smoke_after:
                        banner(
                            f"AUTO-ITERATE {it}/{max_iters}: DEVICE_NEEDS_RESET cleared "
                            f"after `tt-smi -r` + pytest re-run — all PCC tests pass"
                        )
                        return 0
                last_failures = str(rerun_report.get("summary", last_failures))
                last_failure_details = str(rerun_report.get("details", last_failure_details))
                last_failed_components = list(rerun_report.get("failed_components", last_failed_components))
                last_failed_tests = list(rerun_report.get("failed_tests", last_failed_tests))
                continue

        if last_failed_components:
            at_cap_now = [
                c
                for c in set(last_failed_components)
                if c not in permanently_skipped and c not in graduated_this_run and _is_at_cap(c)
            ]
            # Brain (G8) gets the final say on each component at cap:
            # extend the cap (component is close to converging) or route
            # to CPU fallback. Without this, the cap is a hard flag-gated
            # wall the brain never sees — which is exactly how today's
            # run shortcut at iter 1 (cap exhausted → CPU fallback →
            # budget exhaustion never reached → brain never consulted).
            from ..agentic.convergence import should_extend_component_cap as _brain_should_extend_cap

            extended_at_cap: List[str] = []
            for c in sorted(at_cap_now):
                _cap_verdict = _brain_should_extend_cap(
                    component=c,
                    consecutive_same_class=consecutive_same_class_attempts.get(c, 0),
                    effective_cap=_effective_attempt_cap(c),
                    pcc_history=pcc_history_per_component.get(c, []),
                    last_pcc=last_pcc_per_component.get(c),
                    last_failure_class=last_failure_class_per_component.get(c, ""),
                    graduated_this_run=graduated_this_run,
                    extensions_used_for_this_component=cap_extensions_used_per_component.get(c, 0),
                )
                if _cap_verdict.extend:
                    # Reset the consecutive-same-class counter by the
                    # bump amount; this is what `_effective_attempt_cap`
                    # measures against, so a reduction here grants
                    # `bump` more attempts effectively.
                    new_consec = max(0, consecutive_same_class_attempts.get(c, 0) - _cap_verdict.bump)
                    consecutive_same_class_attempts[c] = new_consec
                    cap_extensions_used_per_component[c] = cap_extensions_used_per_component.get(c, 0) + 1
                    banner(
                        f"CAP-EXTEND (brain G8): `{c}` granted +{_cap_verdict.bump} "
                        f"attempts — {_cap_verdict.reason}"
                    )
                    extended_at_cap.append(c)
                else:
                    print(f"  [brain G8] `{c}` cap-fallback: {_cap_verdict.reason}")
                    _skip_component_to_fallback(
                        c,
                        f"hit per-component attempt cap " f"(consec-same-class {_attempts_display(c)})",
                    )
            # Only components NOT extended by brain are routed to fallback.
            at_cap_now = [c for c in at_cap_now if c not in extended_at_cap]
            if at_cap_now:
                last_failed_components = [c for c in last_failed_components if c not in at_cap_now]
                if not last_failed_components:
                    last_failed_tests = []
                    last_failures = ""
                    last_failure_details = ""

        ungrad_now, smoke_now = _auto_iteration_blockers(MODEL)
        if target_components:
            allowed = set(target_components)
            ungrad_now = [c for c in ungrad_now if c in allowed]
            smoke_now = [c for c in smoke_now if c in allowed]

        try:
            new_component_names = {
                str(c.get("name", "")).strip()
                for c in (
                    json.loads((demo_dir / "bringup_status.json").read_text()).get("components", [])
                    if (demo_dir / "bringup_status.json").is_file()
                    else []
                )
                if c.get("status") == "NEW" and str(c.get("name", "")).strip()
            }
        except Exception:
            new_component_names = set()
        unvalidated_seed = (
            new_component_names
            - validated_this_run
            - set(permanently_skipped)
            - set(graduated_this_run)
            - set(unverified_native_this_run)
        )

        partial_cpu_pool: List[str] = [] if allow_partial_cpu else _partial_cpu_components(MODEL)
        partial_cpu_set = set(partial_cpu_pool) - set(permanently_skipped) - set(unverified_native_this_run)
        candidate_pool = sorted(
            (
                set(ungrad_now)
                | set(smoke_now)
                | set(verified_fail)
                | set(last_failed_components or [])
                | unvalidated_seed
                | partial_cpu_set
            )
            - set(permanently_skipped)
            - (set(graduated_this_run) - partial_cpu_set)
            - set(unverified_native_this_run)
        )
        if only_component:
            if only_component in candidate_pool:
                candidate_pool = [only_component]
            else:
                candidate_pool = []
                print(
                    f"  [sandbox --auto-only-component {only_component!r}] "
                    f"component not in current candidate pool — nothing to iterate on; exiting clean."
                )
        if not candidate_pool:
            if permanently_skipped:
                banner(
                    f"AUTO-ITERATE {it}/{max_iters}: every remaining component hit the "
                    f"{max_attempts_per_component}-attempt cap and is now on CPU "
                    f"fallback — running one final pytest to confirm the demo still "
                    f"runs end-to-end"
                )
                print(f"  CPU-fallback components: {', '.join(permanently_skipped)}")
                if graduated_this_run:
                    print(f"  graduated to native TTNN this run: " f"{', '.join(sorted(set(graduated_this_run)))}")
                final_tests = _list_component_pcc_tests(demo_dir)
                if final_tests:
                    final_rc = _run_focused_pytest(
                        model_id=MODEL,
                        test_files=final_tests,
                        allow_kill_stale=allow_kill_stale,
                        allow_device_reset=allow_device_reset,
                    )
                    final_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
                    # Brain G8 phantom-failure check: if any failure is a
                    # stale-decomposed-parent test, archive it and re-pytest
                    # BEFORE applying the "unexpected failure" fallback
                    # logic. Otherwise SAM2-style runs report rc=1 from
                    # phantoms that should never have run.
                    _phantom_post = _brain_handle_phantom_failures(
                        MODEL=MODEL,
                        demo_dir=demo_dir,
                        final_failed=sorted(set(final_report.get("failed_components", []) or [])),
                        banner_fn=banner,
                        allow_kill_stale=allow_kill_stale,
                        allow_device_reset=allow_device_reset,
                    )
                    if _phantom_post is not None:
                        final_rc, final_report = _phantom_post["rc"], _phantom_post["report"]
                    if final_rc != 0 or not bool(final_report.get("all_passed", False)):
                        unexpected_failed = sorted(
                            set(final_report.get("failed_components", []) or []) - set(permanently_skipped)
                        )
                        if unexpected_failed:
                            print(
                                f"  Final pytest revealed {len(unexpected_failed)} "
                                f"component(s) the loop heuristic believed were "
                                f"native but actually fail PCC: "
                                f"{', '.join(unexpected_failed)}"
                            )
                            print(
                                "  Restoring each from ..bak (CPU fallback) and "
                                "re-running the scoped sweep once before giving up."
                            )
                            for c in unexpected_failed:
                                _skip_component_to_fallback(
                                    c,
                                    "final-sweep revealed PCC failure that prior "
                                    "iterations missed (heuristic false positive)",
                                )
                                graduated_this_run = [g for g in graduated_this_run if g != c]
                            final_rc = _run_focused_pytest(
                                model_id=MODEL,
                                test_files=final_tests,
                                allow_kill_stale=allow_kill_stale,
                                allow_device_reset=allow_device_reset,
                            )
                            final_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
                    if final_rc == 0 and bool(final_report.get("all_passed", False)):
                        unverified_inner = sorted(set(unverified_native_this_run))
                        if unverified_inner:
                            banner(
                                f"AUTO-ITERATE {it}/{max_iters}: bring-up "
                                f"NOT complete on {BOX} — "
                                f"{len(unverified_inner)} component(s) "
                                f"lack PCC validation"
                            )
                            print(
                                "  The following component(s) have a "
                                "native ttnn forward on disk but their "
                                "PCC test SKIPPED because _make_arg_for() "
                                "inputs are shape-incompatible with the "
                                "HF reference. Without a passing pcc("
                                "ttnn, torch) >= 0.99, we have NO proof "
                                "of numerical correctness."
                            )
                            for c in unverified_inner:
                                print(f"      - {c}")
                            _print_bringup_summary(MODEL, box=BOX, sep=sep)
                            print("\n  Per-component status:")
                            if graduated_this_run:
                                print(f"    PCC-validated   : {', '.join(sorted(set(graduated_this_run)))}")
                            print(f"    UNVERIFIED NATIVE: {', '.join(unverified_inner)}")
                            if permanently_skipped:
                                print(f"    CPU fallback    : {', '.join(sorted(set(permanently_skipped)))}")
                            print(
                                "\n  Fix the per-component PCC test "
                                "scaffolds for each UNVERIFIED NATIVE "
                                "component (correct arg names, ranks, "
                                "and dtypes), then re-run:\n"
                                f"    python -m scripts.tt_hw_planner promote {MODEL} \\\n"
                                f"        --box {BOX} --auto --auto-agent {provider} \\\n"
                                f"        --auto-max-iters 12 --auto-agent-timeout 1500\n"
                            )
                            return 1
                        banner(
                            f"AUTO-ITERATE {it}/{max_iters}: demo runs end-to-end on "
                            f"{BOX} with partial native bring-up "
                            f"({len(graduated_this_run)} native, "
                            f"{len(permanently_skipped)} on CPU fallback)"
                        )
                        _print_bringup_summary(MODEL, box=BOX, sep=sep)
                        print("\n  Per-component status:")
                        if graduated_this_run:
                            print(f"    native TTNN : {', '.join(sorted(set(graduated_this_run)))}")
                        print(f"    CPU fallback: {', '.join(sorted(set(permanently_skipped)))}")
                        print(
                            "\n  To retry the CPU-fallback components later (higher iter budget,\n"
                            "  bigger model, hand-iteration, or after the scaffolder learns to\n"
                            "  decompose them further):\n"
                            f"    python -m scripts.tt_hw_planner promote {MODEL} \\\n"
                            f"        --box {BOX} --auto --auto-agent {provider} \\\n"
                            f"        --auto-max-iters 12 --auto-agent-timeout 1500\n"
                        )
                        _brain_sync_graduated_to_main_tree(
                            MODEL=MODEL,
                            demo_dir=demo_dir,
                            graduated_this_run=graduated_this_run,
                            banner_fn=banner,
                        )
                        return 0
                    still_failing = sorted(set(final_report.get("failed_components", []) or []))
                    print(
                        f"  Final pytest STILL failing after fallback restoration "
                        f"on components: {', '.join(still_failing) if still_failing else '(unknown)'}\n"
                        f"  These are likely test-scaffolder bugs (synthetic-input "
                        f"generator can't fabricate the multi-arg forward signature) "
                        f"or genuine HF-reference forward errors. Inspect:\n"
                        f"    {safe_relative_to_root(demo_dir) if demo_dir.is_absolute() else demo_dir}/tests/pcc/",
                        file=sys.stderr,
                    )
                    return 1
            if unverified_native_this_run:
                unverified_only = sorted(set(unverified_native_this_run))
                banner(
                    f"AUTO-ITERATE {it}/{max_iters}: bring-up NOT complete "
                    f"on {BOX} — {len(unverified_only)} component(s) "
                    f"native but PCC-unverified"
                )
                print(
                    "  These component(s) have a native ttnn forward on "
                    "disk but PCC could not be measured because the "
                    "auto-generated _make_arg_for() inputs are shape-"
                    "incompatible with the HF reference's expected "
                    "signature. Without a passing pcc(ttnn, torch) >= "
                    "0.99 we have NO proof of numerical correctness, so "
                    "the loop refuses to declare bring-up complete."
                )
                for c in unverified_only:
                    test_rel = demo_dir / "tests" / "pcc" / f"test_{_safe_id(c)}.py"
                    try:
                        test_rel = safe_relative_to_root(test_rel)
                    except Exception:
                        pass
                    print(f"      - {c}  (hand-fix `{test_rel}`)")
                if graduated_this_run:
                    print(f"  PCC-validated this run: " f"{', '.join(sorted(set(graduated_this_run)))}")
                return 1

            _had_real_work = bool(graduated_this_run or validated_this_run or new_component_names)
            if not _had_real_work:
                banner(
                    f"AUTO-ITERATE {it}/{max_iters}: REFUSING TO DECLARE "
                    f"SUCCESS — no components were ever scaffolded or "
                    f"validated. Empty bringup_status.json indicates the "
                    f"scaffold step short-circuited (likely ALREADY-"
                    f"SUPPORTED routing reused a pre-existing demo "
                    f"without producing per-component PCC tests). "
                    f"Investigate the scaffold step; do NOT trust this "
                    f"as a successful bring-up."
                )
                return 1
            banner(
                f"AUTO-ITERATE {it}/{max_iters}: all components already graduated "
                f"to native TTNN — nothing left to do"
            )
            _brain_sync_graduated_to_main_tree(
                MODEL=MODEL,
                demo_dir=demo_dir,
                graduated_this_run=graduated_this_run,
                banner_fn=banner,
            )
            return 0

        def _pick_target() -> str:
            """Pick the next iteration's target component.

            Target-rotation discipline (fixes 2026-05-22 22:00 UTC
            sam2-hiera-tiny observation where the loop spent 4
            attempts on `encoder_stack` while `vision_config` and
            `self_attention` got ZERO attempts in 6 iters):

            1. Of components that failed in the LAST iter AND are not
               at-cap, pick the one with the FEWEST total attempts so
               far. Two-component fairness: if encoder_stack just
               failed and vision_config hasn't been tried, pick
               vision_config. Without this fairness rule, the loop
               always picks alphabetically-first which fixates.
            2. If that's a tie, prefer the one with the LOWER
               consecutive-same-class count (least-recently-stuck).
            3. Final tiebreak is alphabetic for determinism.

            If no last-iter-failed components are under-cap, fall
            through to picking ANY under-cap candidate, with the same
            three-tier key (lowest attempts, lowest consec, alpha).
            """
            live_failed = [c for c in (last_failed_components or []) if c in candidate_pool and not _is_at_cap(c)]
            if live_failed:
                return min(
                    live_failed,
                    key=lambda c: (
                        attempts_per_component.get(c, 0),
                        consecutive_same_class_attempts.get(c, 0),
                        c,
                    ),
                )
            under_cap = [c for c in candidate_pool if not _is_at_cap(c)]
            if not under_cap:
                return min(
                    candidate_pool,
                    key=lambda c: (
                        attempts_per_component.get(c, 0),
                        consecutive_same_class_attempts.get(c, 0),
                        c,
                    ),
                )
            return min(
                under_cap,
                key=lambda c: (
                    attempts_per_component.get(c, 0),
                    consecutive_same_class_attempts.get(c, 0),
                    c,
                ),
            )

        iter_target_component: Optional[str] = _pick_target()

        if _is_at_cap(iter_target_component):
            at_cap = [c for c in candidate_pool if _is_at_cap(c)]
            print(
                f"  All {len(at_cap)} remaining candidate(s) hit the "
                f"{max_attempts_per_component}-consec-same-class cap (hard "
                f"total ceiling {hard_total_attempt_cap}). Marking each as "
                f"CPU-fallback and restarting target selection."
            )
            for c in at_cap:
                _skip_component_to_fallback(c, "exhausted per-component attempt cap during target pick")
            last_failed_components = []
            last_failed_tests = []
            last_failures = ""
            last_failure_details = ""
            continue

        _prev_iter_failed: Set[str] = set(last_failed_components or [])

        last_failed_components = [iter_target_component]
        target_safe = _safe_id(iter_target_component)
        target_test_path = demo_dir / "tests" / "pcc" / f"test_{target_safe}.py"
        if target_test_path.is_file():
            target_rel = str(safe_relative_to_root(target_test_path))
            last_failed_tests = [target_rel]

        _target_known_broken = iter_target_component in verified_fail or iter_target_component in _prev_iter_failed

        failure_class: str = ""
        _target_partial_cpu_info: Dict[str, List[str]] = (
            _runtime_fallback_details(MODEL, iter_target_component)
            if (iter_target_component in partial_cpu_set and not _target_known_broken)
            else {}
        )
        if _target_partial_cpu_info:
            failure_class = "PARTIAL_CPU_FALLBACK"
            _helpers_list = _target_partial_cpu_info.get("helpers") or []
            _kinds_list = _target_partial_cpu_info.get("kinds") or []
            _helper_lines: List[str] = []
            for _idx, _h in enumerate(_helpers_list):
                _k = _kinds_list[_idx] if _idx < len(_kinds_list) else "(unknown op)"
                _helper_lines.append(f"    - `{_h}`  (wraps {_k})")
            last_failures = (
                f"PARTIAL-CPU FALLBACK in `{iter_target_component}`: the "
                f"stub passes PCC (>= 0.99) but "
                f"{len(_helpers_list)} `_apply_*` helper(s) inside "
                f"`__call__` still fall back to a PyTorch CPU "
                f"implementation. Replace those helpers' internals with "
                f"pure ttnn ops WITHOUT touching the rest of `__call__` "
                f"and WITHOUT regressing PCC."
            )
            last_failure_details = (
                "PARTIAL-CPU FALLBACK DETAILS:\n"
                "  The following helpers, instrumented by the planner, "
                "called `ttnn.to_torch(...)` -> torch op -> "
                "`ttnn.from_torch(...)` at least once during the most "
                "recent pytest run:\n"
                + ("\n".join(_helper_lines) if _helper_lines else "    (no helpers listed)")
                + "\n\n"
                "Surgically replace each helper's body with a pure ttnn "
                "implementation of the same op kind. Keep the helper's "
                "name, arguments, and return type/shape/dtype UNCHANGED "
                "(everything else in `__call__` calls into these helpers "
                "and works correctly today). PCC must still pass after "
                "the rewrite — if you can't keep PCC, write nothing for "
                "this iteration."
            )

        if not last_failures:
            last_failures = (
                f"Targeting `{iter_target_component}`: stub is currently on torch "
                f"CPU fallback (or a partial fake-native wrapper that still "
                f"resolves the torch submodule at build time). Replace with a "
                f"real native ttnn implementation that holds PCC >= 0.99."
            )
        if not last_failure_details:
            stub_rel = safe_relative_to_root(demo_dir / "_stubs" / f"{_safe_id(iter_target_component)}.py")
            last_failure_details = (
                f"`{stub_rel}` either runs the HF reference module on host CPU "
                f"or contains an LLM-generated wrapper that still imports "
                f"`transformers` and calls `_get_torch_submodule()`. Either way, "
                f"no real ttnn ops are exercised on the device for this component. "
                f"The PCC test passes trivially (torch vs torch). The job for this "
                f"iteration is to write a forward path made of actual ttnn ops."
            )

        attempts_per_component[iter_target_component] = attempts_per_component.get(iter_target_component, 0) + 1
        attempt_n = attempts_per_component[iter_target_component]

        banner(
            f"AUTO-ITERATE {it}/{max_iters} for {MODEL}: "
            f"target=`{iter_target_component}` "
            f"(attempt {attempt_n}/{max_attempts_per_component}, "
            f"{len(candidate_pool)} candidate(s) remaining: "
            f"{', '.join(candidate_pool[:8])}"
            f"{'…' if len(candidate_pool) > 8 else ''}) "
            f"-> invoke {provider} -> apply -> pytest"
        )

        handoff_argv = argparse.Namespace(
            model_id=MODEL,
            next=False,
            component=None,
            autofill=False,
            overwrite_autofill=False,
            run_tests=False,
            no_emit_tests=False,
            overwrite_tests=False,
            keep_passing_stubs=True,
            format="text",
            synthesize=False,
            synthesize_component=iter_target_component,
            llm_provider=None,
            llm_model=None,
            llm_endpoint=None,
            llm_max_retries=2,
            llm_dry_run=False,
            no_fetch_upstream=False,
            emit_prompts=False,
            apply_response=None,
            handoff_to_chat=True,
            apply_all_responses=False,
            list_synth_targets=False,
            quiet_handoff=True,
        )
        try:
            # Use the per-component dispatcher directly. The
            # ``cmd_bringup`` imported from ..cli is the local
            # brain-orchestrated wrapper that re-enters cmd_up — calling
            # it here would re-run the whole 6-step pipeline inside the
            # iterate loop and cause cmd_up recursion.
            from ..commands.bringup import cmd_bringup as _cmd_bringup_per_component

            _cmd_bringup_per_component(handoff_argv)
        except Exception as exc:
            print(f"  handoff emission failed: {exc}", file=sys.stderr)
            return 2

        handoff = _find_handoff_path(demo_dir)
        if handoff is None:
            print(f"  no handoff file produced under {demo_dir}/_handoff/", file=sys.stderr)
            return 2

        if failure_class != "PARTIAL_CPU_FALLBACK":
            failure_class = _classify_failure(last_failures, last_failure_details)
        signature = _failure_signature(last_failures, last_failure_details)
        strategy_directive = _strategy_directive_for_failure(failure_class, strict_native=strict_native)
        repeated_components: List[str] = []
        skip_agent_patch = False
        if last_failed_components:
            for comp in set(last_failed_components):
                k = f"{comp}|{failure_class}|{signature}"
                if repeat_error_counts.get(k, 0) >= 2:
                    repeated_components.append(comp)
        if failure_class == "L1_OOM" and repeated_components:
            banner("AUTO-ITERATE: repeated L1_OOM detected, applying deterministic fallback rewrite")
            rewritten = _rewrite_components_to_stable_fallback(demo_dir, sorted(set(repeated_components)))
            if rewritten:
                print(f"  Applied deterministic fallback to: {', '.join(rewritten)}")
                skip_agent_patch = True
            else:
                print("  No components rewritten; continuing with agent patch attempt.")

        focused_stub_excerpts: List[str] = []
        any_wrapper_seen = False
        for comp in last_failed_components or []:
            safe = _safe_id(comp)
            stub_path = demo_dir / "_stubs" / f"{safe}.py"
            if _stub_uses_torch_wrapper(stub_path):
                any_wrapper_seen = True
                body = _stub_forward_body_excerpt(stub_path, max_lines=14)
                focused_stub_excerpts.append(f"  # in {stub_path}\n{body}")

        native_directive = _native_directive(
            forbidden_excerpt="\n\n".join(focused_stub_excerpts) if any_wrapper_seen else "",
            strict_native=strict_native,
        )

        _clear_responses_dir(demo_dir)
        if last_failed_components:
            agent_targets = list(last_failed_components)
        elif target_components:
            agent_targets = list(target_components)
        else:
            agent_targets = list(_classify_components(MODEL).get("new_fallback", []))

        latest_report = (
            _scope_report_to_demo(_parse_pytest_report(), demo_dir)
            if (last_failed_tests or last_failed_components)
            else {}
        )
        per_component_failures_struct = (
            latest_report.get("per_component", {}) if isinstance(latest_report, dict) else {}
        )
        if not isinstance(per_component_failures_struct, dict):
            per_component_failures_struct = {}
        per_comp_failure: Dict[str, str] = {}
        for comp in set(list(last_failed_components or []) + list(per_component_failures_struct.keys())):
            block = _format_failure_block_for_component(per_component_failures_struct, comp)
            if block:
                per_comp_failure[comp] = block

        iter_choices: Dict[str, Dict[str, str]] = {}

        def _build_enriched_component_block(comp: str) -> str:
            """Build the full per-component enriched prompt block.

            Both the primary target and every parallel-extra target call this
            so they get the SAME convergence helpers: captured I/O contract,
            activation_diff localization, exemplar, hf source, test source,
            constraints, op-synth, attempt history.

            Before this refactor, only the primary's block was built, and
            extras' prompts inherited the primary's block as `components_block`
            instead of getting their own enrichment. That made extras converge
            far worse than primaries on the same model run.
            """
            safe = _safe_id(comp)
            stub_path = demo_dir / "_stubs" / f"{safe}.py"
            response_path = demo_dir / "_synth_responses" / f"{safe}.py"
            stub_rel = safe_relative_to_root(stub_path) if stub_path.is_absolute() else stub_path
            resp_rel = safe_relative_to_root(response_path) if response_path.is_absolute() else response_path
            stub_src = _stub_source_excerpt(stub_path, max_lines=140)
            torch_ref = _torch_ref_summary(stub_path)
            failure_block = per_comp_failure.get(comp, "(no prior failure recorded for this component)")

            meta = _component_metadata(demo_dir, comp) or {}
            kind = str(meta.get("kind", ""))
            plan_status = str(meta.get("status", "?"))
            reuse_target = meta.get("tt_reuse_target")
            plan_notes = str(meta.get("notes", "")).strip()

            exemplar_source = "(none)"
            if reuse_target and isinstance(reuse_target, str):
                ex_path = BRINGUP_ROOT() / reuse_target
                ex_src = _read_file_excerpt(ex_path, max_lines=140)
                if ex_src:
                    exemplar = (
                        f"  source (from BRING_UP_PLAN as `tt_reuse_target`): {reuse_target}\n"
                        f"```python\n{ex_src}\n```"
                    )
                    exemplar_source = reuse_target
                else:
                    exemplar = _exemplar_block(comp, kind, demo_dir=demo_dir)
                    found = _find_exemplar(comp, kind, demo_dir=demo_dir)
                    if found is not None:
                        try:
                            exemplar_source = str(safe_relative_to_root(found))
                        except Exception:
                            exemplar_source = str(found)
            else:
                exemplar = _exemplar_block(comp, kind, demo_dir=demo_dir)
                found = _find_exemplar(comp, kind, demo_dir=demo_dir)
                if found is not None:
                    try:
                        exemplar_source = str(safe_relative_to_root(found))
                    except Exception:
                        exemplar_source = str(found)

            iter_choices[comp] = {
                "exemplar_used": exemplar_source,
                "model_used": model,
            }
            print(f"  [exemplar] {comp}: source = {exemplar_source}")

            history = _load_attempt_history(demo_dir, comp, max_entries=3)
            history_block = _format_attempt_history_block(history)

            plan_header = f"plan classification : {plan_status}  (kind={kind or 'unknown'})\n" + (
                f"plan note           : {plan_notes}\n" if plan_notes else ""
            )

            test_source_block = _read_test_source(demo_dir, comp, max_lines=80)
            captured_shape_block = _format_captured_shape_contract_block(demo_dir, comp)
            constraints_block = _numerical_constraints_block(
                stub_path,
                model_id=MODEL,
                demo_dir=demo_dir,
                component_name=comp,
            )

            op_synth_block, op_synth_contract = _op_synth_prompt_blocks(demo_dir, comp)

            localization_hint = ""
            full_hf_source = ""
            _localize_classes = ("PCC_ONLY", "DTYPE_MISMATCH", "SHAPE", "TT_FATAL_OPAQUE")
            _crash_in_failure = bool(failure_block) and any(
                _sig in failure_block
                for _sig in ("RuntimeError:", "AttributeError:", "IndexError:", "Traceback (most recent call")
            )
            if failure_class in _localize_classes or (failure_class == "OTHER" and _crash_in_failure):
                try:
                    from .. import activation_diff as _act_diff

                    _loc_result = _act_diff.localize_pcc_divergence(demo_dir, comp, device=None)
                    localization_hint = _act_diff.format_localization_hint_block(comp, _loc_result)
                except Exception as _exc:
                    localization_hint = ""
            if failure_class == "PCC_ONLY":
                try:
                    full_hf_source = _full_hf_reference_source(
                        stub_path,
                        model_id=MODEL,
                        demo_dir=demo_dir,
                        component_name=comp,
                    )
                except Exception:
                    full_hf_source = ""

            extra_pcc_blocks = ""
            if localization_hint:
                extra_pcc_blocks += (
                    f"\n--- LOCALIZATION HINT (PCC_ONLY: per-helper torch reference trace) ---\n"
                    f"{localization_hint}\n"
                )
            if full_hf_source:
                extra_pcc_blocks += (
                    f"\n--- FULL HF REFERENCE SOURCE (PCC_ONLY: the COMPLETE reference class) ---\n"
                    f"{full_hf_source}\n"
                )
            print(
                f"  [prompt-block] {comp}: "
                f"captured_shape={'NON-EMPTY' if captured_shape_block else 'EMPTY'}"
                f"({len(captured_shape_block)})  "
                f"localization={'NON-EMPTY' if localization_hint else 'EMPTY'}"
                f"({len(localization_hint)})  "
                f"hf_source={'NON-EMPTY' if full_hf_source else 'EMPTY'}"
                f"({len(full_hf_source)})"
            )

            return (
                f"================================================================\n"
                f"COMPONENT: {comp}\n"
                f"================================================================\n"
                f"{plan_header}"
                f"write your answer to: {resp_rel}\n"
                f"  (full module contents; tool overwrites {stub_rel} and runs pytest.)\n"
                f"{op_synth_contract}"
                f"\n"
                f"--- WHAT FAILED IN THE LAST ITERATION FOR THIS COMPONENT ---\n"
                f"{failure_block}\n"
                f"\n"
                f"--- HANDOFF FROM PRIOR ITERATIONS (what was tried, what failed, what to try next) ---\n"
                f"{history_block}\n"
                f"\n"
                f"--- CAPTURED I/O CONTRACT (real shapes from HF forward — match exactly) ---\n"
                f"{captured_shape_block if captured_shape_block else '(no capture — infer shapes from torch reference)'}\n"
                f"\n"
                f"--- NUMERICAL HARDWARE CONSTRAINTS (act on these BEFORE writing the stub) ---\n"
                f"{constraints_block}\n"
                f"\n"
                f"--- ASSERTION CONTRACT (the actual pytest test that validates this component) ---\n"
                f"{test_source_block}\n"
                f"\n"
                f"--- CURRENT STUB ({stub_rel}) ---\n"
                f"```python\n{stub_src}\n```\n"
                f"{op_synth_block}"
                f"\n"
                f"--- TORCH REFERENCE (resolved by `_get_torch_submodule()` in the stub) ---\n"
                f"{torch_ref}\n"
                f"{extra_pcc_blocks}"
                f"\n"
                f"--- EXEMPLAR: an already-working ttnn module with a similar role ---\n"
                f"{exemplar}\n"
            )

        component_blocks: List[str] = [_build_enriched_component_block(comp) for comp in agent_targets]
        components_block = "\n".join(component_blocks) if component_blocks else "(no failing components)"

        failure_context = ""
        if last_failures:
            failure_context = f"PREVIOUS OUTER ITERATION SUMMARY:\n{last_failures}\n\n"

        _iter_complexity_bonus_for_budget = (
            _component_complexity_bonus(iter_target_component) if iter_target_component else 0
        )
        _effective_budget_s_for_prompt = (
            _agent_complexity_timeout(agent_timeout_s, _iter_complexity_bonus_for_budget) if agent_timeout_s > 0 else 0
        )
        budget_min = max(1, _effective_budget_s_for_prompt // 60) if _effective_budget_s_for_prompt > 0 else 0

        deliverable_deadline_min = max(1, int(budget_min * 0.8)) if budget_min > 0 else 0
        budget_clause = (
            f"WALL-CLOCK BUDGET: ~{budget_min} minutes for this invocation. "
            f"Spend it writing files, not running tools. Pytest takes 30-60s per "
            f"call on TT hardware — DO NOT run it yourself.\n"
            f"DELIVERABLE DEADLINE: You MUST write at least one response file to "
            f"`_synth_responses/<safe>.py` within the FIRST "
            f"{deliverable_deadline_min} minutes. The tool monitors this directory "
            f"and will KILL you at the {deliverable_deadline_min}-minute mark if "
            f"nothing has appeared, regardless of how much investigation is in "
            f"progress. If you run out of time, write a PARTIAL stub (with `pass` "
            f"placeholders for unfinished ops) rather than writing nothing — a "
            f"partial stub keeps the loop moving; an unwritten file wastes the "
            f"whole iteration.\n\n"
            if agent_timeout_s > 0
            else ""
        )

        hw_header = (
            f"HARDWARE / TARGET CONTEXT (apply to every component below):\n"
            f"  model        : {MODEL}\n"
            f"  box          : {BOX}\n"
            f"  mesh         : {mesh or '(planner default)'}\n"
            f"  dtype        : {dtype or 'bfloat16 (default)'}\n"
            f"  device path  : weights -> ttnn.from_torch(..., layout=ttnn.TILE_LAYOUT, "
            f"dtype=ttnn.bfloat16) onto `device` passed into `build(device, torch_module)`\n"
            f"  test harness : pytest under `models/demos/.../tests/pcc/test_<comp>.py`; "
            f"the test resolves the torch ref via `_get_torch_submodule()` and "
            f"compares your ttnn output against torch via `comp_pcc` (target >= 0.99)\n\n"
        )

        all_op_synth = bool(last_failed_components) and all(
            _op_synth_manifest(demo_dir, c) is not None for c in last_failed_components
        )
        if all_op_synth:
            task_block = (
                f"TASK: This iteration's target component(s) ship as PARTIAL TTNN stubs.\n"
                f"Weights are already loaded inside `__init__` as `self.w_*` ttnn tensors,\n"
                f"and the deterministic `_apply_*` helpers (one per op-REUSE / op-ADAPT\n"
                f"leaf) are already correct. Your ONLY job is to replace the body of\n"
                f"`__call__` with a pure-ttnn forward pass that wires those helpers\n"
                f"together. For each component below:\n"
                f"  1. Read the CURRENT STUB section.\n"
                f"  2. Write the response file at the path shown for that component\n"
                f"     by copying the existing stub VERBATIM and replacing ONLY the\n"
                f"     body of `__call__`. Same imports, same class name, same\n"
                f"     `__init__`, same `_apply_*` helpers, same `build()`, same\n"
                f"     module-level shim, same `_LLM_GAPS`.\n"
                f"  3. Implement the new `__call__` using `self._apply_*` helpers\n"
                f"     (listed in OP-SYNTH PALETTE) plus raw ttnn ops (reshape,\n"
                f"     transpose, permute, matmul, softmax, scaled_dot_product_attention,\n"
                f"     etc.). For each op-NEW gap, either inline it in `__call__` or\n"
                f"     add a private `_apply_<name>` helper. Do NOT call\n"
                f"     `self._torch_module(...)` or `_coerce_to_torch(...)` inside\n"
                f"     the new `__call__` — the forward path must be pure ttnn.\n"
                f"\n"
                f"HARD CONSTRAINTS:\n"
                f"  - A response that is byte-identical to the existing stub is\n"
                f"    treated as a NO-OP and counted as a failed attempt. You MUST\n"
                f"    actually edit the `__call__` body. The starting stub falls\n"
                f"    back to `self._torch_module(...)`; that is the body you must\n"
                f"    replace.\n"
                f"  - DO NOT run pytest, DO NOT iterate. The outer tool runs pytest\n"
                f"    after you exit and feeds the traceback back next round.\n\n"
            )
        else:
            task_block = (
                f"TASK: For each component below, write the COMPLETE new contents of "
                f"`_synth_responses/<safe>.py` (path is given in each component block). "
                f"You are writing ONE file per component, that is ALL. The outer tool "
                f"will: (a) copy your response over the current stub, (b) run pytest, "
                f"(c) re-invoke you with the actual pytest traceback for any component "
                f"that still fails.\n\n"
                f"DO NOT run pytest, DO NOT iterate. The current stub source, the "
                f"resolved torch reference's `forward` source, its `state_dict()` "
                f"shapes, an EXEMPLAR ttnn module with a similar role, the BRING_UP_PLAN "
                f"classification (REUSE/ADAPT/NEW), and any prior failure trace are "
                f"ALREADY INLINED below. Use the EXEMPLAR as your template for sharding, "
                f"memory configs, dtype choices, and `build()` weight-loading patterns.\n\n"
                f"OUTPUT CONTRACT per component:\n"
                f"  - One file at `_synth_responses/<safe>.py` containing the full module.\n"
                f"  - Must implement the same class + `build(device, torch_module)` + "
                f"module-level shim that the current stub exposes (the class/function "
                f"names and signatures are visible in the CURRENT STUB section).\n"
                f"  - `__call__` / `forward` must compute with `ttnn.*` ops on the "
                f"device (no torch wrapper, no `_get_torch_submodule(...)` in the "
                f"forward path).\n"
                f"  - Use `_torch_module.state_dict()` shapes shown in TORCH REFERENCE "
                f"to size `ttnn.from_torch(...)` inside `build()`.\n\n"
            )

        escalated_scope_block = _format_escalated_edit_scope_block(demo_dir, failure_class)

        systemic_block = ""
        has_systemic_for_class = False
        for sk_class, sk_sig in sorted(systemic_pattern_seen):
            if sk_class == failure_class:
                systemic_block = _format_systemic_pattern_block(sk_class, sk_sig)
                has_systemic_for_class = True
                break

        shape_probe_block = _format_shape_probe_block(last_shape_probes)

        max_consec_for_iter = 0
        for comp_for_consec in last_failed_components or []:
            max_consec_for_iter = max(
                max_consec_for_iter,
                consecutive_same_class_attempts.get(comp_for_consec, 0),
            )
        agentic_block = _format_agentic_affordances_block(
            failure_class,
            consec_count=max_consec_for_iter,
            has_systemic_pattern=has_systemic_for_class,
        )

        cross_component_block = _build_cross_component_context_block(
            demo_dir,
            current_target=iter_target_component,
            attempts_per_component=attempts_per_component,
            last_failure_class_per_component=last_failure_class_per_component,
        )

        # Constraint catalog block — computed via the shared helper so
        # the primary-target prompt and the parallel-extras prompts use
        # identical logic.
        constraint_block = ""
        if iter_target_component:
            try:
                from .iter_prompt import build_constraint_block

                constraint_block = build_constraint_block(
                    demo_dir=demo_dir,
                    target_component=iter_target_component,
                )
            except Exception as _hint_exc:
                print(
                    f"  [constraint-catalog] non-fatal: " f"{type(_hint_exc).__name__}: {_hint_exc}",
                    file=sys.stderr,
                )

        # G8 (agentic.convergence) + G4 (agentic.actions): if this
        # component's PCC history is stagnant (plateau detected), apply
        # ONE untried mechanical action BEFORE invoking the LLM again.
        # Each action is a cheap toggle (cache invalidate, env var, etc.);
        # if none move the needle, we fall through to the LLM iteration.
        # Direct reuse of the existing engine — no logic duplicated here.
        if iter_target_component:
            try:
                from ..agentic.actions import default_mechanical_actions
                from ..agentic.convergence import is_stagnant

                _hist = pcc_history_per_component.get(iter_target_component) or []
                if len(_hist) >= 2 and is_stagnant(_hist):
                    _tried = tried_actions_per_component.setdefault(iter_target_component, set())
                    _actions = default_mechanical_actions(
                        model_id=MODEL,
                        workspace_root=demo_dir,
                    )
                    for _action in _actions:
                        _name = getattr(_action, "name", "") or _action.__class__.__name__
                        if _name in _tried:
                            continue
                        _tried.add(_name)
                        try:
                            _result = _action.apply({})
                            _applied = bool(getattr(_result, "applied", False))
                            _notes = "; ".join(getattr(_result, "notes", []) or [])
                        except Exception as _aexc:
                            _applied = False
                            _notes = f"raised {type(_aexc).__name__}: {_aexc}"
                        print(
                            f"  [agentic:G4] PCC plateau on `{iter_target_component}` "
                            f"(history={[round(p, 4) for p in _hist[-3:]]}); "
                            f"applied mechanical action `{_name}`: "
                            f"applied={_applied}; {_notes}"
                        )
                        # One action per plateau-detection; the next pytest
                        # run will reveal whether the toggle moved PCC.
                        break
            except Exception as _g8_exc:
                print(
                    f"  [agentic:G4+G8] non-fatal: " f"{type(_g8_exc).__name__}: {_g8_exc}",
                    file=sys.stderr,
                )

        # Investigative-mode preamble: when this component's PCC has
        # plateaued (G8 is_stagnant + history >= 2), prepend an explicit
        # OVERRIDE of the default "DO NOT iterate" task_block. Tells the
        # LLM to use Read/Edit/Bash/Write freely + write probe scripts
        # + make targeted edits instead of one-shot file rewrites.
        # Pattern mirrors _build_forced_edit_preamble.
        investigative_preamble = ""
        if iter_target_component:
            try:
                from ..agentic.convergence import is_stagnant
                from ..cli import _build_investigative_mode_preamble

                _hist = pcc_history_per_component.get(iter_target_component) or []
                if len(_hist) >= 2 and is_stagnant(_hist):
                    # Convert mismatch_ratio history (1-pcc) back to
                    # PCC values for the preamble's display.
                    _pcc_seq = [1.0 - m for m in _hist]
                    investigative_preamble = _build_investigative_mode_preamble(
                        iter_idx=it,
                        component=iter_target_component,
                        pcc_history=_pcc_seq,
                    )
                    print(
                        f"  [investigative-mode] `{iter_target_component}` "
                        f"plateaued ({_pcc_seq[-3:]}); next iter prompt "
                        f"includes the INVESTIGATIVE MODE preamble."
                    )
            except Exception as _inv_exc:
                print(
                    f"  [investigative-mode] non-fatal: " f"{type(_inv_exc).__name__}: {_inv_exc}",
                    file=sys.stderr,
                )

        prompt = (
            f"{investigative_preamble}"
            f"{hw_header}"
            f"{task_block}"
            f"{systemic_block}"
            f"{shape_probe_block}"
            f"{agentic_block}"
            f"{budget_clause}"
            f"{constraint_block}"
            f"{failure_context}"
            f"STRATEGY DIRECTIVE FOR THIS ITERATION:\n{strategy_directive}\n"
            f"{escalated_scope_block}"
            f"{native_directive}\n"
            f"{cross_component_block}"
            f"COMPONENTS:\n{components_block}\n"
        )
        pre_apply_state: Dict[str, bool] = {}
        pre_apply_hashes: Dict[str, str] = {}
        for comp in last_failed_components or []:
            safe = _safe_id(comp)
            stub_path = demo_dir / "_stubs" / f"{safe}.py"
            pre_apply_state[comp] = _stub_uses_torch_wrapper(stub_path)
            try:
                pre_apply_hashes[comp] = hashlib.sha1(stub_path.read_bytes()).hexdigest() if stub_path.is_file() else ""
            except Exception:
                pre_apply_hashes[comp] = ""

            _snapshot_preiter_native_stub(comp)

        agent_produced_any = False
        if not skip_agent_patch:
            _iter_complexity_bonus = _component_complexity_bonus(iter_target_component) if iter_target_component else 0
            _iter_attempts_so_far = attempts_per_component.get(iter_target_component, 0) if iter_target_component else 0
            _iter_model, _iter_model_reason = _pick_agent_model_for_iter(
                model_default=model,
                model_light=model_light,
                model_heavy=model_heavy,
                complexity_bonus=_iter_complexity_bonus,
                failure_class=failure_class,
                attempts_so_far=_iter_attempts_so_far,
            )
            if (model_light or model_heavy) and _iter_model_reason != "default":
                print(
                    f"  [auto:{provider}] tiered model pick: {_iter_model} "
                    f"({_iter_model_reason}; complexity=+{_iter_complexity_bonus}, "
                    f"failure_class={failure_class or 'NONE'}, "
                    f"attempts={_iter_attempts_so_far})"
                )

            _expected_targets: List[Path] = []
            _responses_dir = demo_dir / "_synth_responses"
            for _comp_for_deliv in last_failed_components or []:
                _expected_targets.append(_responses_dir / f"{_safe_id(_comp_for_deliv)}.py")
            if iter_target_component and iter_target_component not in (last_failed_components or []):
                _expected_targets.append(_responses_dir / f"{_safe_id(iter_target_component)}.py")
            from .agent import _bringup_cwd as _bcwd

            _parallel_extra_jobs = []
            if parallel_agents > 1 and iter_target_component:
                from .iter_prompt import (
                    assemble_iter_prompt,
                    build_per_target_blocks,
                    build_target_header,
                )
                from .parallel_iterate import (
                    AgentJob,
                    pick_n_distinct_targets,
                    run_parallel_agents,
                )

                _ungraduated_now, _ = _auto_iteration_blockers(MODEL)
                _extras_pool = set(_ungraduated_now) | set(candidate_pool or [])
                _extras_pool -= set(graduated_this_run)
                _exclude = set([iter_target_component]) | set(permanently_skipped)
                _at_cap_now = {c for c in _extras_pool if _is_at_cap(c)}
                _exclude |= _at_cap_now
                _ungraduated_ranked = sorted(
                    _extras_pool,
                    key=lambda c: (
                        attempts_per_component.get(c, 0),
                        consecutive_same_class_attempts.get(c, 0),
                        c,
                    ),
                )
                _extra_targets = pick_n_distinct_targets(
                    _ungraduated_ranked, n=parallel_agents - 1, exclude=list(_exclude)
                )
                if only_component:
                    _extra_targets = []
                for _extra in _extra_targets:
                    if _extra in pre_apply_hashes:
                        continue
                    _ex_safe = _safe_id(_extra)
                    _ex_stub_path = demo_dir / "_stubs" / f"{_ex_safe}.py"
                    pre_apply_state[_extra] = _stub_uses_torch_wrapper(_ex_stub_path)
                    try:
                        pre_apply_hashes[_extra] = (
                            hashlib.sha1(_ex_stub_path.read_bytes()).hexdigest() if _ex_stub_path.is_file() else ""
                        )
                    except Exception:
                        pre_apply_hashes[_extra] = ""
                    _snapshot_preiter_native_stub(_extra)
                if _at_cap_now:
                    print(f"  [parallel] skipping at-cap component(s) from extras: " f"{sorted(_at_cap_now)}")
                from .iter_prompt import build_constraint_block

                for _extra in _extra_targets:
                    # Count this scheduled spawn as a real attempt for
                    # the extra. Mirrors the primary-target bump at
                    # _run_auto_iterate_loop line ~1676. Without this,
                    # attempts_per_component[_extra] stays at whatever
                    # value the extra hit as a PRIMARY target — so
                    # _is_at_cap() never fires for extras and the
                    # per-component escalation chain (_skip_component_
                    # to_fallback → decompose-auto, etc.) never engages
                    # for them.
                    attempts_per_component[_extra] = attempts_per_component.get(_extra, 0) + 1
                    _extra_attempts = attempts_per_component[_extra]
                    _extra_blocks = build_per_target_blocks(
                        demo_dir=demo_dir,
                        target_component=_extra,
                        per_comp_failure=per_comp_failure,
                        last_failure_class_per_component=last_failure_class_per_component,
                        attempts_per_component=attempts_per_component,
                        focused_stub_excerpts=focused_stub_excerpts if any_wrapper_seen else [],
                        strict_native=strict_native,
                    )
                    _extra_target_header = build_target_header(
                        target_component=_extra,
                        attempts_so_far=_extra_attempts,
                        prior_failure_class=_extra_blocks["failure_class"],
                    )
                    _extra_components_block = _build_enriched_component_block(_extra)
                    # Per-target catalog block so each parallel agent sees
                    # its OWN shape/dtype constraint hints.
                    _extra_constraint_block = build_constraint_block(
                        demo_dir=demo_dir,
                        target_component=_extra,
                    )
                    _extra_prompt = assemble_iter_prompt(
                        hw_header=hw_header,
                        task_block=task_block,
                        systemic_block=systemic_block,
                        shape_probe_block=shape_probe_block,
                        agentic_block=agentic_block,
                        budget_clause=budget_clause,
                        failure_context=_extra_blocks["failure_context"],
                        strategy_directive=_extra_blocks["strategy_directive"],
                        escalated_scope_block=_extra_blocks["escalated_scope_block"],
                        native_directive=_extra_blocks["native_directive"],
                        cross_component_block=_extra_blocks["cross_component_block"],
                        components_block=_extra_components_block,
                        target_header=_extra_target_header,
                        constraint_block=_extra_constraint_block,
                    )
                    _parallel_extra_jobs.append(
                        AgentJob(
                            component=_extra,
                            prompt=_extra_prompt,
                            cwd=_bcwd(),
                            provider=provider,
                            agent_bin=agent_bin,
                            model=_iter_model,
                            timeout_s=agent_timeout_s,
                            complexity_bonus=_iter_complexity_bonus,
                            iter_tag=f"iter_{it}_{_safe_id(_extra)}",
                            deliverable_dirs=[_responses_dir],
                            expected_deliverable_files=[_responses_dir / f"{_safe_id(_extra)}.py"],
                        )
                    )

            if os.environ.get("TT_PLANNER_DRY_RUN_PROMPTS", "") == "1":
                _dr_dir = Path("/tmp") / "tt_planner_dry_run" / _safe_id(MODEL) / f"iter_{it}"
                _dr_dir.mkdir(parents=True, exist_ok=True)
                _written: List[Path] = []
                if iter_target_component:
                    _p = _dr_dir / f"{_safe_id(iter_target_component)}.prompt.txt"
                    _p.write_text(prompt)
                    _written.append(_p)
                for _job in _parallel_extra_jobs:
                    _p = _dr_dir / f"{_safe_id(_job.component)}.prompt.txt"
                    _p.write_text(_job.prompt)
                    _written.append(_p)
                print(
                    f"  [dry-run-prompts] wrote {len(_written)} prompt file(s) "
                    f"to {_dr_dir}; skipping LLM invocation."
                )
                for _p in _written:
                    print(f"    {_p}  ({_p.stat().st_size} bytes)")
                print(
                    "  [dry-run-prompts] TT_PLANNER_DRY_RUN_PROMPTS=1 set; "
                    "exiting after iter 1 since no agent code was applied."
                )
                return 0
            if _parallel_extra_jobs:
                from .parallel_iterate import AgentJob, run_parallel_agents

                _all_jobs = [
                    AgentJob(
                        component=iter_target_component,
                        prompt=prompt,
                        cwd=_bcwd(),
                        provider=provider,
                        agent_bin=agent_bin,
                        model=_iter_model,
                        timeout_s=agent_timeout_s,
                        complexity_bonus=_iter_complexity_bonus,
                        iter_tag=f"iter_{it}_{_safe_id(iter_target_component)}",
                        deliverable_dirs=[_responses_dir],
                        expected_deliverable_files=_expected_targets or None,
                    ),
                    *_parallel_extra_jobs,
                ]
                _par_results = run_parallel_agents(_all_jobs, max_workers=parallel_agents)
                rc = _par_results[0].rc if _par_results else 1
                for _r in _par_results[1:]:
                    if _r.rc != 0:
                        print(
                            f"  agent for {_r.component!r} returned non-zero " f"({_r.rc}); continuing to apply step.",
                            file=sys.stderr,
                        )
            else:
                rc = _invoke_agent(
                    prompt,
                    provider=provider,
                    agent_bin=agent_bin,
                    cwd=_bcwd(),
                    model=_iter_model,
                    timeout_s=agent_timeout_s,
                    complexity_bonus=_iter_complexity_bonus,
                    iter_tag=f"iter_{it}_{_safe_id(iter_target_component)}" if iter_target_component else f"iter_{it}",
                    deliverable_dirs=[_responses_dir],
                    expected_deliverable_files=_expected_targets or None,
                )
            if rc != 0:
                print(f"  agent invocation returned non-zero ({rc}); continuing to apply step.", file=sys.stderr)
            try:
                responses_dir = demo_dir / "_synth_responses"
                response_files = (
                    [p for p in responses_dir.iterdir() if p.is_file() and p.suffix == ".py"]
                    if responses_dir.is_dir()
                    else []
                )
                agent_produced_any = len(response_files) > 0
                print(f"  agent produced {len(response_files)} response file(s) in _synth_responses/")
            except Exception as exc:
                print(f"  could not list _synth_responses/: {exc}", file=sys.stderr)
                agent_produced_any = False

            banner(f"AUTO-ITERATE {it}/{max_iters}: apply _synth_responses/ -> _stubs/")
            apply_argv = argparse.Namespace(
                model_id=MODEL,
                next=False,
                component=None,
                autofill=False,
                overwrite_autofill=False,
                run_tests=False,
                no_emit_tests=False,
                overwrite_tests=False,
                keep_passing_stubs=True,
                format="text",
                synthesize=False,
                synthesize_component=None,
                llm_provider=None,
                llm_model=None,
                llm_endpoint=None,
                llm_max_retries=2,
                llm_dry_run=False,
                no_fetch_upstream=False,
                emit_prompts=False,
                apply_response=None,
                handoff_to_chat=False,
                apply_all_responses=True,
                list_synth_targets=False,
            )
            try:
                # Per-component dispatcher (not the local cli wrapper)
                # — see the handoff-call site above for the same fix.
                from ..commands.bringup import cmd_bringup as _cmd_bringup_per_component

                apply_rc = _cmd_bringup_per_component(apply_argv)
            except Exception as exc:
                print(f"  apply-all-responses failed: {exc}", file=sys.stderr)
                return 2

            if isinstance(apply_rc, int) and apply_rc not in (0, None):
                print(
                    f"  apply-all-responses returned non-zero ({apply_rc}); "
                    f"treating as agent-output-broken and recording "
                    f"per-target failure so cap accounting catches it.",
                    file=sys.stderr,
                )
                for _broken_target in last_failed_components or []:
                    _record_failure_for_component(_broken_target, "EMPTY_AGENT", None)

            try:
                regen_result = run_bringup_loop(
                    model_id=MODEL,
                    emit_tests=True,
                    run_tests=False,
                    remove_passing_stubs=False,
                    overwrite_tests=False,
                )
                upgraded = [
                    a.component
                    for a in regen_result.actions
                    if any("regenerated test from" in n for n in (a.notes or []))
                ]
                if upgraded:
                    print(
                        f"  test-regen: upgraded SMOKE -> real PCC for "
                        f"{len(upgraded)} graduated stub(s): {', '.join(upgraded)}"
                    )

                try:
                    from ..capture_inputs import upgrade_all_tests_in_demo

                    _post_regen_ups = upgrade_all_tests_in_demo(demo_dir)
                    _post_regen_n = sum(1 for _n, m in _post_regen_ups if m)
                    if _post_regen_n:
                        print(
                            f"  test-regen: re-injected captured-input "
                            f"short-circuit into {_post_regen_n} test "
                            f"file(s) after SMOKE -> PCC regen."
                        )
                except Exception as exc:
                    print(
                        f"  test-regen post-upgrade failed (non-fatal): {exc}",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"  test-regen failed (non-fatal): {exc}", file=sys.stderr)

            regressed: List[str] = []
            if strict_native:
                for comp, was_wrapper_before in pre_apply_state.items():
                    safe = _safe_id(comp)
                    stub_path = demo_dir / "_stubs" / f"{safe}.py"
                    bak_path = stub_path.with_suffix(".py.bak")
                    preiter_path = stub_path.with_suffix(".py.preiter_native")
                    last_good_path = stub_path.with_suffix(".py.last_good_native")
                    is_wrapper_now = _stub_uses_torch_wrapper(stub_path)
                    if is_wrapper_now and not was_wrapper_before:
                        candidates = []
                        if last_good_path.is_file() and not _stub_uses_torch_wrapper(last_good_path):
                            candidates.append(("last_good_native", last_good_path))
                        if preiter_path.is_file() and not _stub_uses_torch_wrapper(preiter_path):
                            candidates.append(("preiter_native", preiter_path))
                        if bak_path.is_file() and not _stub_uses_torch_wrapper(bak_path):
                            candidates.append(("bak", bak_path))
                        if candidates:
                            label, src_path = candidates[0]
                            try:
                                stub_path.write_text(src_path.read_text())
                                regressed.append(comp)
                                print(
                                    f"  REGRESSION GUARD: agent rewrote `{comp}` as a "
                                    f"CPU-fallback wrapper, but the previous version was "
                                    f"native ttnn. Restored from {src_path} ({label})."
                                )
                            except Exception as exc:
                                print(f"  regression restore failed for {comp}: {exc}", file=sys.stderr)
            if regressed:
                banner(
                    f"AUTO-ITERATE {it}/{max_iters}: rejected {len(regressed)} regression(s); "
                    f"skipping pytest and re-prompting agent"
                )
                last_failures = (
                    f"Iter {it}: agent regressed {', '.join(regressed)} from a native "
                    f"ttnn stub back to a CPU-fallback wrapper. The previous native "
                    f"version was restored from `.bak`. Rewrite the failing path with "
                    f"native ttnn ops; do NOT delegate to `self._torch_module(...)`."
                )
                last_failure_details = (
                    "Components for which the loop restored the previous version "
                    f"because the agent's rewrite matched a forbidden wrapper pattern "
                    f"({', '.join(_TORCH_WRAPPER_PATTERNS)}):\n  - " + "\n  - ".join(regressed)
                )
                last_failed_components = list(regressed)

                for regressed_comp in regressed:
                    _record_failure_for_component(regressed_comp, "WRAPPER", None)
                continue

        stub_changed_any = False
        unchanged_targets: List[str] = []
        for comp, before_hash in pre_apply_hashes.items():
            safe = _safe_id(comp)
            stub_path = demo_dir / "_stubs" / f"{safe}.py"
            try:
                after_hash = hashlib.sha1(stub_path.read_bytes()).hexdigest() if stub_path.is_file() else ""
            except Exception:
                after_hash = ""
            if after_hash and after_hash != before_hash:
                stub_changed_any = True
            elif before_hash and after_hash == before_hash:
                unchanged_targets.append(comp)

        op_synth_unchanged: List[str] = [
            c
            for c in unchanged_targets
            if _op_synth_manifest(demo_dir, c) is not None
            and _stub_uses_torch_wrapper(demo_dir / "_stubs" / f"{_safe_id(c)}.py")
        ]
        if (
            not skip_agent_patch
            and op_synth_unchanged
            and iter_target_component in op_synth_unchanged
            and not stub_changed_any
            and it < max_iters
        ):
            banner(
                f"AUTO-ITERATE {it}/{max_iters}: agent wrote byte-identical output "
                f"for op-synth target(s); skipping pytest and re-prompting"
            )
            print(
                f"  no-op detection: agent produced response file(s) but the stub "
                f"hash did not change for {', '.join(op_synth_unchanged)}. The "
                f"existing `__call__` body falls back to `self._torch_module(...)` "
                f"and pytest would pass trivially via torch==torch — counting this "
                f"as a failed attempt without running pytest."
            )
            last_failures = (
                f"Iter {it}: agent wrote a byte-identical response for "
                f"{', '.join(op_synth_unchanged)}. The op-synth partial stub on "
                f"disk still has the autofill `__call__` that delegates to "
                f"`self._torch_module(...)`. You MUST replace the body of "
                f"`__call__` with pure ttnn ops using the `_apply_*` helpers "
                f"in the OP-SYNTH PALETTE. Do not write the file back unchanged."
            )
            last_failure_details = (
                "The following op-synth target(s) were written back to "
                "`_stubs/` byte-for-byte identical to their pre-apply state, "
                "meaning `__call__` still contains:\n"
                "    args = tuple(_coerce_to_torch(a) for a in args)\n"
                "    kwargs = {k: _coerce_to_torch(v) for k, v in kwargs.items()}\n"
                "    return self._torch_module(*args, **kwargs)\n"
                "Components affected:\n  - " + "\n  - ".join(op_synth_unchanged) + "\n"
                "Required action: rewrite the body of `__call__` ONLY, using "
                "`self._apply_*` helpers + raw ttnn ops. Keep everything else."
            )
            last_failed_components = list(op_synth_unchanged)

            for comp in op_synth_unchanged:
                prev_class_for_log = last_failure_class_per_component.get(comp, "")

                _was_progress_log = _record_failure_for_component(comp, "NO_OP", None)
                if comp == iter_target_component and _was_progress_log and prev_class_for_log != "NO_OP":
                    print(
                        f"  PROGRESS: `{comp}` shifted (class NO_OP); "
                        f"consecutive-same-class counter reset to "
                        f"1/{max_attempts_per_component}."
                    )
                _consec_noop = consecutive_same_class_attempts.get(comp, 0)
                if (
                    _consec_noop >= 2
                    and last_failure_class_per_component.get(comp) == "NO_OP"
                    and comp not in permanently_skipped
                    and comp not in graduated_this_run
                ):
                    print(
                        f"  [no-op escalation] `{comp}` produced byte-identical "
                        f"responses {_consec_noop} iter(s) in a row — agent is "
                        f"persistently failing to engage with this component. "
                        f"Escalating to CPU fallback (rather than burning more "
                        f"iters on the same NO_OP pattern); hand-fix or "
                        f"decompose this component before the next run."
                    )
                    _skip_component_to_fallback(
                        comp, f"NO_OP escalation: {_consec_noop} consecutive byte-identical responses"
                    )
            continue

        if not skip_agent_patch and not agent_produced_any and not stub_changed_any:
            consecutive_empty_agent_calls += 1
            print(
                f"  agent produced zero usable response files and no stub was modified. "
                f"Consecutive empty agent calls: {consecutive_empty_agent_calls}/{max_consecutive_empty}."
            )

            if iter_target_component is not None:
                prev_class_for_log = last_failure_class_per_component.get(iter_target_component, "")
                _was_progress_log = _record_failure_for_component(iter_target_component, "EMPTY_AGENT", None)
                if _was_progress_log and prev_class_for_log != "EMPTY_AGENT":
                    print(
                        f"  PROGRESS: `{iter_target_component}` shifted "
                        f"(class EMPTY_AGENT); consecutive-same-class "
                        f"counter reset to 1/{max_attempts_per_component}."
                    )
            if (
                iter_target_component is not None
                and iter_target_component not in graduated_this_run
                and iter_target_component not in permanently_skipped
                and _is_at_cap(iter_target_component)
            ):
                print(
                    f"  Agent produced no code for `{iter_target_component}` and the "
                    f"per-component attempt cap is exhausted — leaving it on CPU "
                    f"fallback and moving on."
                )
                _skip_component_to_fallback(
                    iter_target_component,
                    "agent produced no code (likely component too complex/long for "
                    "the LLM within the wall-clock budget)",
                )
                last_failed_components = [c for c in (last_failed_components or []) if c != iter_target_component]
                if not last_failed_components:
                    last_failed_tests = []
                    last_failures = ""
                    last_failure_details = ""
            if consecutive_empty_agent_calls >= max_consecutive_empty:
                banner(
                    f"AUTO-ITERATE: agent invocation produced no files for "
                    f"{max_consecutive_empty} different attempt(s) in a row — "
                    f"breaking out of the LLM loop and falling back to deterministic "
                    f"stable stubs for whatever remains"
                )
                print(
                    "  Possible root causes:\n"
                    "    - components too complex for the LLM to port in one call\n"
                    "      (the scaffolder may need to emit smaller atomic units)\n"
                    "    - credentials not set (ANTHROPIC_API_KEY / cursor login)\n"
                    "    - wrong --auto-model name for the installed agent CLI\n"
                    "    - CLI version mismatch with the flags this tool passes\n"
                    "  Investigate by re-running the same prompt manually:\n"
                    f"    cat {safe_relative_to_root(demo_dir)}/_handoff/*__handoff.md | {agent_bin} -p --model {model}",
                    file=sys.stderr,
                )
                break

            continue
        else:
            if agent_produced_any or stub_changed_any:
                consecutive_empty_agent_calls = 0

        still_wrappers = []
        if strict_native:
            for comp in last_failed_components or []:
                safe = _safe_id(comp)
                stub_path = demo_dir / "_stubs" / f"{safe}.py"
                if _stub_uses_torch_wrapper(stub_path):
                    still_wrappers.append(comp)
        if still_wrappers and it < max_iters:
            banner(
                f"AUTO-ITERATE {it}/{max_iters}: agent wrote torch wrapper(s) again — "
                f"skipping pytest, re-prompting with explicit forbidden patterns"
            )
            print(f"  components still wrapping torch on CPU: {', '.join(still_wrappers)}")
            breakdown = _ungraduated_breakdown(demo_dir, still_wrappers)
            if breakdown:
                print(breakdown)
            last_failures = (
                f"Iter {it}: agent wrote a CPU-fallback wrapper again for "
                f"{', '.join(still_wrappers)} instead of native ttnn ops."
            )
            last_failure_details = (
                f"The following components still match a FORBIDDEN wrapper pattern "
                f"({', '.join(_TORCH_WRAPPER_PATTERNS)}):\n\n" + (breakdown if breakdown else "  (none)")
            )
            last_failed_components = list(still_wrappers)

            for comp in still_wrappers:
                prev_class_for_log = last_failure_class_per_component.get(comp, "")

                _was_progress_log = _record_failure_for_component(comp, "WRAPPER", None)
                if comp == iter_target_component and _was_progress_log and prev_class_for_log != "WRAPPER":
                    print(
                        f"  PROGRESS: `{comp}` shifted (class WRAPPER); "
                        f"consecutive-same-class counter reset to "
                        f"1/{max_attempts_per_component}."
                    )
            continue

        banner(f"AUTO-ITERATE {it}/{max_iters}: re-run PCC tests on {BOX}")
        focused_rc = 0
        if last_failed_tests:
            iter_test_files = list(last_failed_tests)
            print("  Focused rerun on failing tests only:")
        else:
            iter_test_files = _list_component_pcc_tests(demo_dir)
            print(
                f"  Full rerun of all {len(iter_test_files)} NEW-component PCC "
                f"test(s) (scoped to bringup_status.json; stale leftover tests skipped):"
            )
        for t in iter_test_files:
            print(f"    - {t}")
        if not iter_test_files:
            print("  no tests to run for this iteration", file=sys.stderr)
            focused_rc = 0
        else:
            focused_rc = _run_focused_pytest(
                model_id=MODEL,
                test_files=iter_test_files,
                allow_kill_stale=allow_kill_stale,
                allow_device_reset=allow_device_reset,
            )
            if focused_rc != 0:
                print(f"  pytest exited non-zero ({focused_rc})", file=sys.stderr)

        if focused_rc == 124 or focused_rc < 0:
            if not last_failed_tests:
                last_failed_tests = list(iter_test_files)
            stage_map = dict(_LAST_PYTEST_STAGES)

            def _comp_from_testpath(test_path: str) -> Optional[str]:
                m = re.search(r"test_([A-Za-z0-9_]+)\.py", test_path)
                return m.group(1) if m else None

            hung_tests = list(last_failed_tests)
            inferred_components: List[str] = []
            for tp in hung_tests:
                cmp = _comp_from_testpath(tp)
                if cmp:
                    inferred_components.append(cmp)
            if stage_map:
                stage_inferred: List[str] = []
                for tname in stage_map:
                    cmp = _comp_from_testpath(tname)
                    if cmp:
                        stage_inferred.append(cmp)
                if stage_inferred:
                    inferred_components = stage_inferred
            hung_components = list(last_failed_components or []) or inferred_components

            cause = (
                "wall-clock timeout (tt_hw_planner killed the pytest process group)"
                if focused_rc == 124
                else f"external signal {-focused_rc} (likely SIGKILL/SIGTERM)"
            )
            print(
                f"  HANG detected ({cause}): synthesizing failure record so the "
                f"next iter sees a HANG signal instead of stale XML.",
                file=sys.stderr,
            )

            stage_lines: List[str] = []
            stage_per_component: Dict[str, str] = {}
            for tname, stg in stage_map.items():
                cmp = _comp_from_testpath(tname)
                stage_lines.append(f"  - {tname}: last printed stage was `{stg}`")
                if cmp:
                    stage_per_component[cmp] = stg

            stage_block = (
                "Last test stage(s) printed before the hang:\n" + "\n".join(stage_lines)
                if stage_lines
                else "No `[bringup] stage=...` lines were emitted, so the hang is "
                "probably before `build_torch_reference` (likely in module import / "
                "scaffold load)."
            )

            def _stage_specific_hint(stage: str) -> str:
                s = (stage or "").lower()
                if "ttnn_forward" in s or "ttnn_to_torch" in s:
                    return (
                        "Kernel ran but did not finish — `ttnn.synchronize_device` "
                        "is waiting on a queued op that will never complete. Almost "
                        "always one of: (a) `ttnn.matmul`/`ttnn.linear`/`ttnn.softmax` "
                        "on a TILE_LAYOUT tensor where the inner dim is NOT a "
                        "multiple of 32; (b) `ttnn.reshape` / `ttnn.permute` on a "
                        "TILE_LAYOUT tensor crossing the tile boundary; (c) mixing "
                        "REPLICATED and SHARDED memory layouts in the same op on a "
                        "mesh device. Verify EVERY matmul/linear inner dim mod 32 == 0; "
                        "convert to ROW_MAJOR_LAYOUT before any reshape/permute that "
                        "would cross a tile boundary; pick ONE mesh sharding strategy "
                        "and stick to it for the whole forward."
                    )
                if "ttnn_build" in s:
                    return (
                        "Stub `build()` itself hung. The `ttnn.from_torch(... layout=ttnn.TILE_LAYOUT)` "
                        "on a weight whose dims are NOT multiples of 32 will deadlock. "
                        "Pad weights to tile alignment before `from_torch`, or use "
                        "`layout=ttnn.ROW_MAJOR_LAYOUT` for weights you do not actually "
                        "matmul against in TILE_LAYOUT."
                    )
                if "ttnn_input" in s:
                    return (
                        "Hang was during input tensor creation. On a mesh device, "
                        "ttnn.from_torch needs a `mesh_mapper=ReplicateTensorToMesh(device)` "
                        "or a `ShardTensorToMesh(device, dim=...)`. Without one the "
                        "Python side hangs waiting for shards that were never queued."
                    )
                if "torch_forward" in s or "build_torch_reference" in s:
                    return (
                        "Hang was BEFORE any ttnn code ran — your stub's "
                        "`_get_torch_submodule()` or `build()` calling into torch is "
                        "looping. This usually means an import-time side effect or a "
                        "Python-level `while`/recursion in the stub itself, not a "
                        "device problem."
                    )
                return (
                    "Replace the forward path with straight-line ttnn (no `while True`, "
                    "no per-element Python loops over a tensor, no recursion in `__call__`); "
                    "verify every matmul inner dim is divisible by 32."
                )

            last_failures = (
                "WALL-CLOCK BUDGET EXHAUSTED: pytest was killed because the "
                "previous stub did not return from its forward path within the "
                "wall-clock timeout (likely an unbounded loop / infinite "
                "recursion / device-side deadlock inside the new code)."
            )
            last_failure_details = (
                "Hung tests:\n  - "
                + "\n  - ".join(hung_tests)
                + "\n\nComponents involved: "
                + ", ".join(hung_components or ["(unknown)"])
                + "\n\n"
                + stage_block
                + "\n\nNo pytest XML was produced for this iteration because the "
                "process group was killed."
            )

            for comp in hung_components:
                stg = stage_per_component.get(comp, "(no stage printed)")
                choice = iter_choices.get(comp, {}) if isinstance(iter_choices.get(comp), dict) else {}
                _write_attempt_log(
                    demo_dir=demo_dir,
                    component_name=comp,
                    iter_n=it,
                    stub_path=_component_stub_path(demo_dir, comp),
                    exemplar_used=choice.get("exemplar_used"),
                    model_used=choice.get("model_used", model),
                    failure_class="HANG",
                    failure_signature=f"HANG@stage={stg}",
                    traceback_excerpt=f"(no traceback; process group killed at stage={stg})",
                    diagnosis_override=(
                        f"pytest killed by wall-clock timeout; last reported test "
                        f"stage for this component was `{stg}`"
                    ),
                    next_step_override=_stage_specific_hint(stg),
                )
            last_failed_components = hung_components or last_failed_components

            for comp in hung_components or []:
                prev_class_for_log = last_failure_class_per_component.get(comp, "")

                _was_progress_log = _record_failure_for_component(comp, "HANG", None)
                if comp == iter_target_component and _was_progress_log and prev_class_for_log != "HANG":
                    print(
                        f"  PROGRESS: `{comp}` shifted (class HANG); "
                        f"consecutive-same-class counter reset to "
                        f"1/{max_attempts_per_component}."
                    )

            systemic_key_crossed_hang: Optional[tuple] = None
            for comp in hung_components or []:
                stg = stage_per_component.get(comp, "(no-stage)")
                hang_sig = f"HANG@stage={stg}"
                crossed = _record_systemic_failure(comp, "HANG", hang_sig)
                if crossed is not None and systemic_key_crossed_hang is None:
                    systemic_key_crossed_hang = crossed
            if systemic_key_crossed_hang is not None:
                sk_class_h, sk_sig_h = systemic_key_crossed_hang
                sk_comps_h = sorted(systemic_failure_counts.get(systemic_key_crossed_hang, set()))
                print()
                print(sep)
                print(
                    f"  SYSTEMIC PATTERN DETECTED: {len(sk_comps_h)} distinct "
                    f"components share the SAME hang signature."
                )
                print(f"  affected: {', '.join(sk_comps_h)}")
                print(f"  signature: {sk_sig_h}")
                print(f"  Next prompt will route the LLM to investigate " f"the shared harness path.")
                print(sep)

            print()
            print(sep)
            print(f"  Iter {it} compute split (HANG verdict; structural only):")
            for line in _format_compute_split(MODEL, label="components ", indent="  "):
                print(line)
            for line in _format_op_split(MODEL, label="operations  ", indent="  ", show_per_component=False):
                print(line)
            print(sep)
            continue

        report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
        ungraduated, smoke_tests = _auto_iteration_blockers(MODEL)
        all_passed = bool(report.get("all_passed", False))

        report_failed = set(report.get("failed_components", []) or [])
        report_skipped = set(report.get("skipped_components", []) or [])
        report_passed = set(report.get("passed_components", []) or [])

        validated_this_run |= report_passed
        validated_this_run -= report_failed
        validated_this_run -= report_skipped
        skipped_components_this_run |= report_skipped
        skipped_components_this_run -= report_passed
        skipped_components_this_run -= report_failed
        target_was_validated = (
            iter_target_component is not None
            and iter_target_component in report_passed
            and iter_target_component not in report_skipped
            and iter_target_component not in report_failed
        )
        # First-time graduation: target_was_validated (pytest passed) AND
        # the stub is AST-native (does NOT delegate to torch fallback).
        # We DO NOT consult `ungraduated` here — that list is derived from
        # `_stub_has_graduated_from_autofill`, which itself requires the
        # `.py.last_good_native` snapshot to already exist. That snapshot
        # is only written by `_snapshot_native_stub` AFTER this very
        # check passes. So gating on `ungraduated` creates a chicken-
        # and-egg: snapshot never written for a first-time graduation,
        # component classified "ungraduated" forever even after pytest
        # passes on native code. Direct AST check breaks the cycle.
        target_stub_path = (
            (demo_dir / "_stubs" / f"{_safe_id(iter_target_component)}.py") if iter_target_component else None
        )
        target_is_ast_native = bool(
            target_stub_path and target_stub_path.is_file() and not _stub_uses_torch_wrapper(target_stub_path)
        )
        target_newly_graduated = (
            target_was_validated
            and all_passed
            and target_is_ast_native
            and iter_target_component not in smoke_tests
            and iter_target_component not in graduated_this_run
            and iter_target_component not in permanently_skipped
        )

        target_skipped_due_to_harness = False
        harness_skip_reasons: List[str] = []
        if (
            iter_target_component is not None
            and iter_target_component in report_skipped
            and iter_target_component not in report_failed
            and not target_newly_graduated
        ):
            per_skipped_map = report.get("per_skipped", {}) if isinstance(report, dict) else {}
            if isinstance(per_skipped_map, dict):
                for entry in per_skipped_map.values():
                    if isinstance(entry, dict) and entry.get("component") == iter_target_component:
                        reason = str(entry.get("reason") or "").strip()
                        if reason:
                            harness_skip_reasons.append(reason)
            harness_markers = (
                "HF reference forward",
                "_make_arg_for()",
                "synthetic inputs from _make_arg_for",
                "incompatible with this submodule's expected shapes",
                "the synthetic inputs",
            )
            harness_pattern_hit = any(any(mark in r for mark in harness_markers) for r in harness_skip_reasons)
            target_safe_for_check = _safe_id(iter_target_component)
            stub_path_for_check = demo_dir / "_stubs" / f"{target_safe_for_check}.py"
            stub_is_native = False
            try:
                stub_is_native = _stub_has_graduated_from_autofill(stub_path_for_check)
            except Exception:
                stub_is_native = False
            target_skipped_due_to_harness = bool(
                harness_pattern_hit
                and stub_is_native
                and iter_target_component not in graduated_this_run
                and iter_target_component not in unverified_native_this_run
                and iter_target_component not in permanently_skipped
            )

        _harness_markers = (
            "HF reference forward",
            "_make_arg_for()",
            "synthetic inputs from _make_arg_for",
            "incompatible with this submodule's expected shapes",
            "the synthetic inputs",
        )
        _other_skipped = report_skipped - {iter_target_component}
        if _other_skipped:
            _broad_per_skipped = report.get("per_skipped", {}) if isinstance(report, dict) else {}
            for _comp in sorted(_other_skipped):
                if _comp in graduated_this_run or _comp in unverified_native_this_run or _comp in permanently_skipped:
                    continue
                _comp_reasons: List[str] = []
                if isinstance(_broad_per_skipped, dict):
                    for entry in _broad_per_skipped.values():
                        if isinstance(entry, dict) and entry.get("component") == _comp:
                            r = str(entry.get("reason") or "").strip()
                            if r:
                                _comp_reasons.append(r)
                if not any(any(m in r for m in _harness_markers) for r in _comp_reasons):
                    continue
                _comp_stub = demo_dir / "_stubs" / f"{_safe_id(_comp)}.py"
                try:
                    if not _stub_has_graduated_from_autofill(_comp_stub):
                        continue
                except Exception:
                    continue
                unverified_native_this_run.add(_comp)
                skipped_components_this_run.discard(_comp)
                verified_fail.discard(_comp)
                _r_blob = "; ".join(_comp_reasons) or "(no skip reason captured)"
                # Side-channel UNVERIFIED NATIVE = same root cause as
                # seed-phase: harness can't measure PCC for this stub.
                # Classify as TOOL_BUG for accurate skip-list labeling.
                persist_skip(MODEL, _comp, _r_blob, category="TOOL_BUG")
                print(
                    f"  iter {it} side-channel: `{_comp}` UNVERIFIED NATIVE "
                    f"(stub native + harness-SKIP). Removed from candidate "
                    f"pool to avoid wasted LLM iteration. Reason: {_r_blob}"
                )

        if target_newly_graduated:
            graduated_this_run.append(iter_target_component)
            verified_fail.discard(iter_target_component)
            _snapshot_native_stub(iter_target_component)
            banner(
                f"AUTO-ITERATE {it}/{max_iters}: `{iter_target_component}` "
                f"GRADUATED to native TTNN on {BOX} (snapshot saved, PCC test PASSED)"
            )
            # G7 (agentic.learnings): persist the graduated stub diff so a
            # future run on the same HF architecture can apply it
            # directly. arch_signature is computed from the HF config;
            # first_diverging_qn = component name; diff = git diff of
            # the stub file. Direct reuse of the existing engine — no
            # logic duplicated here.
            try:
                from ..agentic.executor import _load_hf_config as _load_hf
                from ..agentic.learnings import compute_arch_signature, register_fix

                _arch_sig = compute_arch_signature(_load_hf(MODEL))
                if _arch_sig:
                    _safe_grad = _safe_id(iter_target_component)
                    _stub_rel = (demo_dir / "_stubs" / f"{_safe_grad}.py").resolve().relative_to(BRINGUP_ROOT())
                    _diff_proc = subprocess.run(
                        ["git", "diff", "--no-color", "HEAD", "--", str(_stub_rel)],
                        cwd=str(BRINGUP_ROOT()),
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    _diff_text = _diff_proc.stdout or ""
                    if _diff_text.strip():
                        _ok = register_fix(
                            arch_signature=_arch_sig,
                            first_diverging_qn=iter_target_component,
                            diff=_diff_text,
                            diff_files=[str(_stub_rel)],
                            source_model_id=MODEL,
                            notes=f"graduated via auto_iterate iter {it}",
                        )
                        if _ok:
                            print(
                                f"  [agentic:G7] registered learned fix for "
                                f"`{iter_target_component}` "
                                f"(arch_sig={_arch_sig[:12]}, diff={len(_diff_text)} bytes)"
                            )
            except Exception as _g7_exc:
                print(
                    f"  [agentic:G7] non-fatal: " f"{type(_g7_exc).__name__}: {_g7_exc}",
                    file=sys.stderr,
                )
        elif target_skipped_due_to_harness:
            unverified_native_this_run.add(iter_target_component)
            verified_fail.discard(iter_target_component)
            skipped_components_this_run.discard(iter_target_component)
            persist_skip(
                MODEL,
                iter_target_component,
                "harness-incompatible (target SKIP)",
                category="TOOL_BUG",
            )
            last_failed_components = []
            last_failed_tests = []
            last_failures = ""
            last_failure_details = ""
            skip_reason_blob = "\n".join(f"    - {r}" for r in harness_skip_reasons) or "    (no skip reason captured)"
            banner(
                f"AUTO-ITERATE {it}/{max_iters}: `{iter_target_component}` "
                f"UNVERIFIED NATIVE on {BOX} "
                f"(native ttnn forward present; PCC NOT measured — test "
                f"harness cannot synthesize valid inputs)"
            )
            print(
                f"  Stub passes the native-forward heuristic but pytest "
                f"SKIPPED because the auto-generated _make_arg_for() "
                f"inputs don't match the HF reference signature. Reasons:\n"
                f"{skip_reason_blob}\n"
                f"  This component is NOT counted as graduated (no PCC "
                f"proof) and the auto-loop will not retry it. To verify "
                f"correctness, hand-edit `tests/pcc/test_"
                f"{_safe_id(iter_target_component)}.py` so the inputs "
                f"match this submodule's expected shapes/dtypes, then "
                f"re-run; the loop will then graduate (or fail) it for "
                f"real. The bring-up-complete banner is BLOCKED until "
                f"that happens."
            )
        elif (
            iter_target_component is not None
            and iter_target_component in report_skipped
            and iter_target_component not in report_failed
        ):
            verified_fail.add(iter_target_component)
            if iter_target_component in graduated_this_run:
                graduated_this_run.remove(iter_target_component)
            skip_reasons: List[str] = list(harness_skip_reasons)
            if not skip_reasons:
                per_skipped_map = report.get("per_skipped", {}) if isinstance(report, dict) else {}
                if isinstance(per_skipped_map, dict):
                    for entry in per_skipped_map.values():
                        if isinstance(entry, dict) and entry.get("component") == iter_target_component:
                            reason = str(entry.get("reason") or "").strip()
                            if reason:
                                skip_reasons.append(reason)
            skip_reason_blob = "\n".join(f"    - {r}" for r in skip_reasons) or "    (no skip reason captured)"
            target_safe = _safe_id(iter_target_component)
            target_test_rel = safe_relative_to_root(demo_dir / "tests" / "pcc" / f"test_{target_safe}.py")
            print(
                f"\n  AUTO-ITERATE {it}/{max_iters}: `{iter_target_component}` "
                f"PCC test SKIPPED (test harness could not build inputs). "
                f"NOT graduating — needs another attempt with shape-aware "
                f"inputs from the agent.",
                file=sys.stderr,
            )
            last_failures = (
                f"Iter {it}: `{iter_target_component}` PCC test SKIPPED — "
                f"the test harness's synthetic inputs in `_make_arg_for()` "
                f"do not match what the HF reference forward expects. The "
                f"stub itself may compile fine, but there is NO numerical "
                f"validation. Fix the test scaffold so the reference "
                f"forward succeeds, then PCC can be measured."
            )
            last_failure_details = (
                f"SKIP reasons reported by pytest for `{iter_target_component}`:\n"
                f"{skip_reason_blob}\n\n"
                f"Action required next iteration:\n"
                f"  1. Inspect the HF reference module's forward signature "
                f"(arg names, dtypes, expected ranks).\n"
                f"  2. Update `{target_test_rel}` so the inputs passed to "
                f"both the torch reference and the ttnn port have the "
                f"correct shapes/dtypes. The scaffolder's `_make_arg_for()` "
                f"only knows generic shapes — write component-specific "
                f"input construction in the test (or in a helper imported "
                f"by the test).\n"
                f"  3. Re-run the test and confirm a numeric PCC value is "
                f"reported (not a SKIP).\n"
                f"  4. If PCC fails, iterate on `_stubs/{target_safe}.py`."
            )
            last_failed_components = [iter_target_component]
            last_failed_tests = [str(target_test_rel)]
        elif iter_target_component is not None and (not all_passed or iter_target_component in report_failed):
            verified_fail.add(iter_target_component)
            if iter_target_component in graduated_this_run:
                graduated_this_run.remove(iter_target_component)

        if target_newly_graduated and graduated_this_run:
            previously_graduated = [g for g in graduated_this_run if g != iter_target_component]
            regression_test_files: List[str] = []
            for g in previously_graduated:
                tp = demo_dir / "tests" / "pcc" / f"test_{_safe_id(g)}.py"
                if tp.is_file():
                    regression_test_files.append(str(safe_relative_to_root(tp)))
            if regression_test_files:
                banner(
                    f"AUTO-ITERATE {it}/{max_iters}: regression sweep on "
                    f"{len(regression_test_files)} previously-graduated component(s)"
                )
                print(
                    f"  This catches the case where the agent's change for "
                    f"`{iter_target_component}` silently broke an earlier "
                    f"native-graduation. On detection we roll back the regressed "
                    f"stub(s) to their `.last_good_native` snapshot and re-queue "
                    f"them for another attempt."
                )
                regression_rc = _run_focused_pytest(
                    model_id=MODEL,
                    test_files=regression_test_files,
                    allow_kill_stale=allow_kill_stale,
                    allow_device_reset=allow_device_reset,
                )
                regression_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
                regressed_names = [
                    n for n in regression_report.get("failed_components", []) if n in previously_graduated
                ]
                if regression_rc == 124 or regression_rc < 0:
                    print(
                        "  regression sweep hung (wall-clock killed); "
                        "recording HANG against previously-graduated "
                        "components and continuing — next iter will "
                        "re-evaluate them.",
                        file=sys.stderr,
                    )
                    for hung_g in previously_graduated:
                        _record_failure_for_component(hung_g, "HANG", None)
                elif regressed_names:
                    print(
                        f"  REGRESSION DETECTED: {len(regressed_names)} previously "
                        f"native component(s) now fail: {', '.join(regressed_names)}"
                    )
                    restored_names: List[str] = []
                    for r in regressed_names:
                        if _restore_native_snapshot(r):
                            restored_names.append(r)
                            print(f"    restored `{r}` from ..last_good_native snapshot")
                        else:
                            print(
                                f"    NO snapshot available for `{r}`; leaving "
                                f"the regressed stub in place — next iter will "
                                f"re-attempt it",
                                file=sys.stderr,
                            )
                    if restored_names:
                        verify_tests = [
                            str(safe_relative_to_root(demo_dir / "tests" / "pcc" / f"test_{_safe_id(r)}.py"))
                            for r in restored_names
                            if (demo_dir / "tests" / "pcc" / f"test_{_safe_id(r)}.py").is_file()
                        ]
                        verify_rc = (
                            _run_focused_pytest(
                                model_id=MODEL,
                                test_files=verify_tests,
                                allow_kill_stale=allow_kill_stale,
                                allow_device_reset=allow_device_reset,
                            )
                            if verify_tests
                            else 0
                        )
                        verify_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
                        still_broken = [n for n in verify_report.get("failed_components", []) if n in restored_names]
                        truly_healed = [r for r in restored_names if r not in still_broken]
                        if truly_healed:
                            print(
                                f"  regression healed by snapshot restore for: "
                                f"{', '.join(truly_healed)} (re-verified by pytest)"
                            )
                        for r in still_broken:
                            print(
                                f"  `{r}` still fails after snapshot restore — "
                                f"shared infra likely changed under it. Demoting "
                                f"to CPU fallback so the demo keeps running."
                            )
                            _skip_component_to_fallback(
                                r,
                                "regression survived snapshot restore (shared " "infra or test harness changed)",
                            )
                            if r in graduated_this_run:
                                graduated_this_run.remove(r)
                    for r in regressed_names:
                        if r in graduated_this_run and r not in restored_names:
                            graduated_this_run.remove(r)

                        attempts_per_component.pop(r, None)
                        consecutive_same_class_attempts.pop(r, None)
                        last_failure_class_per_component.pop(r, None)
                        last_pcc_per_component.pop(r, None)
                else:
                    print(
                        f"  regression sweep PASSED: all {len(regression_test_files)} "
                        f"previously-graduated component(s) still pass"
                    )
                    for g in previously_graduated:
                        _snapshot_native_stub(g)

        print()
        print(sep)
        print(f"  Iter {it} compute split (after pytest on {BOX}):")
        for line in _format_compute_split(MODEL, label="components ", indent="  "):
            print(line)
        for line in _format_op_split(MODEL, label="operations  ", indent="  ", show_per_component=False):
            print(line)
        print(sep)

        remaining_ungrad_for_conv = sorted(set(ungraduated) - set(permanently_skipped))
        remaining_smoke_for_conv = sorted(set(smoke_tests) - set(permanently_skipped))
        remaining_verified_fail = sorted(set(verified_fail) - set(permanently_skipped))

        all_new_components = {c for c in (ungraduated or []) if c}
        for comp_meta in (
            json.loads((demo_dir / "bringup_status.json").read_text()).get("components", [])
            if (demo_dir / "bringup_status.json").is_file()
            else []
        ):
            if comp_meta.get("status") == "NEW":
                nm = str(comp_meta.get("name", "")).strip()
                if nm:
                    all_new_components.add(nm)
        unvalidated_new = sorted(all_new_components - validated_this_run - set(permanently_skipped))
        still_skipped_new = sorted(
            (skipped_components_this_run & all_new_components) - validated_this_run - set(permanently_skipped)
        )

        _live_partial_cpu = set(_partial_cpu_components(MODEL)) - set(permanently_skipped)
        _partial_cpu_block_convergence = not allow_partial_cpu and bool(_live_partial_cpu)
        if strict_native:
            converged_native = (
                all_passed
                and not ungraduated
                and not smoke_tests
                and not verified_fail
                and not unvalidated_new
                and not _partial_cpu_block_convergence
            )
            converged_partial = (
                all_passed
                and bool(permanently_skipped)
                and not remaining_ungrad_for_conv
                and not remaining_smoke_for_conv
                and not remaining_verified_fail
                and not unvalidated_new
                and not _partial_cpu_block_convergence
            )
            converged = converged_native or converged_partial
            if converged_partial and not converged_native:
                converge_msg = (
                    f"best-effort bring-up: "
                    f"{len(graduated_this_run)} native, "
                    f"{len(permanently_skipped)} on CPU fallback "
                    f"(after {max_attempts_per_component}-attempt cap)"
                )
            else:
                if _live_partial_cpu:
                    converge_msg = (
                        f"all PCC tests pass; "
                        f"{len(_live_partial_cpu)} component(s) have "
                        f"runtime CPU fallback(s) (allowed via "
                        f"--allow-partial-cpu): "
                        f"{', '.join(sorted(_live_partial_cpu))}"
                    )
                else:
                    converge_msg = f"model runs natively on {BOX} (no CPU fallback)"

            sweep_candidates = [c for c in unvalidated_new if c not in permanently_skipped and c not in verified_fail]
            from .sweep_cache import should_skip_validation_sweep

            if sweep_candidates and should_skip_validation_sweep(focused_rc):
                print(
                    f"  validation sweep: SKIPPED (focused pytest "
                    f"crashed/timed-out rc={focused_rc}; sweep on other "
                    f"components yields no new info this iter)"
                )
                sweep_candidates = []
            if sweep_candidates:
                must_run, cached_pass = _SWEEP_CACHE.split_sweep_candidates(
                    sweep_candidates,
                    {c: demo_dir / "_stubs" / f"{_safe_id(c)}.py" for c in sweep_candidates},
                )
                if cached_pass:
                    print(
                        f"  validation sweep cache: skipping {len(cached_pass)} "
                        f"component(s) whose stub is unchanged since last PASS "
                        f"({', '.join(sorted(cached_pass))})"
                    )
                    validated_this_run |= set(cached_pass)
                sweep_targets = must_run
                sweep_tests = _list_component_pcc_tests(demo_dir, only=sweep_targets) if sweep_targets else []
                if sweep_tests:
                    banner(
                        f"AUTO-ITERATE {it}/{max_iters}: validation sweep on "
                        f"{len(sweep_tests)} NEW component(s) not yet "
                        f"numerically validated this run "
                        f"({', '.join(sweep_targets)})"
                    )
                    sweep_rc = _run_focused_pytest(
                        model_id=MODEL,
                        test_files=sweep_tests,
                        allow_kill_stale=allow_kill_stale,
                        allow_device_reset=allow_device_reset,
                    )
                    sweep_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
                    sw_passed = set(sweep_report.get("passed_components", []) or [])
                    sw_failed = set(sweep_report.get("failed_components", []) or [])
                    sw_skipped = set(sweep_report.get("skipped_components", []) or [])
                    for _c in sw_passed:
                        _SWEEP_CACHE.record(_c, demo_dir / "_stubs" / f"{_safe_id(_c)}.py", "PASS")
                    for _c in sw_failed:
                        _SWEEP_CACHE.record(_c, demo_dir / "_stubs" / f"{_safe_id(_c)}.py", "FAIL")
                    validated_this_run |= sw_passed
                    validated_this_run -= sw_failed
                    validated_this_run -= sw_skipped
                    skipped_components_this_run = (skipped_components_this_run | sw_skipped) - sw_passed - sw_failed
                    verified_fail |= sw_failed
                    print(
                        f"  validation sweep: "
                        f"{len(sw_passed)} passed, "
                        f"{len(sw_failed)} failed, "
                        f"{len(sw_skipped)} skipped"
                    )
                    if sw_passed:
                        print(f"    passed   : {', '.join(sorted(sw_passed))}")
                    if sw_failed:
                        print(f"    failed   : {', '.join(sorted(sw_failed))}")
                    if sw_skipped:
                        print(f"    skipped  : {', '.join(sorted(sw_skipped))}")
                    unvalidated_new = sorted(all_new_components - validated_this_run - set(permanently_skipped))
                    if sw_failed:
                        last_failed_components = list(sw_failed)
                        last_failed_tests = list(sweep_report.get("failed_tests", []))
                        last_failures = str(sweep_report.get("summary", ""))
                        last_failure_details = str(sweep_report.get("details", ""))

                    _live_partial_cpu = set(_partial_cpu_components(MODEL)) - set(permanently_skipped)
                    _partial_cpu_block_convergence = not allow_partial_cpu and bool(_live_partial_cpu)
                    converged_native = (
                        all_passed
                        and not ungraduated
                        and not smoke_tests
                        and not verified_fail
                        and not unvalidated_new
                        and not _partial_cpu_block_convergence
                    )
                    converged_partial = (
                        all_passed
                        and bool(permanently_skipped)
                        and not remaining_ungrad_for_conv
                        and not remaining_smoke_for_conv
                        and not remaining_verified_fail
                        and not unvalidated_new
                        and not _partial_cpu_block_convergence
                    )
                    converged = converged_native or converged_partial
                    if sweep_rc != 0 and converged:
                        converged = False
        else:
            converged = all_passed and not smoke_tests
            converge_msg = (
                f"model runs end-to-end on {BOX} (PCC tests pass; some components "
                f"may still be CPU fallback — see summary)"
                if ungraduated
                else f"model runs natively on {BOX}"
            )

        if (
            strict_native
            and all_passed
            and not ungraduated
            and not smoke_tests
            and not verified_fail
            and not unvalidated_new
            and _partial_cpu_block_convergence
            and not converged
        ):
            print()
            print(sep)
            print(
                f"  AUTO-ITERATE {it}/{max_iters}: PCC tests pass but "
                f"{len(_live_partial_cpu)} component(s) have runtime "
                f"CPU fallback(s) — continuing to push toward 100% "
                f"on-device"
            )
            print(f"  partial-CPU: {', '.join(sorted(_live_partial_cpu))}")
            print(f"  (use --allow-partial-cpu to converge here at " f"PCC-pass; default is strict push-to-100%)")
            print(sep)
        if converged:
            banner(f"AUTO-ITERATE converged after {it} iteration(s) — {converge_msg}")
            _print_bringup_summary(MODEL, box=BOX, sep=sep)
            if permanently_skipped:
                print()
                if graduated_this_run:
                    print(f"  native TTNN : {', '.join(sorted(set(graduated_this_run)))}")
                print(f"  CPU fallback: {', '.join(sorted(set(permanently_skipped)))}")
                print(
                    "\n  To retry the CPU-fallback components later:\n"
                    f"    python -m scripts.tt_hw_planner promote {MODEL} \\\n"
                    f"        --box {BOX} --auto --auto-agent {provider} \\\n"
                    f"        --auto-max-iters 12 --auto-agent-timeout 1500\n"
                )

            # Emit RUN_REPORT.md — markdown summary of the run.
            # Wrapped: write failure must never propagate (the run
            # itself succeeded; report is a diagnostic artifact).
            try:
                from ..run_report import emit_run_report

                report_path = emit_run_report(
                    MODEL,
                    demo_dir,
                    converged=True,
                    iterations_run=it,
                )
                if report_path is not None:
                    print(f"  run report: {report_path}")
            except Exception:
                pass

            # NOTE: end-to-end demo emission has moved OUT of the
            # auto-iterate convergence path. It now happens in cmd_up
            # AFTER Phase 2 stage + final categorization gate, so the
            # demo only emits when every HOT component is graduated.
            # The convergence banner here just announces "Phase 1 done".
            return 0
        if (
            strict_native
            and all_passed
            and (remaining_ungrad_for_conv or remaining_smoke_for_conv or remaining_verified_fail)
        ):
            ungraduated = remaining_ungrad_for_conv
            smoke_tests = remaining_smoke_for_conv
            blocked = sorted(set(ungraduated) | set(smoke_tests) | set(remaining_verified_fail))
            breakdown = _ungraduated_breakdown(demo_dir, ungraduated)
            print()
            print(sep)
            print(f"  Iter {it}: target PASSED but {len(blocked)} non-skipped component(s) still on fallback")
            print(sep)
            print("  The PCC >= 0.99 contract is `pcc(ttnn_native(x), torch_ref(x)) >= 0.99`.")
            print("  Components still wrapping torch on CPU pass PCC trivially " "(torch == torch).")
            if breakdown:
                print(breakdown)
            if smoke_tests:
                print(f"  Phase-1 SMOKE tests still present: {', '.join(smoke_tests)}")
            if permanently_skipped:
                print(
                    f"  (Additionally on CPU fallback after attempt cap, not retried: "
                    f"{', '.join(sorted(set(permanently_skipped)))})"
                )
            last_failures = (
                "Bring-up not complete: tests passed only because the listed "
                "components delegate to torch on CPU. Continue iterating on "
                "the non-skipped components below."
            )
            last_failure_details = (
                "Components still CPU-wrapping (PCC trivially passes via torch==torch):\n"
                + (breakdown if breakdown else "  (none)")
                + (f"\nPhase-1 SMOKE tests still present: {', '.join(smoke_tests)}" if smoke_tests else "")
            )
            last_failed_components = blocked
            last_failed_tests = []
        elif (not strict_native) and all_passed and smoke_tests:
            print()
            print(f"  Iter {it}: tests PASSED but Phase-1 SMOKE tests are still present")
            print(f"  Phase-1 SMOKE tests: {', '.join(smoke_tests)}")
            last_failures = "Tests passed, but Phase-1 SMOKE tests are still present " "(no real PCC validation yet)."
            last_failure_details = (
                f"Phase-1 SMOKE tests still present: {', '.join(smoke_tests)}\n"
                "Synthesize each missing component into _stubs/<comp>.py and "
                "regenerate real PCC tests."
            )
            last_failed_components = list(smoke_tests)
            last_failed_tests = []
        else:
            last_failures = str(report.get("summary", "(no failure summary)"))
            last_failure_details = str(report.get("details", "(no failure traceback parsed)"))
            last_failed_components = list(report.get("failed_components", []))
            last_failed_tests = list(report.get("failed_tests", []))
            failure_class = _classify_failure(last_failures, last_failure_details)
            signature = _failure_signature(last_failures, last_failure_details)
            for comp in set(last_failed_components):
                k = f"{comp}|{failure_class}|{signature}"
                repeat_error_counts[k] = repeat_error_counts.get(k, 0) + 1
        print(f"\n  Iteration {it} failures (will be fed back to agent next round):")
        print(last_failures)

        iter_failure_class = _classify_failure(last_failures, last_failure_details)
        iter_signature = _failure_signature(last_failures, last_failure_details)
        iter_pcc = _extract_pcc_from_failure(last_failures, last_failure_details)

        last_shape_probes = _extract_shape_probes_from_report(report)
        if last_shape_probes:
            print(
                f"  SHAPE_PROBE: captured {len(last_shape_probes)} probe "
                f"line(s) from this iter; will fold into next prompt."
            )

        systemic_key_crossed: Optional[tuple] = None
        for failed_comp_for_sys in set(last_failed_components or []):
            crossed = _record_systemic_failure(failed_comp_for_sys, iter_failure_class, iter_signature)
            if crossed is not None and systemic_key_crossed is None:
                systemic_key_crossed = crossed
        if systemic_key_crossed is not None:
            sk_class, sk_sig = systemic_key_crossed
            sk_comps = sorted(systemic_failure_counts.get(systemic_key_crossed, set()))
            print()
            print(sep)
            print(
                f"  SYSTEMIC PATTERN DETECTED: {len(sk_comps)} distinct "
                f"components share the SAME failure class `{sk_class}` and"
            )
            print(
                f"  signature. The bug is almost certainly in a SHARED "
                f"path (harness, conftest, op_helper) — not in any one"
            )
            print(f"  stub. Next prompt will include investigation directives.")
            print(f"  affected: {', '.join(sk_comps)}")
            print(f"  signature: {sk_sig[:160]}{'...' if len(sk_sig) > 160 else ''}")
            print(sep)

        for failed_comp in set(last_failed_components or []):
            prev_class_for_log = last_failure_class_per_component.get(failed_comp, "")
            prev_pcc_for_log = last_pcc_per_component.get(failed_comp)

            _snapshot_best_native_stub(failed_comp, iter_pcc)
            was_progress = _record_failure_for_component(failed_comp, iter_failure_class, iter_pcc)
            if was_progress and failed_comp == iter_target_component:
                change_parts: List[str] = []
                if prev_class_for_log and prev_class_for_log != iter_failure_class:
                    change_parts.append(f"class {prev_class_for_log} -> {iter_failure_class}")
                elif iter_failure_class:
                    change_parts.append(f"class {iter_failure_class}")
                if iter_pcc is not None and prev_pcc_for_log is not None and iter_pcc > prev_pcc_for_log + 0.001:
                    change_parts.append(f"PCC {prev_pcc_for_log:.4f} -> {iter_pcc:.4f}")
                elif iter_pcc is not None:
                    change_parts.append(f"PCC {iter_pcc:.4f}")
                change_note = "; ".join(change_parts) if change_parts else "first failure observed"
                print(
                    f"  PROGRESS: `{failed_comp}` shifted ({change_note}); "
                    f"consecutive-same-class counter reset to "
                    f"1/{max_attempts_per_component}."
                )
        latest_report_for_log = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
        per_component_struct = (
            latest_report_for_log.get("per_component", {}) if isinstance(latest_report_for_log, dict) else {}
        )
        if not isinstance(per_component_struct, dict):
            per_component_struct = {}
        for failed_comp in set(last_failed_components or []):
            choice = iter_choices.get(failed_comp, {})
            safe_failed = _safe_id(failed_comp)
            stub_for_log = demo_dir / "_stubs" / f"{safe_failed}.py"
            comp_traceback = _format_failure_block_for_component(per_component_struct, failed_comp)
            if not comp_traceback or comp_traceback.startswith("(no failure"):
                comp_traceback = last_failure_details
            _write_attempt_log(
                demo_dir=demo_dir,
                component_name=failed_comp,
                iter_n=it,
                stub_path=stub_for_log,
                exemplar_used=choice.get("exemplar_used"),
                model_used=choice.get("model_used", model),
                failure_class=iter_failure_class,
                failure_signature=iter_signature,
                traceback_excerpt=comp_traceback,
            )
        if last_failed_components:
            log_root = demo_dir / "_attempts"
            print(
                f"  per-component handoff logs written to: {safe_relative_to_root(log_root) if log_root.is_absolute() else log_root}"
            )

        if (
            iter_target_component is not None
            and iter_target_component not in graduated_this_run
            and iter_target_component not in permanently_skipped
            and _is_at_cap(iter_target_component)
        ):
            _skip_component_to_fallback(
                iter_target_component,
                f"failed {_attempts_display(iter_target_component)} attempt(s); "
                f"moving on to next ungraduated component",
            )
            last_failed_components = [c for c in (last_failed_components or []) if c != iter_target_component]
            if not last_failed_components:
                last_failed_tests = []
                last_failures = ""
                last_failure_details = ""

    if last_failed_components and _only_pcc_threshold_failures(last_failures):
        banner("AUTO-ITERATE exhausted; attempting automatic stabilization fallback for PCC-only failures")
        rewritten = _rewrite_components_to_stable_fallback(demo_dir, last_failed_components)
        if rewritten:
            print(f"  Stabilized fallback applied to: {', '.join(rewritten)}")
            for r in rewritten:
                if r not in permanently_skipped:
                    permanently_skipped.append(r)
            if last_failed_tests:
                focused_rc = _run_focused_pytest(
                    model_id=MODEL,
                    test_files=last_failed_tests,
                    allow_kill_stale=allow_kill_stale,
                    allow_device_reset=allow_device_reset,
                )
                if focused_rc != 0:
                    print(f"  focused pytest exited non-zero ({focused_rc}) after stabilization", file=sys.stderr)
            else:
                prepare_argv = argparse.Namespace(
                    model_id=MODEL,
                    box=BOX,
                    mesh=mesh,
                    dtype=dtype,
                    batch=batch,
                    max_seq_len=max_seq_len,
                    max_generated_tokens=max_generated_tokens,
                    accuracy=accuracy,
                    no_trace=no_trace,
                    no_paged_attention=no_paged_attention,
                    no_instruct=no_instruct,
                    format="text",
                    write_script=None,
                    execute=True,
                    download_first=False,
                    strict=strict,
                    allow_port=False,
                )
                try:
                    cmd_prepare(prepare_argv)
                except Exception as exc:
                    print(f"  prepare/execute failed after stabilization: {exc}", file=sys.stderr)
            report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
            ungraduated, smoke_tests = _auto_iteration_blockers(MODEL)
            if bool(report.get("all_passed", False)) and not ungraduated and not smoke_tests:
                banner("AUTO-ITERATE stabilized after fallback rewrite — all PCC tests pass")
                return 0
            last_failures = str(report.get("summary", "(no failure summary)"))

    all_pcc_tests = _list_component_pcc_tests(demo_dir)

    final_ungrad, final_smoke = _auto_iteration_blockers(MODEL)
    still_pending = sorted((set(final_ungrad) | set(final_smoke)) - set(permanently_skipped))
    if still_pending:
        banner(
            f"AUTO-ITERATE exhausted {max_iters} iter(s); leaving {len(still_pending)} "
            f"still-pending component(s) on CPU fallback so the demo runs end-to-end"
        )
        print(f"  still-pending -> CPU fallback: {', '.join(still_pending)}")
        # Route each still-pending component through the SAME escalation
        # chain that per-component cap exhaustion uses. This ensures
        # failure_classifier fires, persist_skip records any
        # KERNEL_MISSING verdict, decompose-auto runs for composites,
        # and learned-fix state is captured. Without this call, the
        # loop-level max_iters exit would silently bypass every
        # escalation primitive — gap discovered in the SAM2 brain-test
        # run on 2026-05-30.
        for _pending_comp in still_pending:
            try:
                _skip_component_to_fallback(
                    _pending_comp,
                    f"loop-level max_iters ({max_iters}) exhausted with this " f"component still pending",
                )
            except Exception as _exit_esc_exc:
                print(
                    f"  [exit-escalation] non-fatal for `{_pending_comp}`: "
                    f"{type(_exit_esc_exc).__name__}: {_exit_esc_exc}",
                    file=sys.stderr,
                )
        # After escalation classification + side effects (skip-list,
        # decompose-plan, etc.) are persisted, do the actual stub
        # stabilization so the demo can still emit end-to-end.
        rewritten = _rewrite_components_to_stable_fallback(demo_dir, still_pending)
        for r in rewritten:
            if r not in permanently_skipped:
                permanently_skipped.append(r)

    final_rc = 0
    if all_pcc_tests:
        final_rc = _run_focused_pytest(
            model_id=MODEL,
            test_files=all_pcc_tests,
            allow_kill_stale=allow_kill_stale,
            allow_device_reset=allow_device_reset,
        )
    final_report = _scope_report_to_demo(_parse_pytest_report(), demo_dir)
    final_all_passed = bool(final_report.get("all_passed", False))
    final_passed = sorted(set(final_report.get("passed_components", []) or []))
    final_failed = sorted(set(final_report.get("failed_components", []) or []))
    final_skipped = sorted(set(final_report.get("skipped_components", []) or []))

    # Brain phantom-failure detection: ask the brain whether each
    # failure is a stale-decomposed-parent test. If yes, archive the
    # test file and re-run pytest. The mechanical safety net in
    # decomposition_consumer.py covers decomposition WRITE time;
    # this brain primitive covers OLD / external decompositions.
    _brain_post = _brain_handle_phantom_failures(
        MODEL=MODEL,
        demo_dir=demo_dir,
        final_failed=final_failed,
        banner_fn=banner,
        allow_kill_stale=allow_kill_stale,
        allow_device_reset=allow_device_reset,
    )
    if _brain_post is not None:
        final_rc, final_report = _brain_post["rc"], _brain_post["report"]
        final_all_passed = bool(final_report.get("all_passed", False))
        final_passed = sorted(set(final_report.get("passed_components", []) or []))
        final_failed = sorted(set(final_report.get("failed_components", []) or []))
        final_skipped = sorted(set(final_report.get("skipped_components", []) or []))

    # Brain-owned worktree → main-tree sync. Bypasses the overlay-
    # capture/apply mechanism (which has been observed to leave the
    # main tree with scaffold-stage stubs even when worktree had the
    # brain's iterated work product — SAM2, 2026-05-30).
    if final_all_passed:
        _brain_sync_graduated_to_main_tree(
            MODEL=MODEL,
            demo_dir=demo_dir,
            graduated_this_run=graduated_this_run,
            banner_fn=banner,
        )

    def _print_validation_breakdown() -> None:
        """Print what actually ran on hardware vs what's structurally-but-
        unvalidated, so the user can never confuse 'heuristic 100% native'
        with 'PCC-validated 100% native'."""
        try:
            new_names_set = {
                str(c.get("name", "")).strip()
                for c in (
                    json.loads((demo_dir / "bringup_status.json").read_text()).get("components", [])
                    if (demo_dir / "bringup_status.json").is_file()
                    else []
                )
                if c.get("status") == "NEW" and str(c.get("name", "")).strip()
            }
        except Exception:
            new_names_set = set()
        if not new_names_set:
            return
        unvalidated = sorted(
            new_names_set - set(final_passed) - set(final_failed) - set(final_skipped) - set(permanently_skipped)
        )
        print()
        print(sep)
        print("  NUMERICAL VALIDATION BREAKDOWN (final pytest on QB2):")
        print(sep)
        print(
            "  This is the ground truth from PCC tests. The compute split "
            "above is a STRUCTURAL heuristic (no torch fallback in forward) "
            "and may overstate what is truly working."
        )
        print()
        print(f"  PCC-validated on device : {len(final_passed)}/{len(new_names_set)}")
        if final_passed:
            for c in final_passed:
                print(f"      - {c}")
        print(f"  PCC FAILED (broken)     : {len(final_failed)}/{len(new_names_set)}")
        if final_failed:
            for c in final_failed:
                print(f"      - {c}")
        print(f"  SKIPPED (test scaffold) : {len(final_skipped)}/{len(new_names_set)}")
        if final_skipped:
            for c in final_skipped:
                per_skipped_map = final_report.get("per_skipped") or {}
                reason = ""
                if isinstance(per_skipped_map, dict):
                    for entry in per_skipped_map.values():
                        if isinstance(entry, dict) and entry.get("component") == c:
                            reason = str(entry.get("reason") or "").strip()
                            break
                if reason:
                    head = reason.splitlines()[0][:120]
                    print(f"      - {c}  ({head})")
                else:
                    print(f"      - {c}")
        print(f"  CPU fallback (capped)   : {len(permanently_skipped)}/{len(new_names_set)}")
        if permanently_skipped:
            for c in sorted(set(permanently_skipped)):
                print(f"      - {c}")
        if unverified_native_this_run:
            unverified_list = sorted(set(unverified_native_this_run))
            print(
                f"  UNVERIFIED NATIVE       : {len(unverified_list)}/{len(new_names_set)}  "
                f"(native ttnn forward, but PCC could not be measured — "
                f"test scaffold inputs incompatible with HF signature)"
            )
            for c in unverified_list:
                print(f"      - {c}")

        residual_unvalidated = [c for c in unvalidated if c not in unverified_native_this_run]
        if residual_unvalidated:
            print(f"  UNTESTED (no PCC run)   : {len(residual_unvalidated)}/{len(new_names_set)}")
            for c in residual_unvalidated:
                print(f"      - {c}")

    if final_all_passed and final_rc == 0:
        graduated_set = sorted(set(graduated_this_run))
        skipped_set = sorted(set(permanently_skipped))
        unverified_set = sorted(set(unverified_native_this_run))
        if unverified_set:
            banner(
                f"AUTO-ITERATE: bring-up NOT complete on {BOX} — "
                f"{len(unverified_set)} component(s) lack PCC validation"
            )
            print(
                "  The following component(s) have a native ttnn forward "
                "on disk but their PCC test SKIPPED because the auto-"
                "generated _make_arg_for() inputs are shape-incompatible "
                "with the HF reference's expected signature. Without a "
                "passing pcc(ttnn, torch) >= 0.99, we have NO proof of "
                "numerical correctness and cannot claim the model is "
                "running successfully on TT hardware."
            )
            for c in unverified_set:
                test_rel = demo_dir / "tests" / "pcc" / f"test_{_safe_id(c)}.py"
                try:
                    test_rel = safe_relative_to_root(test_rel)
                except Exception:
                    pass
                print(f"      - {c}  (hand-fix `{test_rel}`)")
            _print_bringup_summary(MODEL, box=BOX, sep=sep)
            _print_validation_breakdown()
            if graduated_set:
                print(f"  PCC-validated this run: {', '.join(graduated_set)}")
            if unverified_set:
                print(f"  UNVERIFIED NATIVE     : {', '.join(unverified_set)}")
            if skipped_set:
                print(f"  CPU fallback this run : {', '.join(skipped_set)}")
            print(
                "\n  To recover validation, edit each listed "
                "`tests/pcc/test_<comp>.py` so it constructs inputs that "
                "match the HF submodule's `forward(...)` signature "
                "(correct arg names, ranks, and dtypes), then re-run:\n"
                f"    python -m scripts.tt_hw_planner promote {MODEL} \\\n"
                f"        --box {BOX} --auto --auto-agent {provider} \\\n"
                f"        --auto-max-iters 12 --auto-agent-timeout 1500\n"
            )
            return 1
        ungraduated_now, _ = _auto_iteration_blockers(MODEL)
        ungraduated_now = sorted(set(ungraduated_now) - set(skipped_set))
        if not skipped_set and not ungraduated_now:
            banner(
                f"AUTO-ITERATE: demo runs natively end-to-end on {BOX} " f"after {max_iters} iter(s) (no CPU fallback)"
            )
        elif allow_partial_cpu:
            banner(
                f"AUTO-ITERATE: best-effort bring-up complete on {BOX} — "
                f"{len(graduated_set)} native, "
                f"{len(skipped_set) + len(ungraduated_now)} on CPU fallback "
                f"(--allow-partial-cpu was set)"
            )
        else:
            banner(
                f"AUTO-ITERATE: PARTIAL — {len(graduated_set)} component(s) "
                f"native on {BOX}; "
                f"{len(skipped_set) + len(ungraduated_now)} still on CPU fallback. "
                f"Re-run with --allow-partial-cpu to accept this as SUCCESS, "
                f"or with --auto-max-iters >= 24 to push the remaining "
                f"components to native."
            )
        _print_bringup_summary(MODEL, box=BOX, sep=sep)
        _print_validation_breakdown()
        if graduated_set:
            print(f"  native TTNN  this run: {', '.join(graduated_set)}")
        if skipped_set or ungraduated_now:
            cpu_components = sorted(set(skipped_set) | set(ungraduated_now))
            print(f"  CPU fallback this run: {', '.join(cpu_components)}")
            print(
                "\n  Retry CPU-fallback components later (re-scaffolding not needed):\n"
                f"    python -m scripts.tt_hw_planner promote {MODEL} \\\n"
                f"        --box {BOX} --auto --auto-agent {provider} \\\n"
                f"        --auto-max-iters 12 --auto-agent-timeout 1500\n"
            )
        if (skipped_set or ungraduated_now) and not allow_partial_cpu:
            return 1
        return 0

    banner(f"AUTO-ITERATE exhausted {max_iters} iter(s); final pytest did not pass even with CPU fallback")
    print("  This is unusual — the CPU-fallback stubs are deterministic and should pass PCC trivially.")
    print("  Final failure tail:")
    print(last_failures)
    _print_bringup_summary(MODEL, box=BOX, sep=sep)
    _print_validation_breakdown()
    print(
        "\n  Inspect _stubs/<comp>.py for the offending component(s) and either\n"
        "  hand-iterate or rerun with a fresh scaffold:\n"
        f"    python -m scripts.tt_hw_planner promote {MODEL} --box {BOX} \\\n"
        "        --auto --auto-agent <cursor|claude> --auto-max-iters <N>\n"
    )
    return 1
