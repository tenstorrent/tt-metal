from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _runtime_repair_loop(
    *,
    model_id: str,
    prepare_argv: argparse.Namespace,
    initial_rc: int,
    initial_output: str,
    agent_bin: str,
    agent_model: str,
    max_iters: int,
    agent_timeout_s: int,
    sep: str,
    model_light: Optional[str] = None,
    model_heavy: Optional[str] = None,
    model_super_heavy: Optional[str] = None,
) -> int:
    """LLM-driven runtime-repair driver for fast-path failures.

    The contract:
      * Called only when ``--auto`` is set AND the fast-path
        ``cmd_prepare --execute`` returned non-zero AND the failure
        is classified as repairable (Python error in ``models/``).
      * Iterates up to ``max_iters`` times: parse traceback -> ask
        agent to patch the offending file -> re-run cmd_prepare.
      * Returns 0 on graduation (pytest passes), or the latest
        non-zero rc on exhaustion.
      * Echoes a one-line audit summary per iteration so the user
        can see what was tried.

    All caller-side state (banner, learning loop, exit code) is
    handled by the caller; this driver just returns the final rc."""
    from ..cli import (
        REPO_ROOT,
        _build_forced_edit_preamble,
        _git_worktree_diff_hash,
        _invoke_agent,
        _pick_agent_model_for_iter,
        _run_prepare_capture,
    )
    from ..runtime_repair import (
        parse_pytest_traceback,
        is_repairable_failure,
        build_repair_prompt,
    )

    print()
    print(sep)
    print("  RUNTIME-REPAIR LOOP  (fast-path failed; entering LLM iterate)")
    print(sep)
    print(f"  model        : {model_id}")
    print(f"  max_iters    : {max_iters}")
    print(f"  agent_bin    : {agent_bin}")
    print(f"  agent_model  : {agent_model}")
    print(f"  agent_budget : {agent_timeout_s}s per iter")
    print(sep)

    info = parse_pytest_traceback(initial_output)
    repairable, reason = is_repairable_failure(info)
    if not repairable:
        # WIRING #1 (Path 2 parity): the failure isn't repairable —
        # classify it and persist as a known skip so the next run
        # doesn't repeat the wasted work. Mirrors auto_iterate.py's
        # behavior when classify_failure returns KERNEL_MISSING.
        _classify_and_persist_skip(
            model_id=model_id,
            captured_output=initial_output,
            reason_hint=f"not-repairable: {reason}",
        )
        print(
            f"  Not repairable: {reason}\n"
            f"  Falling back to the standard FAIL banner. The original\n"
            f"  pytest error stands; see the suggested-next-steps block."
        )
        return initial_rc

    print(f"  Initial failure: {info.exception_type} at " f"{info.failure_file}:{info.failure_line}")
    if info.exception_message:
        print(f"    msg: {info.exception_message}")

    # WIRING #3 (Path 2 parity): consult learned-fix store. If a prior
    # run registered a fix for the same arch+failure signature, surface
    # it as a head-start preamble — same machinery auto_iterate uses.
    _learned_head_start = _brain_lookup_learned_fix_head_start(
        model_id=model_id,
        failure_file=info.failure_file or "",
        exception_type=info.exception_type or "",
    )
    if _learned_head_start:
        print(
            f"  [brain G8] learned-fix lookup: HIT — prior run had a "
            f"working fix for this failure signature. Injecting as prompt "
            f"preamble.\n  {_learned_head_start[:200]}..."
        )

    previous_attempts: List[str] = []
    current_rc = initial_rc
    current_output = initial_output

    consecutive_no_edit_iters = 0
    # WIRING #9 (Path 2 parity): track mismatch history for is_stagnant
    # — so the loop can use the brain's plateau detector instead of a
    # hardcoded "consecutive_no_edit_iters >= 2" cap.
    _mismatch_history: List[float] = []
    for iter_idx in range(1, max_iters + 1):
        print()
        print(sep)
        print(f"  REPAIR ITER {iter_idx}/{max_iters}  -- " f"asking agent to patch {info.failure_file}")
        print(sep)

        _iter_model, _iter_model_reason = _pick_agent_model_for_iter(
            model_default=agent_model,
            model_light=model_light,
            model_heavy=model_heavy,
            model_super_heavy=model_super_heavy,
            complexity_bonus=0,
            failure_class=info.exception_type or "",
            attempts_so_far=iter_idx - 1,
            force_heavy=consecutive_no_edit_iters >= 1,
        )
        if (model_light or model_heavy or model_super_heavy) and _iter_model_reason != "default":
            print(f"  [auto:runtime-repair] tiered model pick: " f"{_iter_model} ({_iter_model_reason})")
        forced_edit_mode = consecutive_no_edit_iters >= 1

        pre_iter_diff_hash = _git_worktree_diff_hash()
        prompt = build_repair_prompt(
            model_id=model_id,
            info=info,
            iter_idx=iter_idx,
            max_iters=max_iters,
            previous_attempts=previous_attempts,
        )
        if forced_edit_mode:
            prompt = _build_forced_edit_preamble(iter_idx) + "\n\n" + prompt
        agent_log_dir = REPO_ROOT / "generated" / "tt_hw_planner"
        agent_log_dir.mkdir(parents=True, exist_ok=True)
        provider = "claude" if agent_bin.endswith("claude") or agent_bin == "claude" else "cursor"
        try:
            from .agent import _bringup_cwd as _bcwd

            agent_rc = _invoke_agent(
                prompt,
                provider=provider,
                agent_bin=agent_bin,
                cwd=_bcwd(),
                model=_iter_model,
                timeout_s=agent_timeout_s,
                iter_tag=f"repair_iter_{iter_idx}",
                require_edit_progress=forced_edit_mode,
            )
        except Exception as exc:
            print(
                f"  agent invocation failed: {type(exc).__name__}: {exc}.\n"
                f"  Aborting repair loop; returning the latest pytest rc."
            )
            return current_rc
        if agent_rc not in (0, None):
            print(
                f"  agent exited with rc={agent_rc} (non-zero). The agent\n"
                f"  may have hit its own budget or refused to patch.\n"
                f"  Re-running pytest anyway in case partial edits were\n"
                f"  applied; if pytest still fails, will retry."
            )

        post_iter_diff_hash = _git_worktree_diff_hash()
        if pre_iter_diff_hash == post_iter_diff_hash:
            consecutive_no_edit_iters += 1
            print(
                f"  AGENT MADE NO FILE CHANGES this iteration "
                f"(working-tree hash unchanged; consecutive no-edit "
                f"iters: {consecutive_no_edit_iters}). Next iter will "
                f"escalate to the heavy model (if tiered) and inject "
                f"a forced-edit preamble."
            )
            # WIRING #9: consult the brain's is_stagnant detector on
            # top of the local "no edits" counter. Brain's plateau
            # detector uses a mismatch-history trend; for runtime-repair
            # we synthesize a synthetic history (1.0 = full failure)
            # because there's no PCC progress signal here — but the
            # "no edits" counter is itself a plateau signal we can feed.
            _mismatch_history.append(1.0)
            from ..agentic.convergence import is_stagnant as _brain_is_stagnant

            _brain_says_stuck = len(_mismatch_history) >= 3 and _brain_is_stagnant(_mismatch_history)
            if consecutive_no_edit_iters >= 2 or _brain_says_stuck:
                print()
                print(sep)
                print(
                    f"  RUNTIME-REPAIR LOOP TERMINATED EARLY: "
                    f"{consecutive_no_edit_iters} consecutive iters "
                    f"made zero edits despite forced-edit preamble "
                    f"and heavy-model escalation"
                    f"{' (brain is_stagnant CONFIRMED plateau)' if _brain_says_stuck else ''}. "
                    f"The agent is stuck on this problem; continuing "
                    f"would waste budget. Latest failure: "
                    f"{info.exception_type} at "
                    f"{info.failure_file}:{info.failure_line}"
                )
                print(sep)
                # WIRING #1 cont: classify + persist the stuck failure
                # as a known skip.
                _classify_and_persist_skip(
                    model_id=model_id,
                    captured_output=current_output,
                    reason_hint=(
                        f"runtime-repair terminated early: stuck on " f"{info.exception_type}@{info.failure_file}"
                    ),
                )
                return current_rc
        else:
            consecutive_no_edit_iters = 0
        previous_attempts.append(
            f"asked agent to patch {info.failure_file}:{info.failure_line} " f"({info.exception_type})"
        )

        if forced_edit_mode:
            prompt = prompt  # noqa: PLW0127 — placeholder kept for clarity
        # WIRING #3 cont: inject the learned-fix preamble (if any) on
        # the very first iter only. Subsequent iters use the in-loop
        # previous_attempts as feedback.
        if iter_idx == 1 and _learned_head_start:
            prompt = _learned_head_start + "\n\n" + prompt

        print()
        print(f"  re-running prepare --execute after repair attempt {iter_idx} …")
        current_rc, current_output = _run_prepare_capture(prepare_argv)
        if current_rc == 0:
            print()
            print(sep)
            print(f"  REPAIR LOOP GRADUATED at iter {iter_idx} -- pytest " f"now passes.")
            print(sep)
            # WIRING #2 (Path 2 parity): persist the successful fix so
            # next run with the same arch+failure-signature can apply
            # the learned head-start immediately. Mirrors auto_iterate.
            _brain_register_learned_fix(
                model_id=model_id,
                failure_file=info.failure_file or "",
                exception_type=info.exception_type or "",
                iter_idx=iter_idx,
            )
            return 0

        new_info = parse_pytest_traceback(current_output)
        if not new_info.is_parseable:
            print(
                f"  iter {iter_idx} produced an un-parseable failure; "
                f"stopping the loop to avoid wasting iterations."
            )
            return current_rc

        if (
            new_info.failure_file == info.failure_file
            and new_info.failure_line == info.failure_line
            and new_info.exception_type == info.exception_type
        ):
            print(
                f"  iter {iter_idx} did not change the failure site "
                f"({info.exception_type} at {info.failure_file}:"
                f"{info.failure_line}). Re-prompting with the previous-"
                f"attempts log so the agent doesn't repeat itself."
            )
        else:
            print(
                f"  iter {iter_idx} moved the failure: "
                f"{info.exception_type}@{info.failure_file}:"
                f"{info.failure_line} -> {new_info.exception_type}@"
                f"{new_info.failure_file}:{new_info.failure_line}. "
                f"Progress; continuing."
            )
        info = new_info
        repairable, reason = is_repairable_failure(info)
        if not repairable:
            print(f"  iter {iter_idx} hit a non-repairable failure: " f"{reason}. Stopping the loop.")
            return current_rc

    print()
    print(sep)
    print(
        f"  REPAIR LOOP EXHAUSTED  ({max_iters} iters consumed, pytest "
        f"still failing). Latest failure: {info.exception_type} at "
        f"{info.failure_file}:{info.failure_line}."
    )
    print(sep)
    # WIRING #1 cont: budget exhausted — classify + persist.
    _classify_and_persist_skip(
        model_id=model_id,
        captured_output=current_output,
        reason_hint=(
            f"runtime-repair exhausted {max_iters} iters; last failure " f"{info.exception_type}@{info.failure_file}"
        ),
    )
    return current_rc


def _classify_and_persist_skip(
    *,
    model_id: str,
    captured_output: str,
    reason_hint: str,
) -> None:
    """Thin shim around failure_classifier.classify_and_persist_skip.

    Kept as a thin wrapper because the runtime_repair call sites use
    the old kwarg name `captured_output`. The SHARED implementation
    (consolidated 2026-05-31) lives in failure_classifier.py.
    """
    from ..failure_classifier import classify_and_persist_skip

    classify_and_persist_skip(
        model_id=model_id,
        failure_text=captured_output,
        reason_hint=reason_hint,
    )


def _arch_signature_for_model(model_id: str) -> str:
    """Compute the arch_signature for ``model_id`` by loading its HF
    config and delegating to ``agentic.learnings.compute_arch_signature``.

    Returns '' on any error so the caller can no-op cleanly."""
    try:
        from ..agentic.executor import _load_hf_config
        from ..agentic.learnings import compute_arch_signature

        cfg = _load_hf_config(model_id)
        return compute_arch_signature(cfg)
    except Exception:
        return ""


def _brain_lookup_learned_fix_head_start(
    *,
    model_id: str,
    failure_file: str,
    exception_type: str,
) -> str:
    """Helper for WIRING #3: consult agentic.learnings.lookup_fix and
    return a prompt-preamble string if a prior run registered a working
    fix for this signature. Returns '' on miss or any error."""
    if not failure_file or not exception_type:
        return ""
    try:
        from ..agentic.learnings import lookup_fix

        arch_sig = _arch_signature_for_model(model_id)
        if not arch_sig:
            return ""
        learned = lookup_fix(
            arch_signature=arch_sig,
            first_diverging_qn=failure_file,
        )
        if learned is None:
            return ""
        return (
            "LEARNED-FIX HEAD-START (from a prior successful repair run):\n"
            "A previous run with this same architecture signature fixed a similar\n"
            f"failure at {failure_file}. The recorded fix:\n\n"
            f"{getattr(learned, 'diff', '(no diff captured)')[:1500]}\n\n"
            "Use this as a starting hint — it may not apply verbatim, but the\n"
            "general approach (which symbols changed) is likely relevant."
        )
    except Exception:
        return ""


def _brain_register_learned_fix(
    *,
    model_id: str,
    failure_file: str,
    exception_type: str,
    iter_idx: int,
) -> None:
    """Helper for WIRING #2: persist the successful fix to
    agentic.learnings so next-run head-start works. Mirrors what
    auto_iterate does on per-component graduation."""
    if not failure_file:
        return
    try:
        from ..agentic.executor import compute_diff
        from ..agentic.learnings import register_fix
        from ..cli import REPO_ROOT

        arch_sig = _arch_signature_for_model(model_id)
        if not arch_sig:
            return
        diff_text = ""
        try:
            diff_text = compute_diff(REPO_ROOT, [failure_file])
        except Exception:
            diff_text = ""
        register_fix(
            arch_signature=arch_sig,
            first_diverging_qn=failure_file,
            diff=diff_text,
            diff_files=[failure_file],
            source_model_id=model_id,
            notes=(f"runtime-repair graduated at iter {iter_idx} for " f"{exception_type or 'unknown-exc'}"),
        )
        print(
            f"  [brain G8] registered learned fix for "
            f"{arch_sig[:32]}@{failure_file} — future bring-ups of "
            f"this architecture get the fix as a head-start"
        )
    except Exception as exc:
        print(
            f"  [brain G8] register_fix non-fatal: " f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


_PCC_FAIL_RC: int = 17
