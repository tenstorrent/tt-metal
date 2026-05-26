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
        print(
            f"  Not repairable: {reason}\n"
            f"  Falling back to the standard FAIL banner. The original\n"
            f"  pytest error stands; see the suggested-next-steps block."
        )
        return initial_rc

    print(f"  Initial failure: {info.exception_type} at " f"{info.failure_file}:{info.failure_line}")
    if info.exception_message:
        print(f"    msg: {info.exception_message}")

    previous_attempts: List[str] = []
    current_rc = initial_rc
    current_output = initial_output

    consecutive_no_edit_iters = 0
    for iter_idx in range(1, max_iters + 1):
        print()
        print(sep)
        print(f"  REPAIR ITER {iter_idx}/{max_iters}  -- " f"asking agent to patch {info.failure_file}")
        print(sep)

        _iter_model, _iter_model_reason = _pick_agent_model_for_iter(
            model_default=agent_model,
            model_light=model_light,
            model_heavy=model_heavy,
            complexity_bonus=0,
            failure_class=info.exception_type or "",
            attempts_so_far=iter_idx - 1,
            force_heavy=consecutive_no_edit_iters >= 1,
        )
        if (model_light or model_heavy) and _iter_model_reason != "default":
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
            if consecutive_no_edit_iters >= 2:
                print()
                print(sep)
                print(
                    f"  RUNTIME-REPAIR LOOP TERMINATED EARLY: "
                    f"{consecutive_no_edit_iters} consecutive iters "
                    f"made zero edits despite forced-edit preamble "
                    f"and heavy-model escalation. The agent is stuck "
                    f"on this problem; continuing would waste budget. "
                    f"Latest failure: {info.exception_type} at "
                    f"{info.failure_file}:{info.failure_line}"
                )
                print(sep)
                return current_rc
        else:
            consecutive_no_edit_iters = 0
        previous_attempts.append(
            f"asked agent to patch {info.failure_file}:{info.failure_line} " f"({info.exception_type})"
        )

        print()
        print(f"  re-running prepare --execute after repair attempt {iter_idx} …")
        current_rc, current_output = _run_prepare_capture(prepare_argv)
        if current_rc == 0:
            print()
            print(sep)
            print(f"  REPAIR LOOP GRADUATED at iter {iter_idx} -- pytest " f"now passes.")
            print(sep)
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
    return current_rc


_PCC_FAIL_RC: int = 17
