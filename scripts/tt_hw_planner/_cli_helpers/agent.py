from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..discovery import safe_relative_to_root


def resolve_claude_bin() -> Optional[str]:
    """Resolve the `claude` CLI to an absolute path (fixes-plan Point 9).

    Fallback chain (universal, arch-independent): TT_PLANNER_AGENT_BIN env ->
    CLAUDE_BIN env -> ``shutil.which("claude")`` on PATH -> ``~/.local/bin/claude``
    if present. Returns None if none resolve, so callers can fail fast once with a
    clear message instead of a late opaque FileNotFoundError from a bare spawn.
    Makes every spawn point PATH-independent (login/non-login shells, cron, CI).
    """
    local = os.path.expanduser("~/.local/bin/claude")
    return (
        os.environ.get("TT_PLANNER_AGENT_BIN")
        or os.environ.get("CLAUDE_BIN")
        or shutil.which("claude")
        or (local if os.path.exists(local) else None)
    )


def require_claude_bin() -> str:
    """resolve_claude_bin() or a single clear error (Point 9 fail-fast)."""
    b = resolve_claude_bin()
    if not b:
        raise RuntimeError("claude CLI not found — install it or set CLAUDE_BIN / TT_PLANNER_AGENT_BIN")
    return b


def _verbose() -> bool:
    """Screen-verbosity gate (matches cli.py's TT_HW_PLANNER_VERBOSE convention).
    Off by default: keep the terminal clean; full detail always lands in the logs."""
    return os.environ.get("TT_HW_PLANNER_VERBOSE", "") not in ("", "0", "false", "False")


def _bringup_cwd() -> Path:
    env = os.environ.get("TT_HW_PLANNER_BRINGUP_CWD")
    if env:
        return Path(env)
    from ..discovery import REPO_ROOT, safe_relative_to_root

    return REPO_ROOT


def _extract_agent_result_text(agent_log: Path) -> Optional[str]:
    """Scan an agent stdout log for the structured `"result":"..."` line
    the Claude CLI emits on success and return the unquoted result text.

    The Claude session emits one line per event as JSON (or as a `cli_output
    |` prefixed line). The final SUCCESS event has shape
    ``{"type":"result","subtype":"success","is_error":false,...,
       "result":"<text>",...}``. We pull `result` out so the brain can
    persist what the prior agent said it did, and feed it into the next
    iter's prompt as cross-iter memory.

    Returns ``None`` if the log doesn't exist, doesn't contain a parseable
    result line, or the result event is missing. Best-effort; failures
    are swallowed because this is decoration on the attempt log.
    """
    try:
        if not agent_log.is_file():
            return None
        text = agent_log.read_text(errors="ignore")
    except Exception:
        return None
    # The Claude CLI streams JSON events; the final `result` event is what
    # we want. There may be multiple `"result":` substrings in the file (e.g.
    # quoted inside an earlier message) — we scan for the LAST event whose
    # outer envelope is `"type":"result"` and pull its `result` string.
    candidates: List[str] = []
    for m in re.finditer(r'\{"type":"result"[^\n]*\}', text):
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
        except Exception:
            continue
        r = obj.get("result")
        if isinstance(r, str) and r.strip():
            candidates.append(r)
    if candidates:
        return candidates[-1]
    # Fallback: simple regex if the JSON envelope isn't a strict one-liner.
    m = re.search(r'"result"\s*:\s*"((?:[^"\\]|\\.)*)"', text[-50000:])
    if m:
        try:
            return json.loads('"' + m.group(1) + '"')
        except Exception:
            return m.group(1)
    return None


def _invoke_agent(
    prompt: str,
    *,
    provider: str,
    agent_bin: str,
    cwd: Path,
    model: str,
    timeout_s: int = 600,
    complexity_bonus: int = 0,
    iter_tag: Optional[str] = None,
    deliverable_dirs: Optional[List[Path]] = None,
    expected_deliverable_files: Optional[List[Path]] = None,
    require_edit_progress: bool = False,
) -> int:
    """Invoke an external LLM agent (claude / cursor) with the given
    prompt and return its exit code.

    Robustness fixes implemented 2026-05-22 after a SAM2-hiera-small
    bring-up burned two full 15-min iters on EMPTY_AGENT failures:

    - **stream-json output** for claude, parsed event-by-event in
      the heartbeat loop. Previously used `--output-format text`,
      which prints nothing until the agent's final response — so the
      heartbeat saw `log=quiet` even while claude was actively
      reading 4.5 MB of source code via its `Read` tool. The loop
      then killed claude mid-investigation. Now every tool call is
      visible.

    - **proc/-walking kill** that finds descendants the parent's
      process group can't reach. The claude CLI internally
      fork()+exit()s its wrapper, leaving the actual worker
      reparented to init the moment we send SIGTERM. `killpg` only
      hits the wrapper; the worker keeps making API calls
      (consuming rate limit) until manual cleanup. Now we walk the
      kernel children table BEFORE signaling and send SIGTERM/
      SIGKILL to every descendant by PID directly.

    - **Stall detection** that's separate from wall-clock. If we go
      `stall_budget_s` (default min(180, timeout/4)) with NO new
      stream-json events AND NO /proc/io read activity, we kill
      early instead of waiting for the full timeout. Catches "agent
      is genuinely hung" without burning the whole budget.

    - **Complexity-aware timeout** via `_agent_complexity_timeout`.
      A component with complexity_bonus=4 (>80 ops + LLM gaps) gets
      `base + 20min` instead of the user's flat 15-min budget.

    - **Per-iter log preservation**. If `iter_tag` is provided, the
      log filename includes it (`claude_iter_1.log`) so iter 1's
      log isn't overwritten by iter 2's. Useful for post-mortem on
      multi-iter bring-ups.

    - **Async stdin write** in a daemon thread so a 100KB prompt
      doesn't block the heartbeat loop on the 64 KiB Linux pipe
      buffer."""
    from ..cli import (
        REPO_ROOT,
        _agent_complexity_timeout,
        _deliverable_changed,
        _kill_agent_tree,
        _parse_stream_json_event,
        _read_proc_rchar,
        _snapshot_deliverable_state,
        _summarize_stream_json_event,
    )
    import builtins as _bi
    import subprocess
    import threading

    effective_timeout_s = _agent_complexity_timeout(timeout_s, complexity_bonus)

    prompt_via_stdin = False
    if provider == "cursor":
        cmd = [
            agent_bin,
            "-p",
            "--force",
            "--trust",
            "--workspace",
            str(cwd),
            "--model",
            model,
            "--output-format",
            "text",
            prompt,
        ]
        cli_supports_stream_json = False
    elif provider == "claude":
        cmd = [
            agent_bin,
            "-p",
            "--dangerously-skip-permissions",
            "--add-dir",
            str(cwd),
            "--model",
            model,
            "--output-format",
            "stream-json",
            "--verbose",
            "--tools",
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Grep",
        ]
        prompt_via_stdin = True
        cli_supports_stream_json = True
    else:
        _bi.print(f"  unknown agent provider {provider!r}", file=sys.stderr)
        return 2

    # Every `[auto:<provider>] ...` line below is tool-progress narration about
    # the agent subprocess — NOT the agent's own output (that streams to the
    # _handoff/<provider>_*.log file). Keep the terminal clean by shadowing
    # print() so this status goes to screen only under TT_HW_PLANNER_VERBOSE=1.
    # Genuine errors (file=sys.stderr) always show. Placed AFTER the provider
    # block on purpose: `def print` makes the name function-local, so any
    # print() call above it would UnboundLocalError (the unknown-provider error
    # uses _bi.print for that reason), and it keeps the claude branch inside the
    # span that test_invoke_agent_uses_stream_json_for_claude delimits by "next def".
    def print(*a, **k):  # noqa: A001 - intentional local shadow to gate status output
        if _verbose() or k.get("file") is sys.stderr:
            _bi.print(*a, **k)

    if effective_timeout_s != timeout_s:
        budget_str = (
            f"{effective_timeout_s}s ({effective_timeout_s // 60} min, "
            f"bumped from {timeout_s}s due to complexity "
            f"+{complexity_bonus})"
        )
    else:
        budget_str = f"{timeout_s}s ({timeout_s // 60} min)" if timeout_s > 0 else "unbounded"
    print(f"\n  [auto:{provider}] invoking {agent_bin} (model={model}, budget={budget_str}) ...")
    if _verbose():
        print(f"  [auto:{provider}] cmd: {' '.join(cmd)}")

    log_name = f"{provider}_{iter_tag}.log" if iter_tag else f"{provider}_last_run.log"
    agent_log = cwd / "_handoff" / log_name
    log_fh = None
    try:
        agent_log.parent.mkdir(parents=True, exist_ok=True)

        log_fh = open(agent_log, "w", buffering=1)
        try:
            display_log = safe_relative_to_root(agent_log)
        except Exception:
            display_log = agent_log
        print(f"  [auto:{provider}] stdout/stderr -> {display_log}")
    except Exception as exc:
        print(
            f"  [auto:{provider}] could not open log file {agent_log} "
            f"({exc}); subprocess output will be discarded.",
            file=sys.stderr,
        )
        log_fh = None

    if prompt_via_stdin:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=log_fh if log_fh else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
            text=True,
        )

        def _write_prompt_async() -> None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(prompt)
                proc.stdin.close()
            except Exception as exc:
                print(
                    f"  [auto:{provider}] could not write prompt to " f"stdin: {exc}",
                    file=sys.stderr,
                )

        prompt_thread = threading.Thread(target=_write_prompt_async, daemon=True)
        prompt_thread.start()
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=log_fh if log_fh else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
        )

    start = time.monotonic()
    heartbeat_s = 15
    last_log_size = 0
    last_rchar: Optional[int] = None
    last_progress_t = start
    last_parsed_pos = 0
    counts: Dict[str, int] = {
        "tool_use": 0,
        "assistant": 0,
        "result": 0,
        "error": 0,
        "other": 0,
        "edit_count": 0,
        "read_count": 0,
    }

    REQUIRE_EDIT_DEADLINE_FRAC = 0.7
    READ_LIMIT_FOR_NO_EDIT_KILL = 5
    require_edit_warning_emitted = False

    stall_budget_s = max(180, min(300, effective_timeout_s // 4))

    deliverable_dirs_resolved: List[Path] = list(deliverable_dirs or [])
    expected_files_resolved: Optional[List[Path]] = (
        list(expected_deliverable_files) if expected_deliverable_files else None
    )
    deliverable_baseline = _snapshot_deliverable_state(
        deliverable_dirs_resolved, expected_files=expected_files_resolved
    )
    deliverable_deadline_s: int = int(effective_timeout_s * 0.8) if effective_timeout_s > 0 else 0
    deliverable_written = False
    deliverable_warning_emitted = False

    heartbeat_screen_s = 60
    last_heartbeat_t = 0.0  # 0.0 => first heartbeat prints promptly, then throttled

    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                elapsed = int(time.monotonic() - start)
                if log_fh:
                    try:
                        log_fh.flush()
                    except Exception:
                        pass
                tail = ""
                try:
                    if agent_log.is_file():
                        tail = agent_log.read_text(errors="ignore")[-2000:]
                except Exception:
                    pass
                print(
                    f"  [auto:{provider}] agent finished in {elapsed}s "
                    f"(exit={rc}) — events: "
                    f"tool_use={counts['tool_use']} "
                    f"asst={counts['assistant']} "
                    f"result={counts['result']} "
                    f"err={counts['error']}"
                )
                if tail.strip():
                    if _verbose():
                        print(f"  [auto:{provider}] last log output:")
                        for line in tail.splitlines()[-25:]:
                            print(f"    | {line}")
                    else:
                        # Clean screen: one human-readable result line instead of
                        # the raw stream-json tail (the full log is at the path
                        # printed above). Surface a hint on non-zero exit.
                        result_text = _extract_agent_result_text(agent_log)
                        if result_text and result_text.strip():
                            first = result_text.strip().splitlines()[0][:200]
                            print(f"  [auto:{provider}] result: {first}")
                        elif rc != 0:
                            print(f"  [auto:{provider}] non-zero exit ({rc}); see log for details")
                return rc

            elapsed = int(time.monotonic() - start)

            cur_size = 0
            try:
                if agent_log.is_file():
                    cur_size = agent_log.stat().st_size
            except Exception:
                pass
            if cur_size > last_log_size:
                last_progress_t = time.monotonic()

            cur_rchar = _read_proc_rchar(proc.pid)
            if cur_rchar is not None:
                if last_rchar is not None and cur_rchar > last_rchar:
                    last_progress_t = time.monotonic()
                last_rchar = cur_rchar

            new_events_this_tick = 0
            if cli_supports_stream_json:
                try:
                    with open(agent_log, "r", errors="ignore") as f:
                        f.seek(last_parsed_pos)
                        new_text = f.read()
                        last_parsed_pos = f.tell()
                except Exception:
                    new_text = ""
                for raw in new_text.splitlines():
                    evt = _parse_stream_json_event(raw)
                    if evt is None:
                        continue
                    _summarize_stream_json_event(evt, counts)
                    new_events_this_tick += 1
                if new_events_this_tick > 0:
                    last_progress_t = time.monotonic()

            last_log_size = cur_size

            if effective_timeout_s > 0 and elapsed >= effective_timeout_s:
                print(
                    f"  [auto:{provider}] WALL-CLOCK BUDGET EXHAUSTED "
                    f"at {elapsed}s (limit={effective_timeout_s}s); "
                    f"killing agent process tree."
                )
                _kill_agent_tree(proc, provider=provider)
                tail = ""
                try:
                    if log_fh:
                        log_fh.flush()
                    if agent_log.is_file():
                        tail = agent_log.read_text(errors="ignore")[-2000:]
                except Exception:
                    pass
                if tail.strip():
                    print(f"  [auto:{provider}] last log output before kill:")
                    for line in tail.splitlines()[-25:]:
                        print(f"    | {line}")
                else:
                    print(
                        f"  [auto:{provider}] agent produced NO stdout "
                        f"in {elapsed}s — likely the model never "
                        f"started generating. Common causes:\n"
                        f"    - bad/expired API key "
                        f"(ANTHROPIC_API_KEY or claude login)\n"
                        f"    - network/auth proxy issue "
                        f"(check {agent_log})\n"
                        f"    - model name '{model}' rejected by "
                        f"the CLI"
                    )
                print(
                    f"  [auto:{provider}] outer loop will continue "
                    f"with whatever the agent managed to write "
                    f"so far. Events seen: tool_use="
                    f"{counts['tool_use']} asst={counts['assistant']} "
                    f"result={counts['result']} err={counts['error']}"
                )
                return 124

            stall_age = time.monotonic() - last_progress_t
            if elapsed > stall_budget_s and stall_age > stall_budget_s:
                print(
                    f"  [auto:{provider}] STALL DETECTED: no progress "
                    f"signal in {int(stall_age)}s "
                    f"(stall_budget={stall_budget_s}s); killing "
                    f"agent process tree. Final event counts: "
                    f"tool_use={counts['tool_use']} "
                    f"asst={counts['assistant']} "
                    f"result={counts['result']} "
                    f"err={counts['error']}."
                )
                _kill_agent_tree(proc, provider=provider)
                return 124

            if (
                require_edit_progress
                and effective_timeout_s > 0
                and elapsed >= int(effective_timeout_s * REQUIRE_EDIT_DEADLINE_FRAC)
                and counts.get("edit_count", 0) == 0
                and counts.get("read_count", 0) >= READ_LIMIT_FOR_NO_EDIT_KILL
            ):
                print(
                    f"  [auto:{provider}] REQUIRE-EDIT EARLY-KILL at "
                    f"{elapsed}s ({elapsed * 100 // effective_timeout_s}% "
                    f"of budget): agent has done "
                    f"{counts['read_count']} Read-class tool calls and "
                    f"ZERO Edit-class tool calls. The caller demanded "
                    f"a forced-edit iter; killing now so the outer "
                    f"loop can move on instead of waiting for the "
                    f"full wall clock."
                )
                _kill_agent_tree(proc, provider=provider)
                return 124
            elif (
                require_edit_progress
                and not require_edit_warning_emitted
                and effective_timeout_s > 0
                and elapsed >= effective_timeout_s // 2
                and counts.get("edit_count", 0) == 0
                and counts.get("read_count", 0) >= 3
            ):
                require_edit_warning_emitted = True
                print(
                    f"  [auto:{provider}] WARNING (forced-edit iter): "
                    f"{counts['read_count']} Read calls but ZERO "
                    f"Edit calls at {elapsed}s "
                    f"({elapsed * 100 // effective_timeout_s}% of "
                    f"budget). If no Edit is committed by "
                    f"{int(effective_timeout_s * REQUIRE_EDIT_DEADLINE_FRAC)}s "
                    f"the loop will EARLY-KILL this iter."
                )

            if deliverable_dirs_resolved and not deliverable_written:
                current_state = _snapshot_deliverable_state(
                    deliverable_dirs_resolved,
                    expected_files=expected_files_resolved,
                )
                if _deliverable_changed(deliverable_baseline, current_state):
                    deliverable_written = True
                    print(
                        f"  [auto:{provider}] deliverable file written " f"at {elapsed}s — agent is producing output."
                    )
                else:
                    if (
                        not deliverable_warning_emitted
                        and effective_timeout_s > 0
                        and elapsed >= effective_timeout_s // 2
                    ):
                        deliverable_warning_emitted = True
                        print(
                            f"  [auto:{provider}] WARNING: agent has not "
                            f"written ANY response file yet at {elapsed}s "
                            f"({elapsed * 100 // effective_timeout_s}% of "
                            f"budget). If no file is written by "
                            f"{deliverable_deadline_s}s, will EARLY-KILL "
                            f"so the iter can move on instead of "
                            f"consuming the full wall clock."
                        )

                    if deliverable_deadline_s > 0 and elapsed >= deliverable_deadline_s:
                        print(
                            f"  [auto:{provider}] DELIVERABLE DEADLINE: "
                            f"no response file written in {elapsed}s "
                            f"(deadline={deliverable_deadline_s}s, "
                            f"{elapsed * 100 // effective_timeout_s}% of "
                            f"budget). Killing agent — iter is "
                            f"unrecoverable. Event counts: "
                            f"tool_use={counts['tool_use']} "
                            f"asst={counts['assistant']} "
                            f"result={counts['result']} "
                            f"err={counts['error']}."
                        )
                        _kill_agent_tree(proc, provider=provider)
                        return 124

            growth = cur_size - last_log_size if cur_size > 0 else 0
            growth_note = f"log+={growth}B" if growth > 0 else "log=quiet"
            if cli_supports_stream_json:
                summary = (
                    f"tool_use={counts['tool_use']} "
                    f"asst={counts['assistant']} "
                    f"+{new_events_this_tick}evt "
                    f"stall_age={int(stall_age)}s"
                )
            else:
                summary = f"stall_age={int(stall_age)}s"
            if deliverable_dirs_resolved:
                summary += f" deliverable={'YES' if deliverable_written else 'NO'}"
            _hb_now = time.monotonic()
            # Rate-limit the on-screen heartbeat to once per heartbeat_screen_s.
            # (The previous `events-changed OR stale` condition printed on nearly
            # every tick during active work — the event count almost always
            # changes — which defeated the throttle and, with N parallel agents,
            # produced the `still running...` spam.) The poll/stall/kill logic
            # below still runs every heartbeat_s tick regardless of this gate.
            if (_hb_now - last_heartbeat_t) >= heartbeat_screen_s:
                print(f"  [auto:{provider}] still running... {elapsed}s " f"elapsed [{summary}] {growth_note}")
                last_heartbeat_t = _hb_now
            time.sleep(heartbeat_s)
    finally:
        if log_fh:
            try:
                log_fh.close()
            except Exception:
                pass


# Thresholds for escalating to super_heavy (opus) in
# _pick_agent_model_for_iter. Lowered 2026-06-03 from (5, 3) to (3, 2)
# after the Phi-3.5 attention case showed sonnet plateauing on the
# same STATE_DICT_KEY for two consecutive iters before opus engaged.
# Lower values get opus's deeper reasoning into play sooner, not after
# the iter budget is half-spent on sonnet retries that aren't moving.
_SUPER_HEAVY_ATTEMPTS_THRESHOLD = 3
_SUPER_HEAVY_CONSEC_SAME_CLASS_THRESHOLD = 2


def _pick_agent_model_for_iter(
    *,
    model_default: str,
    model_light: Optional[str],
    model_heavy: Optional[str],
    complexity_bonus: int,
    failure_class: str,
    attempts_so_far: int,
    force_heavy: bool = False,
    model_super_heavy: Optional[str] = None,
    consecutive_same_class: int = 0,
) -> Tuple[str, str]:
    """Pick the agent model alias for this iteration.

    Tiered switching is opt-in: pass `model_light` and/or `model_heavy`
    (typically via `--auto-model-light` / `--auto-model-heavy` /
    `--auto-model-tiered`). When neither is set, returns
    `(model_default, "default")` — the legacy single-model behavior.

    When tiered, escalates progressively:

    - **super_heavy** (opus, typically) fires when sonnet (heavy) has
      ALREADY been tried and is plateauing. Specifically:
        * ``attempts_so_far >= _SUPER_HEAVY_ATTEMPTS_THRESHOLD`` (3 by
          default: light iter-1 + one heavy iter-2 didn't graduate)
        * OR ``consecutive_same_class >= _SUPER_HEAVY_CONSEC_SAME_CLASS_THRESHOLD``
          (2 by default: heavy is repeating the same error class —
          don't burn a second heavy iter on it)
      Only fires when ``model_super_heavy`` is provided; otherwise the
      tier ladder caps at heavy as before.

      Defaults were lowered 2026-06-03 (attempts 5→3, consec 3→2) after
      the Phi-3.5 attention case showed sonnet hitting the same
      STATE_DICT_KEY twice without breakthrough — opus's deeper
      reasoning should engage sooner, not after burning 2 more sonnet
      iters.

    - **heavy** (sonnet, typically) fires when light has had a shot:
        * ``force_heavy`` (used by repair loops on no-edit iters)
        * ``complexity_bonus >= 2`` (palette > 30 ops, or LLM gaps) AND
          ``attempts_so_far >= 2`` (iter-1 keeps the haiku fast-pass even on
          complex components; complexity escalates from iter-2)
        * ``failure_class`` in ``_HEAVY_FAILURE_CLASSES``
        * ``attempts_so_far >= 2``

    - **light** is the default first-iter model. For claude this is
      haiku (fast-pass to produce a starting stub); for cursor it's
      sonnet-4.

    Returns `(chosen_model, reason)` where `reason` is a short tag for
    the log line — useful for post-mortems.
    """
    from ..cli import _HEAVY_FAILURE_CLASSES

    if not model_light and not model_heavy and not model_super_heavy:
        return (model_default, "default")
    light = model_light or model_default
    heavy = model_heavy or model_default
    super_heavy = model_super_heavy or None

    # Super-heavy tier (opus). Only fires if explicitly enabled AND the
    # heavy tier has demonstrably plateaued. Triggers:
    #   - attempts_so_far >= _SUPER_HEAVY_ATTEMPTS_THRESHOLD (3): heavy
    #     has had at least one shot (light iter-1 + heavy iter-2 both
    #     failed) so we escalate before sonnet burns more budget.
    #   - consecutive_same_class >= _SUPER_HEAVY_CONSEC_SAME_CLASS_THRESHOLD (2):
    #     heavy repeated the same error class once already — don't
    #     wait for a third sonnet attempt, give opus the slot.
    if super_heavy:
        if attempts_so_far >= _SUPER_HEAVY_ATTEMPTS_THRESHOLD:
            return (super_heavy, f"super_heavy:attempts={attempts_so_far}")
        if consecutive_same_class >= _SUPER_HEAVY_CONSEC_SAME_CLASS_THRESHOLD:
            return (super_heavy, f"super_heavy:consec_same_class={consecutive_same_class}")

    if force_heavy:
        return (heavy, "heavy:forced(no-edit-or-stuck-iter)")
    if complexity_bonus >= 2 and attempts_so_far >= 2:
        return (heavy, f"heavy:complexity={complexity_bonus}")
    if failure_class in _HEAVY_FAILURE_CLASSES:
        return (heavy, f"heavy:failure={failure_class}")
    if attempts_so_far >= 2:
        return (heavy, f"heavy:attempts={attempts_so_far}")
    return (light, "light")


def _resolve_tiered_model_aliases(
    *,
    provider: str,
    auto_model: Optional[str],
    auto_model_light: Optional[str],
    auto_model_heavy: Optional[str],
    auto_model_tiered: bool,
    auto_model_super_heavy: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve tiered model aliases into ``(light, heavy, super_heavy)``.

    Returns ``(None, None, None)`` when tiered mode is OFF and no
    explicit tier flags are set — the loop then uses the legacy
    single-model path with ``auto_model``.

    When ``--auto-model-tiered`` is set, applies provider defaults:
      * claude: ``haiku → sonnet → opus`` (3-tier auto-escalation;
        haiku for iter 1 as fast-pass, sonnet at iter 2 if iter 1
        failed, opus from iter 3 via lowered super_heavy thresholds).
      * cursor: ``sonnet-4 → opus`` (2-tier; cursor's ladder differs;
        super_heavy stays None)

    Explicit ``--auto-model-light`` / ``--auto-model-heavy`` /
    ``--auto-model-super-heavy`` always override the defaults.

    2026-06-03 history (chronological):
      * Initial 3-tier add: super_heavy = opus by default for claude,
        with attempts ≥ 5 / consec ≥ 3 triggers.
      * Threshold lowering: attempts 5→3, consec 3→2. Opus engages
        sooner so sonnet doesn't burn budget repeating the same class.
      * Haiku drop (REVERTED): briefly tried light=sonnet to skip
        haiku's flawed iter-1 outputs, but Phi-3.5 run #4 proved
        sonnet at iter 1 burned its 480s deliverable deadline
        investigating without writing — haiku's "produce SOMETHING
        fast" property is load-bearing. Ladder restored to
        haiku → sonnet → opus.

    Backward compat: callers receiving the previous 2-tuple shape will
    get a TypeError on unpack — the call site update is mechanical
    (add a 3rd variable name).
    """
    if not auto_model_light and not auto_model_heavy and not auto_model_super_heavy and not auto_model_tiered:
        return (None, None, None)
    if auto_model_tiered:
        if provider == "claude":
            # 2026-06-03 (revised): keep haiku as the iter-1 fast-pass.
            # An earlier revision dropped haiku entirely (light==heavy==
            # sonnet) but Phi-3.5 run #4 showed sonnet at iter 1 burned
            # its 480s deliverable deadline investigating without
            # writing _synth_responses/. Haiku's "produce SOMETHING
            # fast" property is load-bearing — even a flawed first stub
            # gives later iters a concrete starting point. Ladder:
            # haiku (iter 1) → sonnet (iter 2) → opus (iter 3+).
            default_light, default_heavy, default_super = "haiku", "sonnet", "opus"
        else:
            # cursor: 2-tier only for now; super stays unset
            default_light, default_heavy, default_super = "sonnet-4", "opus", None
    else:
        default_light, default_heavy, default_super = auto_model, auto_model, None
    return (
        auto_model_light or default_light,
        auto_model_heavy or default_heavy,
        auto_model_super_heavy or default_super,
    )
