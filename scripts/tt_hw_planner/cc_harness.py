# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generic Claude-Code driver harness — the domain-agnostic core extracted from the optimize
cc engine (models/experimental/perf_automation/cc_optimize/run.py).

A driver that re-invokes ``claude -p`` against a domain MCP tool server, once per round, until the
server's OWN deterministic ``termination_check`` gate returns ``can_stop`` (or ``halt``), bounded by
``max_rounds``. Each domain (optimize / emit-e2e / bring-up) supplies its own MCP server, gate,
allowed-tools, and per-round prompt.

The gates are owned by the domain server and are NEVER touched here — the harness only READS the
gate's verdict and drives the subprocess. This module contains no gate logic.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from ._cli_helpers.agent import resolve_claude_bin


_HB_EVERY_S = 45
_AGENT_LINE = "[agent] "


def _verbose() -> bool:
    return os.environ.get("TT_HW_PLANNER_VERBOSE", "") not in ("", "0", "false", "False")


def _fmt_tool(name: str, inp: dict) -> str:
    """One-line summary of an agent tool call, matching commands/emit_e2e.py's _fmt_tool format so
    cc bring-up output reads identically to the emit-e2e builder's clean stream."""
    inp = inp or {}
    if name == "Bash":
        return "Bash: " + str(inp.get("command", ""))[:150]
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return f"{name} {inp.get('file_path') or inp.get('notebook_path') or inp.get('path') or ''}"
    if name in ("Grep", "Glob"):
        return f"{name} {inp.get('pattern', '')} {inp.get('path', '')}".rstrip()
    if name in ("Task", "Agent"):
        return f"{name}: {str(inp.get('description') or inp.get('prompt') or '')[:120]}"
    return f"{name} {json.dumps(inp)[:120]}"


def _render_cc_event(line: str):
    """Parse one `claude -p --output-format stream-json` line into a clean rendering, mirroring
    emit_e2e._render_stream_event. Returns (rendered_or_None, n_tool_use). System/user/result frames
    render to nothing on a clean screen (framework chatter stays off-screen, exactly like fsm)."""
    line = (line or "").strip()
    if not line or not line.startswith("{"):
        return ((_AGENT_LINE + line) if (_verbose() and line) else None, 0)
    try:
        ev = json.loads(line)
    except Exception:
        return (None, 0)
    t = ev.get("type")
    if t in ("system", "result"):
        return (None, 0)
    if t == "user":
        return (None, 0)
    if t == "assistant":
        msg = ev.get("message") or {}
        parts, n = [], 0
        for b in msg.get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text"):
                parts.append(_AGENT_LINE + b["text"].replace("\n", "\n" + _AGENT_LINE))
            elif b.get("type") == "tool_use":
                parts.append(_AGENT_LINE + "→ " + _fmt_tool(b.get("name", ""), b.get("input")))
                n += 1
        return ("\n".join(parts) if parts else None, n)
    return (None, 0)


def build_mcp_config(python_bin: str, server_path: str | Path, env: dict, server_name: str) -> dict:
    """A ``--mcp-config`` payload pointing claude at one stdio MCP server (the domain's tool server)."""
    return {
        "mcpServers": {
            server_name: {
                "command": str(python_bin),
                "args": [str(server_path)],
                "env": dict(env),
            }
        }
    }


def gate_status(
    python_bin: str,
    server_dir: str | Path,
    server_module: str,
    mcp_env: dict,
    cwd: str | Path,
    timeout_s: int = 3600,
) -> dict:
    """Call the server's ``termination_check()`` out-of-band (driver-side, NOT the agent) and return
    ``{can_stop, halt, reason, next_op, next_rung}``. The gate logic lives in ``server_module`` — this
    only reads its verdict, exactly as the optimize engine's ``_gate_status`` does. A crashed/timed-out
    gate is reported as not-done so the loop retries."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        f"import {server_module} as P; "
        "t=P.termination_check\n"
        "for a in ('fn','func','_fn','__wrapped__'):\n"
        "    if hasattr(t,a): t=getattr(t,a); break\n"
        "r=t()\n"
        "nt=r.get('next_target') or {}\n"
        "print('CANSTOP=' + str(bool(r.get('can_stop'))))\n"
        "print('HALT=' + str(bool(r.get('halt'))))\n"
        "print('HALTREASON=' + str(r.get('halt_reason') or ''))\n"
        "print('NEXTOP=' + str(nt.get('op') or nt.get('unit') or ''))\n"
        "print('NEXTRUNG=' + str(nt.get('rung') or ''))\n"
        "print('GRAD=' + ','.join(r.get('graduated') or []))\n"
        "print('SHARDGRAD=' + ','.join(r.get('shard_graduated') or []))"
    )
    env = dict(os.environ)
    env.update(mcp_env)
    try:
        r = subprocess.run(
            [str(python_bin), "-c", code, str(server_dir)],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:  # noqa: BLE001
        return {"can_stop": False, "halt": False, "reason": "", "next_op": "", "next_rung": ""}
    out = r.stdout or ""

    def pick(pfx: str) -> str:
        for line in out.splitlines():
            if line.startswith(pfx):
                return line[len(pfx) :]
        return ""

    return {
        "can_stop": "CANSTOP=True" in out,
        "halt": "HALT=True" in out,
        "reason": pick("HALTREASON="),
        "next_op": pick("NEXTOP="),
        "next_rung": pick("NEXTRUNG="),
        "graduated": [c for c in pick("GRAD=").split(",") if c],
        "shard_graduated": [c for c in pick("SHARDGRAD=").split(",") if c],
    }


def _resolve_agent_timeout_s(agent_timeout_s: int | None) -> int | None:
    """Per-round wall-clock budget for one `claude -p` invocation. None/<=0 disables the watchdog.
    Defaults generously (longer than any legitimate round, which can wrap a ~1800s device pytest) so it
    only fires on a truly wedged agent, and is env-tunable via TT_HW_PLANNER_CC_AGENT_TIMEOUT_S."""
    if agent_timeout_s is None:
        try:
            agent_timeout_s = int(os.environ.get("TT_HW_PLANNER_CC_AGENT_TIMEOUT_S", "3600"))
        except ValueError:
            agent_timeout_s = 3600
    return agent_timeout_s if agent_timeout_s and agent_timeout_s > 0 else None


def _kill_agent_tree(proc: subprocess.Popen) -> None:
    """Kill a wedged `claude -p` and everything it spawned (its MCP server + any pytest) via the
    process group, SIGTERM then SIGKILL — mirrors _run_focused_pytest's tree-kill."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def run_cc_loop(
    *,
    prompt: str,
    mcp_config_path: str | Path,
    allowed_tools: list[str],
    cwd: str | Path,
    env: dict,
    gate_fn: Callable[[], dict],
    max_rounds: int = 1000,
    claude_bin: str = resolve_claude_bin() or "claude",
    on_round: Callable[[int, dict], None] | None = None,
    pre_round: Callable[[int, dict], None] | None = None,
    on_heartbeat: Callable[[dict], None] | None = None,
    agent_timeout_s: int | None = None,
    max_consecutive_timeouts: int = 2,
) -> dict:
    """The domain-agnostic driver loop, identical in shape to the optimize engine's per-pipeline loop:
    each round ask ``gate_fn()`` (the deterministic stop authority); stop on ``halt`` or ``can_stop``;
    otherwise re-invoke ``claude -p`` against the MCP server. Returns ``{rounds, can_stop, halted}``.
    The agent is re-invoked with the same prompt every round (the gate carries state), matching the
    current optimize behavior.

    Terminal output matches the fsm loop's clean style rather than dumping raw stream-json: the agent's
    stdout is piped through `_render_cc_event`, so a clean screen shows only a throttled heartbeat
    (`· round R working… Ns, N tool calls`) while full tool/text lines appear under TT_HW_PLANNER_VERBOSE
    (framework chatter stays off-screen). Domain callers print fsm-style banners via ``pre_round`` (fired
    with the gate state before each agent invocation) and progress via ``on_round`` (after).

    Agent watchdog (parity with perf_automation's sdk_retry timeout->retry->abandon): each round is
    bounded by ``agent_timeout_s``; a `claude -p` that exceeds it has its whole tree killed and the round
    retried (the gate carries state, so no progress is lost); after ``max_consecutive_timeouts``
    consecutive wedges the loop abandons (halted=True). A round that ends on its own resets the counter.
    None/<=0 timeout disables the watchdog."""
    import threading

    rounds, can_stop, halted = 0, False, False
    timeout_s = _resolve_agent_timeout_s(agent_timeout_s)
    consecutive_timeouts = 0
    verbose = _verbose()
    while rounds < max_rounds:
        st = gate_fn()
        if st.get("halt"):
            halted = True
            break
        if st.get("can_stop"):
            can_stop = True
            break
        if pre_round is not None:
            pre_round(rounds + 1, st)
        proc = subprocess.Popen(
            [
                claude_bin,
                "-p",
                prompt,
                "--mcp-config",
                str(mcp_config_path),
                "--strict-mcp-config",
                "--allowedTools",
                *allowed_tools,
                "--output-format",
                "stream-json",
                "--verbose",
            ],
            cwd=str(cwd),
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _start = time.monotonic()
        _tool_calls = [0]
        _last_hb = [_start]

        def _pump(p=proc, tc=_tool_calls, hb=_last_hb, rnd=rounds + 1):
            try:
                for raw in p.stdout:
                    try:
                        rendered, n = _render_cc_event(raw)
                    except Exception:
                        rendered, n = (None, 0)
                    tc[0] += n
                    if verbose and rendered:
                        sys.stdout.write(rendered + "\n")
                        sys.stdout.flush()
                    now = time.monotonic()
                    if (now - hb[0]) >= _HB_EVERY_S:
                        hb[0] = now
                        sys.stdout.write(f"  · round {rnd} working… {int(now - _start)}s, {tc[0]} tool calls\n")
                        sys.stdout.flush()
                        # Live graduation feed: same cadence as the heartbeat. Domain caller
                        # inspects the gate state (cheap file read) and prints any new rows.
                        # Wrapped so a caller-side raise can never wedge the pump thread.
                        if on_heartbeat is not None:
                            try:
                                on_heartbeat(gate_fn())
                            except Exception:
                                pass
            except Exception:
                pass

        pump = threading.Thread(target=_pump, daemon=True)
        pump.start()
        try:
            proc.wait(timeout=timeout_s)
            pump.join(timeout=5)
            consecutive_timeouts = 0
        except subprocess.TimeoutExpired:
            _kill_agent_tree(proc)
            pump.join(timeout=5)
            consecutive_timeouts += 1
            print(
                f"  [cc-watchdog] agent round exceeded {timeout_s}s with no completion "
                f"(wedged agent); killed the agent tree. consecutive={consecutive_timeouts}/"
                f"{max_consecutive_timeouts}.",
                flush=True,
            )
            if consecutive_timeouts >= max_consecutive_timeouts:
                print("  [cc-watchdog] abandoning — agent wedged repeatedly.", flush=True)
                halted = True
                break
            continue
        rounds += 1
        if on_round is not None:
            on_round(rounds, st)
    return {"rounds": rounds, "can_stop": can_stop, "halted": halted}
