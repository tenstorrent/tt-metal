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

import os
import subprocess
from pathlib import Path
from typing import Callable


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
        "print('NEXTRUNG=' + str(nt.get('rung') or ''))"
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
    }


def run_cc_loop(
    *,
    prompt: str,
    mcp_config_path: str | Path,
    allowed_tools: list[str],
    cwd: str | Path,
    env: dict,
    gate_fn: Callable[[], dict],
    max_rounds: int = 1000,
    claude_bin: str = "claude",
    on_round: Callable[[int, dict], None] | None = None,
) -> dict:
    """The domain-agnostic driver loop, identical in shape to the optimize engine's per-pipeline loop:
    each round ask ``gate_fn()`` (the deterministic stop authority); stop on ``halt`` or ``can_stop``;
    otherwise re-invoke ``claude -p`` against the MCP server. Returns ``{rounds, can_stop, halted}``.
    The agent is re-invoked with the same prompt every round (the gate carries state), matching the
    current optimize behavior."""
    rounds, can_stop, halted = 0, False, False
    while rounds < max_rounds:
        st = gate_fn()
        if st.get("halt"):
            halted = True
            break
        if st.get("can_stop"):
            can_stop = True
            break
        subprocess.run(
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
        )
        rounds += 1
        if on_round is not None:
            on_round(rounds, st)
    return {"rounds": rounds, "can_stop": can_stop, "halted": halted}
