# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strategist — picks which axis to optimize (device vs host/wall) from the baseline breakdown."""

from __future__ import annotations

import json
from typing import Callable

AXIS_TO_METRIC = {"device": "device_ms", "host": "wall_ms"}

PROMPT = (
    "You are the lead optimization agent choosing WHICH AXIS to optimize for this model.\n\n"
    "Runtime breakdown:\n{breakdown}\n\n"
    "Two axes:\n"
    "- 'device': optimize the compute kernels (target device_ms). Choose when device time is a "
    "meaningful share of wall time and the kernels have headroom.\n"
    "- 'host': optimize the host dispatch loop (trace capture / 2 command queues / bucketed "
    "decode — these levers exist). Choose when host_overhead DOMINATES wall time, because then "
    "even a perfect kernel optimization is invisible on the wall clock.\n"
    "Pick the axis with the real, attainable wall-clock win.\n"
    'Respond with ONE JSON object only: {{"axis": "device"|"host", "reasoning": <one sentence>}}.'
)


def build_axis_prompt(profile: dict) -> str:
    wall = float(profile.get("wall_ms", 0.0) or 0.0)
    dev = float(profile.get("device_ms", 0.0) or 0.0)
    host = max(0.0, wall - dev)
    host_pct = (host / wall * 100.0) if wall else 0.0
    buckets = sorted((profile.get("buckets") or []), key=lambda b: -float(b.get("device_ms", 0.0) or 0.0))
    blines = "\n".join(
        f"  {b.get('id')}: {float(b.get('device_ms', 0.0) or 0.0):.2f} ms"
        for b in buckets[:6]
        if b.get("id") != "host_overhead"
    )
    breakdown = (
        f"wall_ms (real end-to-end): {wall:.0f}\n"
        f"device_ms (compute): {dev:.2f}  ({100.0 - host_pct:.1f}% of wall)\n"
        f"host_overhead: {host:.0f}  ({host_pct:.1f}% of wall)\n"
        f"top device buckets:\n{blines or '  (none)'}"
    )
    return PROMPT.format(breakdown=breakdown)


def choose_axis(profile: dict, runner: Callable[[str], object]) -> str:
    """Return the metric to optimize ('device_ms' | 'wall_ms'); any error or unrecognized axis falls back to device_ms."""
    try:
        out = runner(build_axis_prompt(profile))
        if isinstance(out, dict):
            obj = out
        else:
            from .probes import _extract_json_object

            obj = json.loads(_extract_json_object(str(out)))
        return AXIS_TO_METRIC.get((obj or {}).get("axis"), "device_ms")
    except Exception:
        return "device_ms"


def make_cli_axis_runner() -> Callable[[str], str]:
    """CC-native strategist runner: drives the `claude` CLI (no SDK, no model tier, no tools) so cc
    discovery picks the axis via claude-code. Single-shot judgment — same contract as make_axis_runner."""
    import os
    import subprocess

    from .agent_bin import resolve_claude_bin

    _sys = "You pick the optimization axis. Final message is one JSON object, no prose."

    def runner(prompt: str) -> str:
        env = dict(os.environ)
        for _k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
            env.pop(_k, None)
        r = subprocess.run(
            [resolve_claude_bin(), "-p", prompt, "--output-format", "text", "--system-prompt", _sys],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
        if r.returncode != 0:
            raise RuntimeError(f"cc strategist (claude CLI) exit {r.returncode}: {(r.stderr or '')[-200:]}")
        return r.stdout or ""

    runner.last_usage = None
    runner.model = "claude-cli"
    return runner
