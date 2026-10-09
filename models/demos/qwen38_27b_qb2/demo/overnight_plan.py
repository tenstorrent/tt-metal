# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only contracts for the GPQA-first eight-hour experiment queue."""

import json
from pathlib import Path


def load_followup(path):
    value = json.loads(Path(path).read_text())
    command = value.get("command")
    timeout = value.get("timeout_seconds")
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg for arg in command):
        raise ValueError("Follow-up command must be a nonempty argv list")
    if not Path(command[0]).is_absolute():
        raise ValueError("Follow-up interpreter must be absolute")
    if type(timeout) is not int or not 1 <= timeout <= 7200:
        raise ValueError("Follow-up timeout must be within 1..7200 seconds")
    return value


def perf_cases():
    """Every changed knob is isolated; long-context allocation stays bounded."""
    priority = [(length, batch) for length in (32768, 16384) for batch in (8, 16, 32)]
    return [
        dict(name="bfp8-budget32k", token_budget=32768, cells=priority + [(32768, 1), (16384, 1)], seconds=4500),
        dict(name="bfp8-long-context", token_budget=32768, cells=[(131072, 4), (131072, 8), (262016, 4)], seconds=3600),
        dict(name="bfp8-budget16k", token_budget=16384, cells=priority, seconds=3600),
        dict(name="bfp8-budget64k", token_budget=65536, cells=priority, seconds=3600),
    ]


def remaining_stage_seconds(deadline, now, requested, *, reserve=240, minimum=600):
    """Reserve time for process-group shutdown and do not start tiny leftovers."""
    available = int(deadline - now - reserve)
    return min(requested, available) if available >= minimum else 0
