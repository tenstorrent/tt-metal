"""Narrow this process before exec, so the entire shell payload inherits one CPU."""
import json
import os
import sys
from pathlib import Path


def select_cpu(allowed):
    if not allowed:
        raise RuntimeError("No allowed CPU")
    return {min(allowed)}


def exec_on_one_cpu(command, receipt):
    if not command:
        raise ValueError("Missing shell payload")
    before = set(os.sched_getaffinity(0))
    selected = select_cpu(before)
    os.sched_setaffinity(0, selected)
    inherited = set(os.sched_getaffinity(0))
    if inherited != selected:
        raise RuntimeError("Failed to select exactly one allowed CPU")
    receipt.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "affinity_before": sorted(before),
                "inherited_affinity": sorted(inherited),
                "command": command,
            },
            indent=2,
        )
        + "\n"
    )
    os.execvp(command[0], command)


if __name__ == "__main__":
    exec_on_one_cpu(sys.argv[2:], Path(sys.argv[1]))
