# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Regression test: the in-process agent's teardown must not mask a failed run.

``_DeviceWorker._shutdown`` leaves the process via ``os._exit`` to skip a C++
static-teardown crash.  ``os._exit`` hard-codes the status, and Python runs
atexit handlers while unwinding a failure too, so a fixed ``os._exit(0)`` would
report a broken evaluation as a successful one to any automation reading the
exit code.

These tests drive the real hooks from
``scripts/navsim_inproc/diffusiondrive_ttnn_inproc_agent.py`` in a subprocess
(navsim stubbed, no device needed) and assert the status survives.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_AGENT = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "navsim_inproc" / "diffusiondrive_ttnn_inproc_agent.py"
)

# Loads the real agent module with navsim stubbed, installs the exit-status hooks,
# then exits through the same os._exit(_EXIT_STATUS) call that _shutdown makes.
_CHILD = textwrap.dedent(
    """
    import atexit, importlib.util, os, sys, types

    for name, attrs in (
        ("navsim", {}),
        ("navsim.agents", {}),
        ("navsim.agents.abstract_agent", {"AbstractAgent": type("AbstractAgent", (), {})}),
        ("navsim.agents.diffusiondrive", {}),
        ("navsim.agents.diffusiondrive.transfuser_features", {"TransfuserFeatureBuilder": object}),
        ("navsim.common", {}),
        ("navsim.common.dataclasses", {"SensorConfig": object}),
    ):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location("dd_inproc_agent", "__AGENT_PATH__")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)

    agent._install_exit_status_hooks()
    atexit.register(lambda: os._exit(agent._EXIT_STATUS))

    mode = sys.argv[1]
    if mode == "clean":
        pass
    elif mode == "uncaught":
        raise RuntimeError("late failure after the device worker was created")
    elif mode == "sysexit":
        sys.exit(3)
    """
)


def _run(mode: str) -> int:
    return subprocess.run(
        [sys.executable, "-c", _CHILD.replace("__AGENT_PATH__", str(_AGENT)), mode],
        capture_output=True,
        text=True,
    ).returncode


def test_agent_module_present() -> None:
    assert _AGENT.exists(), f"in-process agent not found at {_AGENT}"


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("clean", 0),  # successful run still gets the fast, crash-free exit
        ("uncaught", 1),  # an uncaught exception must not be reported as success
        ("sysexit", 3),  # an explicit status (hydra's path) is preserved verbatim
    ],
)
def test_exit_status_survives_teardown(mode: str, expected: int) -> None:
    assert _run(mode) == expected
