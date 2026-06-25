# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

import torch
from loguru import logger

from models.tt_dit.utils import walltime

if TYPE_CHECKING:
    import pytest

_WALLTIME_ON = os.environ.get("TT_WALLTIME", "1") != "0"
_walltime_items = 0
_walltime_wall = 0.0


def pytest_configure(config: pytest.Config) -> None:
    """Register the log-start plugin only when output capture is disabled."""
    if config.option.capture == "no":
        config.pluginmanager.register(_LogStartPlugin(), "tt_dit_logstart")


class _LogStartPlugin:
    @staticmethod
    def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:  # noqa: ARG004
        parts = nodeid.split("::")
        filename = parts[0].rsplit("/", 1)[-1]
        rest = "::".join(parts[1:])

        # split off params from last part
        params = []
        if "[" in rest:
            rest, param_str = rest.split("[", 1)
            params = param_str.rstrip("]").split("-")

        dim = "\033[2m"
        bold = "\033[1m"
        reset = "\033[0m"

        label = f"{dim}{filename}{reset}  {bold}{rest}{reset}"
        if params:
            label += f"  {dim}({', '.join(params)}){reset}"

        raw = f"{filename}  {rest}"
        if params:
            raw += f"  ({', '.join(params)})"

        width = shutil.get_terminal_size()[0]
        print(f"\n\n{'━' * width}\n{label}\n")  # noqa: T201


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Print each item's wall-time block at the end of its call phase, then reset so the
    next parametrized item starts clean. The session ledger keeps accumulating underneath."""
    if not _WALLTIME_ON or report.when != "call":
        return
    global _walltime_items, _walltime_wall
    _walltime_items += 1
    _walltime_wall += report.duration
    print(walltime.render(report.nodeid, wall=report.duration))  # noqa: T201
    walltime.reset()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:  # noqa: ARG001
    if _WALLTIME_ON and _walltime_items > 1:
        print(
            walltime.render_session(f"session aggregate ({_walltime_items} items)", wall=_walltime_wall)
        )  # noqa: T201


num_torch_threads = max(1, os.cpu_count())
logger.info(f"Setting torch num_threads to {num_torch_threads}")
torch.set_num_threads(num_torch_threads)
