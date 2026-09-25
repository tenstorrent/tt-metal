# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil

import pytest
import torch
from loguru import logger


def pytest_configure(config: pytest.Config) -> None:
    """Register the log-start plugin only when output capture is disabled."""
    if config.option.capture == "no":
        config.pluginmanager.register(_LogStartPlugin(), "tt_dit_logstart")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # noqa: ARG001
    """Print the traceback at failure time, before fixture teardown.

    On a multi-rank mesh, one rank's failure strands the others in collectives and
    `close_mesh_device`'s own cross-rank barrier then never completes, so the
    post-teardown FAILURES section is unreachable exactly when it is needed.
    """
    outcome = yield
    report = outcome.get_result()
    if report.failed and report.when in ("setup", "call"):
        logger.error(f"{item.nodeid} failed in {report.when} (traceback before teardown):\n{report.longreprtext}")


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


num_torch_threads = max(1, os.cpu_count())
logger.info(f"Setting torch num_threads to {num_torch_threads}")
torch.set_num_threads(num_torch_threads)
