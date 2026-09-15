# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil

import pytest
import torch
from loguru import logger


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "diffvae_gate: LTX-2.5 DiffVAE regression gate, selected by run_diffvae_gates.sh; "
        "under DIFFVAE_GATES_STRICT=1 a runtime skip is treated as a failure",
    )
    config._diffvae_rollups = []
    if config.option.capture == "no":
        config.pluginmanager.register(_LogStartPlugin(), "tt_dit_logstart")


class _LogStartPlugin:
    """One banner line per test when output capture is off, so device logs are attributable."""

    @staticmethod
    def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:  # noqa: ARG004
        parts = nodeid.split("::")
        filename = parts[0].rsplit("/", 1)[-1]
        rest = "::".join(parts[1:])

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

        width = shutil.get_terminal_size()[0]
        print(f"\n\n{'━' * width}\n{label}\n")  # noqa: T201


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Under DIFFVAE_GATES_STRICT=1 a runtime skip on a gate (missing capture or checkpoint) is a
    failure, so a missing dependency goes red instead of quietly passing. Design skips declared
    with @pytest.mark.skip / skipif are intentional and left alone."""
    outcome = yield
    report = outcome.get_result()
    if (
        os.environ.get("DIFFVAE_GATES_STRICT") == "1"
        and item.get_closest_marker("diffvae_gate") is not None
        and report.skipped
        and item.get_closest_marker("skip") is None
        and item.get_closest_marker("skipif") is None
    ):
        report.outcome = "failed"
        report.longrepr = f"GATE SKIPPED under DIFFVAE_GATES_STRICT → treated as FAIL: {report.longrepr}"


@pytest.fixture
def timing_tree(request):
    """Render the decode timing tree of the test's last decode pass, the timed one.

    A decode test warms up first and measures second, so the last root recorded during the test is
    the timed pass. TT_DIT_TREE_ALL=1 also renders the warm-up passes.
    """
    from models.tt_dit.utils import timing_tree as tree

    first = tree.root_count()
    yield
    new = tree.roots()[first:]
    if not new:
        return

    nodeid = request.node.name
    print("\n" + tree.render(new[-1], title=f"{nodeid} · decode pass {len(new)} of {len(new)} (TIMED)"))
    if os.environ.get("TT_DIT_TREE_ALL") == "1":
        for i, root in enumerate(new[:-1]):
            print("\n" + tree.render(root, title=f"{nodeid} · decode pass {i + 1} of {len(new)} (warm-up)"))
    request.config._diffvae_rollups.append((nodeid, tree.category_totals(new[-1]), new[-1].incl_ms))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Repeat each timed decode's category roll-up after the summary line, where it can be read
    side by side across configurations."""
    from models.tt_dit.utils import timing_tree as tree

    for nodeid, (totals, spans), total_ms in getattr(config, "_diffvae_rollups", []):
        terminalreporter.write_line("")
        terminalreporter.write_line(f"DECODE CATEGORIES · {nodeid}")
        terminalreporter.write_line(tree.render_categories(totals, spans, total_ms))


num_torch_threads = max(1, os.cpu_count())
logger.info(f"Setting torch num_threads to {num_torch_threads}")
torch.set_num_threads(num_torch_threads)
