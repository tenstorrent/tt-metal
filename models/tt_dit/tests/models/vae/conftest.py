# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Gate enforcement for the LTX-2.5 DiffVAE regression suite.

Three of the ``diffvae_gate`` tests skip when their deps are missing (``ltx_core`` via a
module-level ``importorskip``; the capture dump / gated checkpoint via ``pytest.skip``), and a
skipped test reports green. Under ``DIFFVAE_GATES_STRICT=1`` a skip on a ``diffvae_gate`` test is
turned into a failure, so a missing checkpoint goes red instead of quietly passing.

The ``ltx_core`` importorskip skips at *collection*, before this hook can see it — that one is the
runner's preflight (``run_diffvae_gates.sh``). This hook catches the in-test / fixture skips
(missing capture or checkpoint).
"""

from __future__ import annotations

import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        os.environ.get("DIFFVAE_GATES_STRICT") == "1"
        and item.get_closest_marker("diffvae_gate") is not None
        and report.skipped
        # Only flip *runtime* skips (a missing capture/checkpoint via pytest.skip()). Design skips
        # declared with @pytest.mark.skip / skipif — e.g. the fp32 stage-5 case NA3D can't run — carry
        # a skip marker and are left alone; they are intentional, not a missing-dependency false green.
        and item.get_closest_marker("skip") is None
        and item.get_closest_marker("skipif") is None
    ):
        report.outcome = "failed"
        report.longrepr = f"GATE SKIPPED under DIFFVAE_GATES_STRICT → treated as FAIL: {report.longrepr}"


@pytest.fixture
def decode_tree(request):
    """Render the decode timing tree for the TIMED pass of a decode test.

    A decode test warms up first and measures second, so the last root recorded during the test IS
    the timed pass -- no marker to set and nothing that can drift out of step with what actually
    ran. The header says which pass it was, derived from the count rather than asserted, so a test
    that grows a third decode reports "3 of 3" instead of quietly mislabelling.

    Set DIFFVAE_TREE_ALL=1 to also render the warm-up passes (JIT compile cost lands there).
    """
    from models.tt_dit.utils import decode_tree as tree

    first = tree.root_count()
    yield
    new = tree.roots()[first:]
    if not new:
        return  # timing off, or the test skipped before decoding

    nodeid = request.node.name
    print("\n" + tree.render(new[-1], title=f"{nodeid} · decode pass {len(new)} of {len(new)} (TIMED)"))
    if os.environ.get("DIFFVAE_TREE_ALL") == "1":
        for i, root in enumerate(new[:-1]):
            print("\n" + tree.render(root, title=f"{nodeid} · decode pass {i + 1} of {len(new)} (warm-up)"))
    request.config._diffvae_rollups.append((nodeid, tree.category_totals(new[-1]), new[-1].incl_ms))


def pytest_configure(config):
    config._diffvae_rollups = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Repeat each timed decode's category roll-up after the summary line.

    The trees themselves are thousands of loguru lines up the scrollback by the time a run ends;
    this is the part worth reading side by side when comparing two configurations.
    """
    from models.tt_dit.utils import decode_tree as tree

    for nodeid, (totals, spans), total_ms in getattr(config, "_diffvae_rollups", []):
        terminalreporter.write_line("")
        terminalreporter.write_line(f"DECODE CATEGORIES · {nodeid}")
        terminalreporter.write_line(tree.render_categories(totals, spans, total_ms))
