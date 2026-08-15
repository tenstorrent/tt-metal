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
