# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the device once per module for this op's unit tests.

Applies @pytest.mark.use_module_device to every collected test; the root
`device` fixture is function-scoped and the marker switches it to module scope.
Do not define a local `device` fixture here — it shadows the root one and
disables the marker.
"""
import os

import pytest

# Perf-tournament harnesses (test_tilize_perf1_*.py) drive kernel-dir variants that their
# generator scripts under ttnn/ttnn/operations/tilize/perf_experiments/<idea>/ produce (the
# generated dirs are not committed). They are opt-in: TILIZE_PERF_EXPERIMENTS=1.
_PERF_EXPERIMENT_PREFIX = "test_tilize_perf1_"


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="perf-experiment harness: set TILIZE_PERF_EXPERIMENTS=1")
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
        if (
            item.fspath.basename.startswith(_PERF_EXPERIMENT_PREFIX)
            and os.environ.get("TILIZE_PERF_EXPERIMENTS") != "1"
        ):
            item.add_marker(skip)
