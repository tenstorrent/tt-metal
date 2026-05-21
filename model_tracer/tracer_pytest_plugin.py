# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin that owns all tracer-specific test infrastructure.

Loaded explicitly by ``model_tracer/generic_ops_tracer.py`` via
``-p model_tracer.tracer_pytest_plugin`` so that the top-level ``conftest.py``
does not need to carry any tracer-only logic.

What lives here:
- The ``--trace-params`` pytest CLI option and the ``pytest_configure`` hook
  that flips ``ttnn.operation_tracer._ENABLE_TRACE`` when the flag is set.
- The autouse ``_per_test_trace_dir`` fixture that gives every test its own
  trace subdirectory under ``TTNN_OPERATION_TRACE_DIR`` and writes a
  ``_status.json`` sidecar so the tracer can keep traces only from passed
  tests.

The plugin owns its own private ``StashKey`` + ``pytest_runtest_makereport``
hookimpl so it does not depend on (or conflict with) report stashing in the
top-level ``conftest.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from loguru import logger


def pytest_addoption(parser):
    parser.addoption(
        "--trace-params",
        action="store_true",
        default=False,
        help=(
            "Enable tracing of operation parameters (serializes all ttnn operation inputs to files). "
            "By default, only tensor metadata is saved. To include tensor values, call "
            "ttnn.operation_tracer.enable_tensor_value_serialization(True). "
            "See tech_reports/ttnn/operation-tracing.md for details."
        ),
    )


def pytest_configure(config):
    """Flip the ttnn operation tracer flag when ``--trace-params`` is set."""
    if config.getoption("--trace-params", default=False):
        import ttnn.operation_tracer

        ttnn.operation_tracer._ENABLE_TRACE = True


# Private stash key — separate from the top-level conftest's so this plugin
# is self-contained and can be loaded/unloaded without touching shared state.
_tracer_phase_report_key = pytest.StashKey()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    item.stash.setdefault(_tracer_phase_report_key, {})[rep.when] = rep


@pytest.fixture(scope="function", autouse=True)
def _per_test_trace_dir(request):
    """Per-test isolation for operation tracing.

    Only active when ``--trace-params`` is passed and ``TTNN_OPERATION_TRACE_DIR``
    is set. Each test gets its own subdirectory and a ``_status.json`` sidecar
    is written after the test runs so ``generic_ops_tracer.collect_operation_jsons``
    can include only traces from passed tests.
    """
    if not request.config.getoption("--trace-params", default=False):
        yield
        return

    base_trace_dir = os.environ.get("TTNN_OPERATION_TRACE_DIR")
    if not base_trace_dir:
        yield
        return

    node_id = request.node.nodeid
    dir_name = hashlib.sha256(node_id.encode()).hexdigest()[:16]
    per_test_dir = os.path.join(os.path.realpath(base_trace_dir), dir_name)
    os.makedirs(per_test_dir, exist_ok=True)

    os.environ["TTNN_OPERATION_TRACE_DIR"] = per_test_dir

    yield

    os.environ["TTNN_OPERATION_TRACE_DIR"] = base_trace_dir

    report = request.node.stash.get(_tracer_phase_report_key, {})
    call_report = report.get("call")
    setup_report = report.get("setup")

    if setup_report and setup_report.failed:
        status = "failed"
    elif setup_report and setup_report.skipped:
        status = "skipped"
    elif call_report is None:
        status = "skipped"
    elif call_report.passed:
        status = "passed"
    elif call_report.skipped:
        # In-test pytest.skip / xfail land here with passed=False, skipped=True;
        # without this branch they would be misclassified as failed.
        status = "skipped"
    else:
        status = "failed"

    try:
        status_path = Path(per_test_dir) / "_status.json"
        status_path.write_text(json.dumps({"test_nodeid": node_id, "status": status}, indent=2))
    except OSError as exc:
        logger.debug("Could not write trace status for {}: {}", node_id, exc)
