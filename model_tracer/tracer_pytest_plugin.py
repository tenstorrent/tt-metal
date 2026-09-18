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

    # Read device_params BEFORE the yield, while the fixture is still active. After the yield this
    # fixture resumes during teardown, by which point the other function-scoped fixtures have been
    # finalised and getfixturevalue no longer resolves -- the sidecar then silently wrote nothing.
    device_params_snapshot = _read_device_params(request, node_id)

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

    _write_device_params_sidecar(device_params_snapshot, per_test_dir, node_id)


# Device configuration the test opened its mesh with. The trace records machine_info
# (board/card_count/mesh shape/firmware) but NOTHING about how the device was opened, so anything
# replaying a traced op has to guess -- and the guesses are not equivalent. conftest's mesh_device
# fixture applies fabric_config, worker_l1_size, trace_region_size, num_command_queues and
# dispatch_core_axis from device_params before open_mesh_device, and none of it is recoverable
# afterwards: every one of those is absent from the live MeshDevice object (verified -- only arch,
# shape and get_num_devices are queryable). Replaying llama3_70b_galaxy ops without its
# worker_l1_size=1345000, or with a fabric the model never set, runs the op under a device
# configuration the model never used.
#
# This sidecar only covers NEW traces. The existing corpus predates it and re-tracing every model is
# not feasible, so device_params_resolver.py recovers the same information for already-traced
# executions by reading the parametrization back out of the test named in their recorded `source`.
# Both paths produce the same key set (DEVICE_PARAM_KEYS there mirrors _DEVICE_PARAM_KEYS here); the
# sidecar is authoritative where it exists, since it records the value the test actually ran with.
#
# Captured from the device_params FIXTURE VALUE rather than by intercepting open_mesh_device: it is
# already materialised by the time the test runs, needs no device access, and stays correct if the
# open path changes. Only read when the test actually declares it, so tests without device_params
# are untouched and no fixture is instantiated as a side effect.
_DEVICE_PARAM_KEYS = (
    "fabric_config",
    "fabric_tensix_config",
    "fabric_manager",
    "fabric_router_config",
    "reliability_mode",
    "worker_l1_size",
    "l1_small_size",
    "trace_region_size",
    "num_command_queues",
    "dispatch_core_axis",
    "dispatch_core_type",
)


def _read_device_params(request, node_id):
    """The test's materialised device_params, or None. Must be called while the fixture is live."""
    if "device_params" not in getattr(request.node, "fixturenames", ()):
        return None
    try:
        params = request.getfixturevalue("device_params")
    except Exception as exc:  # fixture errored, was overridden oddly, or is unavailable
        logger.debug("Could not read device_params for {}: {}", node_id, exc)
        return None
    return params if isinstance(params, dict) else None


def _write_device_params_sidecar(params, per_test_dir, node_id):
    """Record the test's device_params next to its traces, for the tracer to attach.

    Best-effort and never fatal: a missing or unserialisable value is dropped rather than
    guessed at, because a wrong device parameter is worse than an absent one -- an absent one
    leaves the replayer on its documented default, a wrong one silently changes the workload.
    """
    if not params:
        return

    recorded = {}
    for key in _DEVICE_PARAM_KEYS:
        if key not in params:
            continue
        value = params[key]
        # ttnn enums (DispatchCoreAxis, FabricConfig, ...) are not JSON-serialisable; store their
        # str() form, which is what the master JSON already does for dtypes and layouts.
        try:
            json.dumps(value)
            recorded[key] = value
        except (TypeError, ValueError):
            recorded[key] = str(value)

    if not recorded:
        return
    try:
        path = Path(per_test_dir) / "_device_params.json"
        path.write_text(json.dumps({"test_nodeid": node_id, "device_params": recorded}, indent=2, sort_keys=True))
    except OSError as exc:
        logger.debug("Could not write device_params for {}: {}", node_id, exc)
