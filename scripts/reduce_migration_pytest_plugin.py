# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Record actual pytest collection/execution counts for the migration runner."""

from collections import Counter
import json
import os
from pathlib import Path

import pytest

_errors = []
_outcomes = Counter()


def pytest_configure(config):
    if getattr(config.option, "numprocesses", None) not in (None, 0):
        raise pytest.UsageError("Migration-suite counting requires serial pytest; remove -n/xdist from PYTEST_ADDOPTS")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    manifest = json.loads(Path(os.environ["TT_REDUCE_SUITE_MANIFEST"]).read_text())
    group = next(group for group in manifest["groups"] if group["id"] == os.environ["TT_REDUCE_SUITE_GROUP"])
    exclusions = group.get("exclude_parameters", {})
    expected_nodeids = None
    if group.get("nodeid_arch_template"):
        arch = config.getoption("--tt-arch")
        expected_nodeids = {f"{group['source']}::{test.replace('{arch}', arch)}" for test in group["tests"]}
    retained, deselected = [], []
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        if (expected_nodeids is not None and item.nodeid not in expected_nodeids) or any(
            params.get(name) in values for name, values in exclusions.items()
        ):
            deselected.append(item)
        else:
            retained.append(item)
    if deselected:
        items[:] = retained
        config.hook.pytest_deselected(items=deselected)


def pytest_collectreport(report):
    if report.failed:
        _errors.append({"nodeid": report.nodeid, "error": str(report.longrepr)})


def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        _outcomes[report.outcome] += 1


def pytest_sessionfinish(session, exitstatus):
    destination = Path(os.environ["TT_REDUCE_SUITE_RESULT"])
    nodeids = [item.nodeid for item in session.items]
    # Collection records all cases, including markers/fixture skips that are resolved only at execution.
    result = {
        "tt_arch": session.config.getoption("--tt-arch", default=None),
        "collected": len(nodeids),
        "unique_nodeids": len(set(nodeids)),
        "unconditional_skip_marks": sum(item.get_closest_marker("skip") is not None for item in session.items),
        "collection_errors": _errors,
        "pytest_exitstatus": int(exitstatus),
        "outcomes": dict(_outcomes),
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    destination.with_suffix(".nodeids.txt").write_text("\n".join(nodeids) + ("\n" if nodeids else ""))
