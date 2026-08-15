"""Machine-readable pytest outcomes for the SFPU corpus runner.

The output path is supplied by ``SFPU_CORPUS_PYTEST_REPORT``.  Keeping this
reporter in tree avoids depending on optional pytest reporting plugins and,
most importantly, preserves pytest's exact node IDs for row attribution.
"""

from __future__ import annotations

import json
import os
import pathlib


_collected = []
_collection_errors = []
_reports = {}


def pytest_sessionstart(session):
    del session
    _collected.clear()
    _collection_errors.clear()
    _reports.clear()


def pytest_collection_finish(session):
    _collected.extend(item.nodeid for item in session.items)


def pytest_collectreport(report):
    if report.failed:
        _collection_errors.append(
            {"nodeid": report.nodeid, "longrepr": str(report.longrepr)}
        )


def pytest_runtest_logreport(report):
    phases = _reports.setdefault(report.nodeid, {})
    phases[report.when] = {
        "outcome": report.outcome,
        "duration": report.duration,
        "wasxfail": str(report.wasxfail) if hasattr(report, "wasxfail") else "",
    }


def pytest_sessionfinish(session, exitstatus):
    del session
    output = os.environ.get("SFPU_CORPUS_PYTEST_REPORT")
    if not output:
        return
    path = pathlib.Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "exitstatus": int(exitstatus),
        "collected": _collected,
        "collection_errors": _collection_errors,
        "reports": _reports,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
