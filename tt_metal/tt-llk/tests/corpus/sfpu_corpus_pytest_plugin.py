"""Machine-readable pytest outcomes for the SFPU corpus runner.

The output path is supplied by ``SFPU_CORPUS_PYTEST_REPORT``.  Keeping this
reporter in tree avoids depending on optional pytest reporting plugins and,
most importantly, preserves pytest's exact node IDs for row attribution.

Variant attribution (batched classify producer sessions): when
``SFPU_CORPUS_VARIANT_MAP=1`` is set, the plugin snapshots the shared build
tree (``$RUNNER_TEMP/tt-llk-build``) around EVERY test protocol and records
the kernel artefact FILES (``*.elf`` and ``build.h``) each node created —
the node->ELF map that lets one producer session compile many classify legs
and still attribute every ELF to exactly one node.  The diff is taken at
FILE granularity on purpose: directory-level diffs mis-attribute when two
nodes share a parent dir (proven on gcd+rsubint32, whose variants are
siblings under one ``sources/<file>/`` dir — the first node swallowed the
second's ELFs and the second diffed empty).  This is only sound in a
sequential session (no xdist): concurrent workers share the tree and would
pollute each other's diffs, so the sweep never combines the two.  The keys
are additive; report consumers read via .get() (schema stays 1).
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

_collected = []
_collection_errors = []
_deselected = []
_reports = {}
_variant_files = {}
_variant_root = None


def pytest_sessionstart(session):
    del session
    global _variant_root
    _collected.clear()
    _collection_errors.clear()
    _deselected.clear()
    _reports.clear()
    _variant_files.clear()
    _variant_root = None
    if os.environ.get("SFPU_CORPUS_VARIANT_MAP") == "1":
        rt = os.environ.get("RUNNER_TEMP")
        if rt:
            _variant_root = pathlib.Path(rt) / "tt-llk-build"


def pytest_collection_finish(session):
    _collected.extend(item.nodeid for item in session.items)


def pytest_collectreport(report):
    if report.failed:
        _collection_errors.append(
            {"nodeid": report.nodeid, "longrepr": str(report.longrepr)}
        )


def pytest_deselected(items):
    # Runtime-only variant collapse (conftest) deselects same-compile-key
    # duplicates; a batched classify session must know a node was dropped
    # (vs silently absent) so the sweep can fall back to a solo compile.
    _deselected.extend(item.nodeid for item in items)


def _variant_snapshot():
    """The set of kernel artefact file relpaths (*.elf and build.h) in the
    shared build tree ('shared*' scaffolding excluded)."""
    out = set()
    if _variant_root is None or not _variant_root.is_dir():
        return out
    for path in _variant_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix != ".elf" and path.name != "build.h":
            continue
        rel = path.relative_to(_variant_root)
        if "shared" in rel.parts:
            continue
        out.add(str(rel))
    return out


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    del nextitem
    if _variant_root is None:
        yield
        return
    before = _variant_snapshot()
    yield
    created = sorted(_variant_snapshot() - before)
    if created:
        _variant_files[item.nodeid] = created


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
        "deselected": _deselected,
        "reports": _reports,
        "variant_files": _variant_files,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
