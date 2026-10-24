"""Pytest plugin: record which model-source files EXECUTE during the perf test.

The optimizer must only edit code the profiled workload actually runs. A static
file list can't tell that (a multi-modal pipeline ships speech/text/vocoder
stubs but any one task runs only a slice). This plugin installs a sys.settrace
hook for the duration of the test call and records every source file under
TT_EXEC_TRACE_ROOT whose Python code executes, dumping the set to
TT_EXEC_TRACE_OUT (json). exec_scope reads it and scopes the edit targets.

Invoked via `pytest <perf_test> -p agent.exec_trace_plugin`.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_executed: set = set()
_root = ""


def _tracer(frame, event, arg):
    if event == "call":
        fn = frame.f_code.co_filename
        if _root and _root in fn:
            _executed.add(fn)
    return _tracer


def pytest_configure(config):
    global _root
    _root = os.environ.get("TT_EXEC_TRACE_ROOT", "")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # Trace only the test body (the model build + forward), not collection/fixtures.
    if _root:
        sys.settrace(_tracer)
    try:
        yield
    finally:
        sys.settrace(None)


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("TT_EXEC_TRACE_OUT")
    if out:
        try:
            json.dump(sorted(_executed), open(out, "w"))
        except Exception:
            pass
