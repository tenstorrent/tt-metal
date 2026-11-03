"""pytest plugin loaded by run_group.py (PYTEST_ADDOPTS="-p regress_junit_plugin").

Gives every pytest invocation of a step its own junit file under $REGRESS_JUNIT_DIR, so a CI command that runs pytest
several times keeps every report (pytest.ini's --junitxml would overwrite one file; CI renames it only when CI=true,
which also changes test behaviour).
"""

import os
import time

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    d = os.environ.get("REGRESS_JUNIT_DIR")
    if not d or hasattr(config, "workerinput"):  # xdist workers: the controller writes the report
        return
    os.makedirs(d, exist_ok=True)
    config.option.xmlpath = os.path.join(d, f"pytest_{os.getpid()}_{time.time_ns()}.xml")
