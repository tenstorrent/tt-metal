#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Pytest plugin that auto-installs the ttnn step-debugger when TTNVTOP_STEP=1.
# Drop a symlink (or copy) of this file as `conftest.py` in the directory of
# the test you want to step through, OR add `pytest_plugins = ["...conftest_step"]`
# at the top of an existing conftest.
#
# Easiest: from the tt-metal root,
#    cp tt_metal/tools/ttnvtop/scripts/conftest_step.py models/tt_transformers/demo/conftest.py
#
# Then run:
#    TTNVTOP_REGISTER_PROGRAMS=1 TTNVTOP_STEP=1 \
#        pytest -xvs models/tt_transformers/demo/simple_text_demo.py::test_demo_text
#
# The plugin attaches a Stepper to whichever ttnn ops execute inside the test.
# All test fixtures still run normally; the stepper only intercepts ttnn ops.

import os
import sys
import pytest


def pytest_collection_modifyitems(config, items):
    """No-op hook used to confirm the plugin loaded."""
    if os.getenv("TTNVTOP_STEP") == "1":
        sys.stderr.write(
            f"[conftest_step] active — TTNVTOP_STEP=1, will install Stepper " f"around {len(items)} test(s)\n"
        )


@pytest.fixture(autouse=True)
def _ttnvtop_stepper(request):
    """Auto-installs the Stepper around every test when TTNVTOP_STEP=1.

    The stepper wraps a configurable list of ttnn ops; default list lives in
    Stepper.DEFAULT_WATCH. Override with the TTNVTOP_STEP_OPS env var
    (comma-separated):
        TTNVTOP_STEP_OPS=matmul,softmax,linear pytest ...
    """
    if os.getenv("TTNVTOP_STEP") != "1":
        yield
        return

    # Lazy-import so the plugin is harmless when the stepper isn't on PYTHONPATH.
    try:
        # The script directory is a package only if there's an __init__.py;
        # to be safe, append the dir to sys.path and import by basename.
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from ttnn_step import Stepper  # noqa: E402
    except Exception as e:
        sys.stderr.write(f"[conftest_step] cannot import Stepper: {e}\n")
        yield
        return

    # The stepper needs a device handle. Most ttnn-based tests take a `device`
    # fixture argument; some take `mesh_device`. We pull whichever exists.
    device = None
    for fix_name in ("device", "mesh_device", "t3k_mesh_device"):
        try:
            device = request.getfixturevalue(fix_name)
            break
        except Exception:
            continue
    if device is None:
        sys.stderr.write(
            "[conftest_step] no device-like fixture found "
            "(tried: device, mesh_device, t3k_mesh_device) — stepper not installed\n"
        )
        yield
        return

    watch_env = os.getenv("TTNVTOP_STEP_OPS")
    watch = None
    if watch_env:
        watch = [s.strip() for s in watch_env.split(",") if s.strip()]

    step = Stepper(device, watch=watch)
    step.install()
    sys.stderr.write(f"[conftest_step] Stepper installed for {request.node.name}\n")
    try:
        yield
    finally:
        step.uninstall()
        sys.stderr.write(f"[conftest_step] Stepper uninstalled\n")
