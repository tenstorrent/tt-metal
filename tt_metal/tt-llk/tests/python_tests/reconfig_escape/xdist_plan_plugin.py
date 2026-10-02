#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: per-test-item LLK_CFG_RESTORE injection for xdist runs.

cfg_restore.maybe_restore_cfg_from_env reads its env var fresh on every call (no caching), so
no shared infra needs to change to make it per-item: this plugin just sets the env var, in this
worker process, immediately before each test item it knows about runs, then clears it after. Since
each xdist worker runs its assigned items strictly one at a time, this is race-free even though
os.environ is process-global.

--llk-plan-map=PATH points at a JSON {nodeid: restore_path}. A round's map only needs one entry
per victim: every victim gets ONE fresh restore plan for that round, and R separate pytest
invocations (one per trial round) is how repeated trials against the same victim happen, since
pytest only collects a given nodeid once per invocation.

Load with `-p xdist_plan_plugin`, with this file's directory (reconfig_escape/) on PYTHONPATH.
"""

import json
import os

_ENV_VAR = "LLK_CFG_RESTORE"


def pytest_addoption(parser):
    parser.addoption(
        "--llk-plan-map",
        action="store",
        default=None,
        help="JSON file: {nodeid: restore_path}",
    )


def pytest_configure(config):
    path = config.getoption("--llk-plan-map")
    config._llk_plan_map = {}
    if path:
        with open(path) as f:
            config._llk_plan_map = json.load(f)


def pytest_runtest_setup(item):
    restore_path = item.config._llk_plan_map.get(item.nodeid)
    if restore_path:
        os.environ[_ENV_VAR] = restore_path
    else:
        os.environ.pop(_ENV_VAR, None)


def pytest_runtest_teardown(item, nextitem):
    os.environ.pop(_ENV_VAR, None)
