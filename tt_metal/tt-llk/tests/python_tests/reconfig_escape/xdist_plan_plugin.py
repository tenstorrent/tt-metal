#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin that injects the LLK_CFG_RESTORE env var for parallel xdist runs.
Bit of a hack to avoid threading this thing through all of testinfra.

Load with `-p xdist_plan_plugin`, make sure reconfig_escape/ is on PYTHONPATH.
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
