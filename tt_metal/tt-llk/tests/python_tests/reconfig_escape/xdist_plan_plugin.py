#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: per-test-item LLK_CFG_RESTORE injection for xdist runs.

cfg_restore.maybe_restore_cfg_from_env reads its env vars fresh on every call (no caching), so
no shared infra needs to change to make it per-item: this plugin just sets the env var, in this
worker process, immediately before each test item it knows about runs, then clears it after. Since
each xdist worker runs its assigned items strictly one at a time, this is race-free even though
os.environ is process-global.

--llk-plan-map=PATH points at a JSON {nodeid: restore_plan_path}. A round's map only needs one
entry per victim (see xdist_sequential_fuzz.py): every victim gets ONE fresh poison plan for that
round, and R separate pytest invocations (one per trial round) is how repeated trials against the
same victim happen, since pytest only collects a given nodeid once per invocation.

A map entry may be a plain restore_plan_path (legacy, still used by xdist_sequential_fuzz.py) or a
{"restore": path, "addrmod_restore": path} dict (discover_catalog.py/pair_sweep.py) to also carry
the per-thread addr-mod restore plan alongside the main one.

Load with `-p xdist_plan_plugin`, with this file's directory (reconfig_escape/) on PYTHONPATH.
"""

import json
import os

_ENV_VAR = "LLK_CFG_RESTORE"
_ADDRMOD_ENV_VAR = "LLK_CFG_ADDRMOD_RESTORE"


def pytest_addoption(parser):
    parser.addoption(
        "--llk-plan-map",
        action="store",
        default=None,
        help="JSON file: {nodeid: restore_plan_path | {restore, addrmod_restore}}",
    )


def pytest_configure(config):
    path = config.getoption("--llk-plan-map")
    config._llk_plan_map = {}
    if path:
        with open(path) as f:
            config._llk_plan_map = json.load(f)


def pytest_runtest_setup(item):
    entry = item.config._llk_plan_map.get(item.nodeid)
    restore_path, addrmod_path = (
        (entry.get("restore"), entry.get("addrmod_restore"))
        if isinstance(entry, dict)
        else (entry, None)
    )
    if restore_path:
        os.environ[_ENV_VAR] = restore_path
    else:
        os.environ.pop(_ENV_VAR, None)
    if addrmod_path:
        os.environ[_ADDRMOD_ENV_VAR] = addrmod_path
    else:
        os.environ.pop(_ADDRMOD_ENV_VAR, None)


def pytest_runtest_teardown(item, nextitem):
    os.environ.pop(_ENV_VAR, None)
    os.environ.pop(_ADDRMOD_ENV_VAR, None)
