#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin that puts the core back to a clean baseline, runs a test, and records config residue.

--llk-pristine-restore=PATH restore plan applied before every candidate item launches.
--llk-capture-outdir=DIR    each test's residue written to <outdir>/<sanitized-nodeid>.snapshot.json
--llk-candidates=PATH       file of nodeids this plugin should act on

Load with `-p xdist_capture_plugin`, with reconfig_escape/ on PYTHONPATH.
"""

import json
import os

from discover_catalog import _sanitize

_RESTORE_VAR = "LLK_CFG_RESTORE"


def pytest_addoption(parser):
    parser.addoption("--llk-pristine-restore", action="store", default=None)
    parser.addoption("--llk-capture-outdir", action="store", default=None)
    parser.addoption("--llk-candidates", action="store", default=None)


def pytest_configure(config):
    config._llk_restore_path = config.getoption("--llk-pristine-restore")
    config._llk_outdir = config.getoption("--llk-capture-outdir")
    cand_path = config.getoption("--llk-candidates")
    config._llk_candidates = set()
    if cand_path:
        with open(cand_path) as f:
            config._llk_candidates = {line.strip() for line in f if line.strip()}


def pytest_runtest_setup(item):
    if item.nodeid in item.config._llk_candidates and item.config._llk_restore_path:
        os.environ[_RESTORE_VAR] = item.config._llk_restore_path
    else:
        os.environ.pop(_RESTORE_VAR, None)


def pytest_runtest_teardown(item, nextitem):
    os.environ.pop(_RESTORE_VAR, None)
    if item.nodeid not in item.config._llk_candidates or not item.config._llk_outdir:
        return
    from helpers.cfg_restore import (
        snapshot_adc_ch1x,
        snapshot_addr_mod,
        snapshot_cfg,
        thread_items,
    )
    from helpers.chip_architecture import get_chip_architecture
    from helpers.test_config import TestConfig

    arch = get_chip_architecture()
    items = thread_items(arch)
    snap = snapshot_cfg(TestConfig.TENSIX_LOCATION, items)
    out_path = os.path.join(
        item.config._llk_outdir, _sanitize(item.nodeid) + ".snapshot.json"
    )
    with open(out_path, "w") as f:
        json.dump([[s, a, v] for (s, a), v in snap.items()], f)

    addrmod_snap = snapshot_addr_mod(TestConfig.TENSIX_LOCATION)
    addrmod_path = os.path.join(
        item.config._llk_outdir, _sanitize(item.nodeid) + ".addrmod.json"
    )
    with open(addrmod_path, "w") as f:
        json.dump([[t, a, v] for (t, a), v in addrmod_snap.items()], f)

    ch1x_snap = snapshot_adc_ch1x(TestConfig.TENSIX_LOCATION)
    ch1x_path = os.path.join(
        item.config._llk_outdir, _sanitize(item.nodeid) + ".adc_ch1x.json"
    )
    with open(ch1x_path, "w") as f:
        json.dump(ch1x_snap, f)
