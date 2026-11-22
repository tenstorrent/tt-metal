# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F189: the overseer's own files in a seeded run's ledger (F183) are as exempt as in the model's bringup dir."""

import fnmatch
import re

from models.demos.common.bringup.core import stamps
from models.demos.common.bringup.orchestrator import IGNORED

BASE = "models/demos/m/bringup"
LEDGER = f"{BASE}/ledgers/galaxy_8x4-full"


def _ignored(path):
    return any(re.search(p, path) for p in IGNORED)


def _skipped(path):
    return any(fnmatch.fnmatch(path, p) for p in stamps.SKIP)


def test_overseer_files_in_a_seeded_ledger_are_not_an_agents_change():
    for d in (BASE, LEDGER):
        for f in ("overseer.yaml", "supervision.md", "history_state.json"):
            assert _ignored(f"{d}/{f}"), f"{d}/{f}"
    assert not _ignored(f"{LEDGER}/spec.yaml")
    assert not _ignored("models/demos/m/tt/model.py")


def test_overseer_files_in_a_seeded_ledger_never_rerun_a_passed_unit():
    for d in (BASE, LEDGER):
        for f in ("overseer.yaml", "approvals.yaml", "supervision.md", "dashboard/index.html"):
            assert _skipped(f"{d}/{f}"), f"{d}/{f}"
    assert not _skipped("models/demos/m/tt/model.py")
