# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Collection gate for the Quasar pool harness in this directory.

These tests call ttnn.experimental.quasar.* and are sized for craq-sim / the Quasar emulator.
The WH/BH ttnn sanity pool group (`pytest tests/ttnn/unit_tests/operations/pool -xv`, 300 s per
test, tests/pipeline_reorg/ttnn_sanity_tests.yaml) collects this directory too, so everything
here is skipped unless the target is Quasar. QPOOL_RUN_ON_ANY_ARCH=1 overrides the gate: that is
the deliberate WH-silicon cross-check leg (the quasar pool op also runs on WH silicon).
"""

import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _target_is_quasar():
    try:
        from models.common.utility_functions import is_quasar

        return is_quasar()
    except Exception:  # no cluster / no device visible at collection time
        return False


def pytest_collection_modifyitems(config, items):
    if os.environ.get("QPOOL_RUN_ON_ANY_ARCH") or _target_is_quasar():
        return
    skip = pytest.mark.skip(
        reason="Quasar pool harness: target is not Quasar (set QPOOL_RUN_ON_ANY_ARCH=1 to run on WH/BH silicon)"
    )
    for item in items:
        if str(item.path).startswith(_HERE):
            item.add_marker(skip)
