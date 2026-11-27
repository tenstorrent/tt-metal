# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the groupnorm_sc_N_1_HW_C_b pre-migration suite.

The op directory's conftest marks every item ``use_module_device`` (one device per test module). The module-scoped
device reads its parameters from the closest ``use_module_device`` marker of the *module* node, so this conftest marks
each module of this directory with a trace region: ``test_groupnorm_sc_N_1_HW_C_b_output_tensor.py`` replays a
captured call, which needs one. Nothing else about the device changes.

The op plans against the L1 free at call time, and the device lives for the whole module, so every test must leave
L1 as it found it. A failed test does not: pytest keeps its frames (``sys.last_traceback``) and with them its sharded
tensors, and the next tests then see a full L1 and are refused. ``_free_test_tensors`` drops those frames after each
test so one failure stays one failure.
"""

import gc
import sys
from pathlib import Path

import pytest

# Bytes of DRAM reserved for trace capture on the module device. One call of the op on <= 110 cores records well
# under this.
TRACE_REGION_SIZE = 4 * 1024 * 1024

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    marked = set()
    for item in items:
        path = Path(str(item.path)).resolve()
        if _HERE not in path.parents:
            continue
        module = item.getparent(pytest.Module)
        if module is None or id(module) in marked:
            continue
        marked.add(id(module))
        module.add_marker(pytest.mark.use_module_device({"trace_region_size": TRACE_REGION_SIZE}))


@pytest.fixture(autouse=True)
def _free_test_tensors():
    """After each test, release what a failure kept alive (its frames hold device tensors), then collect cycles."""
    yield
    for name in ("last_type", "last_value", "last_traceback", "last_exc"):
        if hasattr(sys, name):
            delattr(sys, name)
    gc.collect()
