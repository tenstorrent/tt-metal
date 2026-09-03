# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared AttnRes test fixtures."""

import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3_d_p.tests.fabric_profiles import assert_requested_tp_wrap_was_realized


@pytest.fixture(scope="session")
def kimi_k3_checkpoint_dir() -> Path | None:
    """The pinned Kimi K3 checkpoint subset, or `None` for random weights.

    `None` rather than a skip: the gates are meant to hold on weights the model never saw,
    so the random arm is the one CI runs and the checkpoint is the opt-in. Point
    `KIMI_K3_CKPT` at a directory an index can be read from — `fetch_query_weights.py`
    writes one holding only the query weights.
    """
    value = os.getenv("KIMI_K3_CKPT")
    return Path(value) if value else None


@pytest.fixture(autouse=True)
def guard_the_requested_wrap(request):
    """Hold every mesh test to the fabric its arm asked for.

    A torus arm exists to put a ring under the TP-axis collective, and the control plane
    answers a request it cannot cable with a quiet downgrade rather than an error. Checking
    it once per test, after the device is open, is what keeps a downgraded box from
    reporting the wrapped arm green.

    Tests that never open a mesh are left alone, so this cannot pull a device into a
    host-only case.
    """
    if "mesh_device" not in request.fixturenames:
        yield
        return
    assert_requested_tp_wrap_was_realized(request.getfixturevalue("mesh_device"))
    yield
