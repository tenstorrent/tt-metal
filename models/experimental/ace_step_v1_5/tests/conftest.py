# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load ACE-Step fixtures when using ``--confcutdir=tests``.

Overrides the parent session ``device`` with a **function-scoped** 1×1 open so module PCC
tests stay isolated from ``MESH_DEVICE=BH_QB`` (CFG denoise opens its own mesh separately).
"""

from __future__ import annotations

import os

import pytest

pytest_plugins = ["models.experimental.ace_step_v1_5.conftest"]

from models.experimental.ace_step_v1_5.conftest import _open_kwargs, require_ttnn


@pytest.fixture(scope="function")
def device():
    """Open a single TTNN device per test (never a multi-device mesh)."""
    ttnn = require_ttnn()
    from models.experimental.ace_step_v1_5.utils.tt_device import close_ace_step_device

    saved_mesh = os.environ.pop("MESH_DEVICE", None)
    saved_ace_mesh = os.environ.pop("ACE_STEP_MESH_DEVICE", None)
    dev = None
    try:
        try:
            dev = ttnn.open_device(**_open_kwargs())
        except RuntimeError as exc:
            pytest.skip(
                "Could not open TT device 0 for module PCC tests "
                f"({exc}). Unset MESH_DEVICE / ACE_STEP_MESH_DEVICE, close other "
                "pytest sessions holding the card, then reset: tt-smi -r 0"
            )
        if hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()
        yield dev
    finally:
        if dev is not None:
            close_ace_step_device(ttnn, dev)
        if saved_mesh is not None:
            os.environ["MESH_DEVICE"] = saved_mesh
        if saved_ace_mesh is not None:
            os.environ["ACE_STEP_MESH_DEVICE"] = saved_ace_mesh


@pytest.fixture(scope="function")
def mesh_device(device):
    """Reuse the per-test 1×1 device instead of a session BH_QB mesh.

    Avoids exhausting MeshDevice slots (remote-only teardown abort) when many
    tests each open their own device while a session 2×2 mesh stays alive.
    """
    return device
