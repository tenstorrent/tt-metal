# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""LTX test hooks.

The earliest pytest hook available to a conftest: pin the process to one hardware thread per physical
core and ``os.execv`` re-exec BEFORE torch/ttnn thread pools and the tt-metal device threads spawn, so
they inherit the pinned mask (taskset-equivalent placement, fully in-process). Gated behind an explicit
``LTX_PIN_PREIMPORT=1`` and default OFF so CI unit/e2e runs never re-exec and behave exactly as before.

The re-exec fires before the mesh_device fixture opens the device -- re-execing after a device is open
risks wedging it.
"""

from __future__ import annotations

import os

import pytest


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_cmdline_main(config):  # noqa: ARG001
    """Runs before collection and before any fixture/device open -- the earliest safe re-exec point."""
    if os.environ.get("LTX_PIN_PREIMPORT") == "1":
        # Imported lazily and locally: keeps this hook free of torch/ttnn.
        from models.tt_dit.utils.host_affinity import reexec_pinned_before_torch

        reexec_pinned_before_torch("LTX tests (pytest_cmdline_main)")
    yield
