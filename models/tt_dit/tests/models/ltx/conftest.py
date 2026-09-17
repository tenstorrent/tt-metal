# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""LTX test hooks (serving line).

Two things happen before any fixture runs:

* ``LTX_QUALITY=high|medium|fast`` (or legacy ``LTX_FAST=1``) expands to the served bundle at collection
  time -- the pipeline reads the sigma schedule at import. ``apply_quality_env`` is the shared definition
  (see ``utils/ltx.py``) so the pytest bundle cannot drift from the ltx_server worker's; with no
  ``LTX_QUALITY`` set it falls back to ``apply_fast_env``.
* The earliest pytest hook pins the process to one hardware thread per physical core and ``os.execv``
  re-execs BEFORE torch/ttnn thread pools and the tt-metal device threads spawn, so they inherit the
  pinned mask (taskset-equivalent placement, fully in-process; galaxy ring traced replay 6.2 s vs
  6.6-6.7 s without). ``LTX_PIN_PREIMPORT=0`` skips the re-exec, ``LTX_PIN_CORES=0`` disables all
  pinning; a no-op without SMT. It fires before the mesh_device fixture opens the device.
"""

from __future__ import annotations

import os

import pytest

from models.tt_dit.utils.ltx import apply_quality_env

apply_quality_env()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_cmdline_main(config):  # noqa: ARG001
    """Runs before collection and before any fixture/device open -- the earliest safe re-exec point."""
    if os.environ.get("LTX_PIN_PREIMPORT", "1") != "0":
        # Imported lazily and locally: keeps this hook free of torch/ttnn.
        from models.tt_dit.utils.host_affinity import reexec_pinned_before_torch

        reexec_pinned_before_torch("LTX tests (pytest_cmdline_main)")
    yield
