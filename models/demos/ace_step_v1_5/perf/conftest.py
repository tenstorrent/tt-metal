# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Perf tests under ``perf/`` must use ``ttnn.open_device`` (demo path).

The repository root ``conftest.py`` exposes a ``device`` fixture built from ``ttnn.CreateDevice``,
which may yield a ``MeshDevice`` handle. ACE-Step VAE conv helpers are validated on the same
``open_device`` entry point as ``run_prompt_to_wav.py``; this nearest ``conftest.py`` overrides
``device`` so Tracy/pytest runs match that path.

L1 / HiFi2 throughput toggles are on by default in ``ttnn_impl/math_perf_env.py`` (same as demo and
E2E). Set ``ACE_STEP_*_PERF=0`` only when debugging DRAM parity.
"""

from __future__ import annotations

import os

import pytest

_DEFAULT_L1_SMALL = int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304"))


@pytest.fixture(scope="session")
def device():
    ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
    dev = ttnn.open_device(
        device_id=int(os.environ.get("TT_DEVICE_ID", "0")),
        l1_small_size=_DEFAULT_L1_SMALL,
        trace_region_size=128 << 20,
    )
    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)
