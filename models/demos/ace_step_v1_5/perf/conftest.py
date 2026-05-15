# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Perf tests under ``perf/`` must use ``ttnn.open_device`` (demo path).

The repository root ``conftest.py`` exposes a ``device`` fixture built from ``ttnn.CreateDevice``,
which may yield a ``MeshDevice`` handle. ACE-Step VAE conv helpers are validated on the same
``open_device`` entry point as ``run_prompt_to_wav.py``; this nearest ``conftest.py`` overrides
``device`` so Tracy/pytest runs match that path.

Perf runs enable L1 placement for reshape/permute outputs (``perf1``/``perf2`` stacked ~49 % of
device time in those ops with ``in0:dram_interleaved``). Demos leave these unset (DRAM default).
"""

from __future__ import annotations

import os

import pytest

_DEFAULT_L1_SMALL = int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304"))

# Enable unless explicitly disabled (e.g. debugging demo parity inside perf/).
if os.environ.get("ACE_STEP_TM_OUTPUT_L1", "").strip() == "":
    os.environ.setdefault("ACE_STEP_TM_OUTPUT_L1", "1")


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
