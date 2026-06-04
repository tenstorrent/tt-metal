# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Perf tests under ``perf/`` must use ``ttnn.open_device`` (demo path).

The repository root ``conftest.py`` exposes a ``device`` fixture built from ``ttnn.CreateDevice``,
which may yield a ``MeshDevice`` handle. ACE-Step VAE conv helpers are validated on the same
``open_device`` entry point as ``run_prompt_to_wav.py``; this nearest ``conftest.py`` overrides
``device`` so Tracy/pytest runs match that path.
"""


from __future__ import annotations

import os

import pytest

_DEFAULT_L1_SMALL = int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304"))
# 2 CQs enable host->device copy on CQ 1 while the trace runs on CQ 0; the trace+2CQ perf tests
# need this on by default, while legacy single-CQ runs are preserved with ACE_STEP_NUM_CQS=1.
_DEFAULT_NUM_CQS = int(os.environ.get("ACE_STEP_NUM_CQS", "2"))


@pytest.fixture(scope="session")
def device():
    ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
    open_kwargs = dict(
        device_id=int(os.environ.get("TT_DEVICE_ID", "0")),
        l1_small_size=_DEFAULT_L1_SMALL,
        trace_region_size=128 << 20,
    )
    if _DEFAULT_NUM_CQS > 1:
        open_kwargs["num_command_queues"] = _DEFAULT_NUM_CQS
    dev = ttnn.open_device(**open_kwargs)

    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


@pytest.fixture
def trace_device(device):
    """Shared device for ``perf/module_trace/test_*_trace_2cq.py`` — aliases the perf ``device``.

    Overrides the demo conftest's ``trace_device(mesh_device)``. Tests under ``perf/`` already
    use the perf-level ``device`` fixture (session-scoped, 2 CQs + 128 MB trace region); making
    ``trace_device`` an alias means a pytest session that runs both ``perf/test_perf_*.py`` and
    ``perf/module_trace/test_*_trace_2cq.py`` opens **one** device on the physical hardware. Two
    handles on the same ``device_id`` corrupt TT's dispatch-state global and break the
    session-level ``close_device`` sync; avoid it.

    Skips if the TTNN build is missing the trace API.
    """
    ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
    if not hasattr(ttnn, "begin_trace_capture") or not hasattr(ttnn, "execute_trace"):
        pytest.skip("TTNN build missing trace API (begin_trace_capture / execute_trace)")
    return device
