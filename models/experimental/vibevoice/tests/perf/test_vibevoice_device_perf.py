# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice-1.5B — device performance (Tracy per-op kernel time, eager, **no metal trace**).

Outer driver (this file) does **not** import ttnn — Tracy spawns the inner pytest.
Inner ``test_device_perf_forwards.py::test_lm``:

  * warms outside the window
  * ``ReadDeviceProfiler`` clears load/warmup markers
  * ``start``/``stop`` signposts bracket one prefill chunk + 2 decode steps

Report aggregation uses ``has_signposts=True`` (Voxtral stage-perf pattern).

Usage::

    pytest models/experimental/vibevoice/tests/perf/test_vibevoice_device_perf.py \\
        -v -m models_device_performance_bare_metal
"""

from __future__ import annotations

import pytest

from models.common.utility_functions import run_for_blackhole
from models.perf.device_perf_utils import prep_device_perf_report, run_device_perf

_PERF_TEST = "models/experimental/vibevoice/tests/perf/test_device_perf_forwards.py"
# Headroom for weight load + warmup before ReadDeviceProfiler clears the buffer
# (Voxtral uses 75k for a similar TTS load path).
_OP_SUPPORT_COUNT = 75000


@run_for_blackhole("VibeVoice LM device perf is targeted for P150/Blackhole")
@pytest.mark.timeout(3600)
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, model_name",
    [(1, "vibevoice_lm")],
)
def test_perf_device_bare_metal_vibevoice_lm(batch_size, model_name):
    """LM device-bound perf via Tracy on a signposted eager prefill+decode window."""
    subdir = f"ttnn_{model_name}"
    num_iterations = 1
    # Quiet inner run — load under Tracy is noisy with -sv.
    # Match inner ``test_lm`` (@pytest.mark.timeout(1800)); Tracy load + measured pass.
    command = f"pytest --timeout=1800 {_PERF_TEST}::test_lm -q"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    post_processed_results = run_device_perf(
        command,
        subdir,
        num_iterations,
        cols,
        batch_size,
        has_signposts=True,
        op_support_count=_OP_SUPPORT_COUNT,
    )

    prep_device_perf_report(
        model_name=f"ttnn_{model_name}_batch{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results={},
        comments="vibevoice_lm_prefill256_decode2_eager_signposted",
    )
