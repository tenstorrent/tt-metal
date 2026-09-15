# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf for dispatch_fabric2d, measured against production `dispatch` in the same capture.

The worker runs both ops on one set of inputs, so the CSV carries DispatchDeviceOperation and
DispatchFabric2dDeviceOperation side by side and the ratio is free of cross-run variance.

The expected values below are PLACEHOLDERS and the worker is NOT YET TRACED.

Untraced, every launch is dispatched from host separately, so the 32 chips start skewed and a fabric
op's kernel duration -- which includes waiting on peers -- is bounded by that skew rather than by the
op. Measured that way both ops land within 0.6% of each other at ~3.18 ms per launch, against a
published `dispatch` baseline of 473k-1248k ns. Those numbers say nothing about either
implementation. Trace capture/replay has to land before anything here is treated as a measurement.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op

_WORKER = (
    "models/demos/deepseek_v3_d_p/tests/perf/test_prefill_dispatch_fabric2d.py"
    "::test_dispatch_fabric2d_perf_worker"
)
_K_FILTER = "torus-xy-8x4-2link"

# Unset until measured on qualified hardware. `run_model_device_perf_test_per_op` fails the test if
# an op substring matches no row, so a typo in a key shows up as a failure rather than a silent pass.
_EXPECTED_NS: dict[str, int] = {
    "DispatchDeviceOperation": 0,
    "DispatchFabric2dDeviceOperation": 0,
}


@pytest.mark.parametrize("margin", [0.045])
@pytest.mark.models_device_performance_bare_metal
def test_device_perf_dispatch_fabric2d(margin):
    if not all(_EXPECTED_NS.values()):
        pytest.skip("expected_ns not yet measured on a perf-qualified machine; see module docstring")
    run_model_device_perf_test_per_op(
        command=f"pytest {_WORKER} -k '{_K_FILTER}' --wrapper-invocation",
        expected_per_op=_EXPECTED_NS,
        subdir="deepseek_v3_dispatch_fabric2d",
        model_name="deepseek_v3_dispatch_fabric2d_torus_xy_8x4_2link",
        margin=margin,
        comments="torus-xy-8x4-2link",
    )
