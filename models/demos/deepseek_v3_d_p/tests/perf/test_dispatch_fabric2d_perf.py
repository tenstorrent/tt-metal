# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf for dispatch_fabric2d, measured against production `dispatch` in the same capture.

The worker runs both ops on one set of inputs, so the CSV carries DispatchDeviceOperation and
DispatchFabric2dDeviceOperation side by side and the ratio is free of cross-run variance.

The expected values below are PLACEHOLDERS, pending numbers from hardware qualified for perf.

The metric itself is sound: device kernel duration tracks the work. Dropping seq_len_per_chip from
640 to 64 takes both ops from ~3.18 ms to ~0.39 ms per launch, so it is measuring data movement and
not dispatch skew or a synchronization floor. At production geometry the two ops come out at parity
(3.18 vs 3.20 ms), which is what the analysis predicts: store-and-forward alone moves zero link
bytes, and the win has to come from fan-out.

Do not compare these against the 473k-1248k ns figures in test_dispatch_combine_perf.py. Those are
an 8x1 LoudBox TorusY proxy replaying captured routing; this is 8x4 TORUS_XY with a uniform draw.
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
    # The reach table multicast needs, produced on device. Additive: nothing else in the pipeline emits
    # it, so this is subtracted from whatever multicast saves over store-and-forward.
    "MoeFanoutReachDeviceOperation": 0,
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
