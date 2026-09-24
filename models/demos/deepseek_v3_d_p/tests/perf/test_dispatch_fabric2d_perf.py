# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf for dispatch_fabric2d, measured against production `dispatch` in the same capture.

The worker runs both ops on one set of inputs, so the CSV carries DispatchDeviceOperation and
DispatchFabric2dDeviceOperation side by side and the ratio is free of cross-run variance.

The metric is device kernel duration summed over the worker's launches, which is what
`run_model_device_perf_test_per_op` compares: `merge_device_rows` collapses the mesh's 32 rows per
launch, so each baseline covers the worker's warm-up plus its ITERATIONS measured launches. Re-measure
both baselines together whenever the worker's launch count changes.

At production geometry dispatch_fabric2d runs about 1.5x faster than production `dispatch`: relaying
through DRAM costs a store and a load per hop, but it replaces the production op's multi-hop fabric
routing and wins by more than the relay costs.

This gate spawns a CHILD pytest that opens the mesh itself, so it must not share a pytest session
with anything holding a device: the child blocks on the open and the run dies at the global 300 s
timeout with nothing useful in the log. CI selects it by path under
`-m models_device_performance_bare_metal`, which keeps it alone; a local run has to do the same.

Do not compare these against the figures in test_dispatch_combine_perf.py. Those are an 8x1 LoudBox
TorusY proxy replaying captured routing; this is 8x4 TORUS_XY.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import adjust_margin_for_ddr_speed, run_model_device_perf_test_per_op

_WORKER = (
    "models/demos/deepseek_v3_d_p/tests/perf/test_prefill_dispatch_fabric2d.py::test_dispatch_fabric2d_perf_worker"
)
_K_FILTER = "fabric2d-torus-xy-8x4-2link"

# Only the op this test exists for is gated. `DispatchDeviceOperation` is still in the capture -- the
# ratio between the two is the transferable quantity and it is right there in the log -- but it is not
# baselined here: production already has its own gates, and over three back-to-back captures at the
# hot profile it moved 1.5% against this op's 0.24%.
#
# Measured at the `hot` routing profile. Re-measure if PRODUCTION_ROUTING changes: at the uniform
# profile the same op runs roughly 40% faster, so a baseline taken under one profile fails under the
# other for no reason connected to the code.
#
# `run_model_device_perf_test_per_op` fails the test if an op substring matches no row, so a typo in a
# key shows up as a failure rather than a silent pass.
_EXPECTED_NS: dict[str, int] = {
    "DispatchFabric2dDeviceOperation": 10_384_185,
}


@pytest.mark.models_device_performance_bare_metal
def test_device_perf_dispatch_fabric2d():
    # The sibling gates' margin and DDR adjustment, so a slow-memory board loosens the same way. 3% is
    # ~150x this op's observed run-to-run spread, so the gate catches a real regression rather than noise.
    margin = adjust_margin_for_ddr_speed(0.03)
    run_model_device_perf_test_per_op(
        command=f"pytest {_WORKER} -k '{_K_FILTER}' --wrapper-invocation",
        expected_per_op=_EXPECTED_NS,
        subdir="deepseek_v3_dispatch_fabric2d",
        model_name="deepseek_v3_dispatch_fabric2d_torus_xy_8x4_2link",
        margin=margin,
        comments="torus-xy-8x4-2link",
    )
