# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device perf for combine_fabric2d.

The measured number is the op's device time summed over all of the worker's launches, warm-up
included. If the worker's launch count changes, re-measure the baseline.

This test starts a separate pytest that opens the device itself, so nothing else in the same session
may hold the device: the child would wait for it until the run times out. CI selects it by path under
`-m models_device_performance_bare_metal`, which keeps it alone; a local run has to do the same.

These numbers are for 8x4 TORUS_XY. Do not compare them with test_dispatch_combine_perf.py, which
runs on an 8x1 LoudBox with recorded routing.
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import adjust_margin_for_ddr_speed, run_model_device_perf_test_per_op

_WORKER = "models/demos/deepseek_v3_d_p/tests/perf/test_prefill_combine_fabric2d.py::test_combine_fabric2d_perf_worker"
_K_FILTER = "fabric2d-torus-xy-8x4-2link"

# To be measured on 8x4 with the worker's synthetic routing, `PRODUCTION_ROUTING` in
# op_unit_tests/test_combine_fabric2d.py; until then 0 fails the gate. Re-measure if the routing changes.
_EXPECTED_NS: dict[str, int] = {
    "CombineFabric2dDeviceOperation": 0,
}


@pytest.mark.models_device_performance_bare_metal
def test_device_perf_combine_fabric2d():
    # Same 3% margin and slow-memory adjustment as the other DeepSeek perf tests.
    margin = adjust_margin_for_ddr_speed(0.03)
    run_model_device_perf_test_per_op(
        command=f"pytest {_WORKER} -k '{_K_FILTER}' --wrapper-invocation",
        expected_per_op=_EXPECTED_NS,
        subdir="deepseek_v3_combine_fabric2d",
        model_name="deepseek_v3_combine_fabric2d_torus_xy_8x4_2link",
        margin=margin,
        comments="torus-xy-8x4-2link",
    )
