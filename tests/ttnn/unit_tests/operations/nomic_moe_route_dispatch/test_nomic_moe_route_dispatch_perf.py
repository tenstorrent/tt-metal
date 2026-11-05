# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-ns perf rows for nomic_moe_route_dispatch (the prompt's performance table).

Each row runs the production config (x in the layer norm's ND-sharded L1 layout, `dispatched` in L1,
default compute config, layer-0 input profile) with the model's L1 occupancy reproduced. Run under the
device profiler to read DEVICE KERNEL DURATION per call:

    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/nomic_moe_route_dispatch/test_nomic_moe_route_dispatch_perf.py

Every row is called WARM_CALLS + 1 times; the last call of each row is the measurement (program cached).
test_perf_row_dram_out runs the same rows with `dispatched` in DRAM interleaved (regime R4);
test_perf_row_interleaved_in with x DRAM / L1 interleaved (regime R2).
"""

import pytest

import ttnn
from eval.golden_tests.nomic_moe_route_dispatch.feature_spec import LOOSE_CASES
from eval.golden_tests.nomic_moe_route_dispatch.helpers import make_weight, make_x, case_seed, occupy_l1, to_device
from ttnn.operations.nomic_moe_route_dispatch import nomic_moe_route_dispatch

WARM_CALLS = 1


def _run_row(device, case, memory_config=None, in_placement=None):
    x_shape, _, spare_tile_rows, profile, name = case["inputs"]
    w = make_weight(profile)
    x = make_x(x_shape, w, profile, seed=case_seed(name))
    x_dev, w_dev = to_device(x, w, device, in_placement=in_placement or case["in_placement"])
    filler = occupy_l1(device, case["extras"]["l1_free_bytes_before_call"])
    try:
        for _ in range(WARM_CALLS + 1):
            outputs = nomic_moe_route_dispatch(
                x_dev, w_dev, spare_tile_rows=spare_tile_rows, memory_config=memory_config
            )
            ttnn.synchronize_device(device)
            for tensor in outputs:
                ttnn.deallocate(tensor)
    finally:
        ttnn.deallocate(filler)


_IDS = [c["inputs"][4] for c in LOOSE_CASES]


@pytest.mark.parametrize("case", LOOSE_CASES, ids=_IDS)
def test_perf_row(device, case):
    _run_row(device, case)


@pytest.mark.parametrize("case", LOOSE_CASES, ids=_IDS)
def test_perf_row_dram_out(device, case):
    _run_row(device, case, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("in_placement", ["dram_interleaved", "l1_interleaved"])
@pytest.mark.parametrize("case", LOOSE_CASES, ids=_IDS)
def test_perf_row_interleaved_in(device, case, in_placement):
    _run_row(device, case, in_placement=in_placement)
