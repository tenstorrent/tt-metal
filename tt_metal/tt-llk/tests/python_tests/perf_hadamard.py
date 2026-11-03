# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

# Import under a non-test name. pytest collects every test* binding in this module.
import test_hadamard as hadamard
from conftest import blackhole_only
from helpers.llk_params import ALL_MATH_FIDELITIES, DestSync
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES

pytestmark = blackhole_only


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(fidelity=ALL_MATH_FIDELITIES, normalize=[False, True], num_tiles=[1, 8])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_hadamard_h128(
    perf_report, fidelity, normalize, num_tiles, run_types, loop_factor, is_perf
):
    hadamard.invoke_with_perf(
        hadamard.test_hadamard_h128,
        fidelity,
        normalize,
        num_tiles,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_hadamard_h128_h16_tile_index(
    perf_report, run_types, loop_factor, is_perf
):
    hadamard.invoke_with_perf(
        hadamard.test_hadamard_h128_h16_tile_index,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(dest_sync=[DestSync.Half, DestSync.Full], normalize=[False, True])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_hadamard_h128_shipping_config(
    perf_report, dest_sync, normalize, run_types, loop_factor, is_perf
):
    # Single-axis @parametrize wraps the value as a 1-tuple; keep dest_sync+normalize
    # together so they unpack as scalars (same axes as the functional test).
    hadamard.invoke_with_perf(
        hadamard.test_hadamard_h128_shipping_config,
        dest_sync,
        normalize,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(num_tiles=[1, 2, 4])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_hadamard_h128_dest_acc(
    perf_report, num_tiles, run_types, loop_factor, is_perf
):
    hadamard.invoke_with_perf(
        hadamard.test_hadamard_h128_dest_acc,
        num_tiles,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
