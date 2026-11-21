# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.llk_params import DestAccumulation, DestSync, GatedReduceScale
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_gated_reduce import BF16, BF16_TO_FP32, FP32, _run

pytestmark = blackhole_only


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    gate=["Silu", "ClampedSilu"],
    up=["Identity", "Clamp"],
    flags=[GatedReduceScale(flags) for flags in range(GatedReduceScale.All + 1)],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_gated_reduce_modes(
    perf_report, dest_acc, gate, up, flags, run_types, loop_factor, is_perf
):
    _run(
        BF16,
        dest_acc,
        gate,
        up,
        flags,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    rows=[4, 8, 16, 32],
    sync=[DestSync.Half, DestSync.Full],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_gated_reduce_geometry(
    perf_report, dest_acc, rows, sync, run_types, loop_factor, is_perf
):
    _run(
        BF16_TO_FP32,
        dest_acc,
        "ClampedSilu",
        "Clamp",
        GatedReduceScale.All,
        rows,
        sync,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@pytest.mark.parametrize("gate", ["Silu", "ClampedSilu"])
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_gated_reduce_fp32_input(
    perf_report, gate, run_types, loop_factor, is_perf
):
    _run(
        FP32,
        DestAccumulation.Yes,
        gate,
        "Clamp",
        GatedReduceScale.All,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_gated_reduce_round_once(perf_report, run_types, loop_factor, is_perf):
    _run(
        BF16,
        DestAccumulation.No,
        "Silu",
        "Identity",
        GatedReduceScale.All,
        rounding=True,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
