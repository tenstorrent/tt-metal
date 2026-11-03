# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_rmsnorm_bcast_scalar_dest_reuse import (
    FORMATS,
    NUM_TILES,
    OPS,
    _clear_dest_values,
    _fidelities,
    _run,
    _skip_unsupported,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    math_fidelity=lambda mathop: _fidelities(mathop),
    num_tiles=NUM_TILES,
    clear_dest=lambda mathop: _clear_dest_values(mathop),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_rmsnorm_bcast_scalar_dest_reuse(
    perf_report,
    formats,
    dest_acc,
    mathop,
    math_fidelity,
    num_tiles,
    clear_dest,
    run_types,
    loop_factor,
    is_perf,
):
    _skip_unsupported(formats, dest_acc)
    _run(
        formats,
        dest_acc,
        mathop,
        math_fidelity,
        num_tiles,
        clear_dest=clear_dest,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    num_faces=[1, 2],
    num_tiles=[1, 2],
    math_fidelity=lambda mathop: _fidelities(mathop),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_rmsnorm_bcast_scalar_dest_reuse_partial_faces(
    perf_report,
    formats,
    dest_acc,
    mathop,
    num_faces,
    num_tiles,
    math_fidelity,
    run_types,
    loop_factor,
    is_perf,
):
    _skip_unsupported(formats, dest_acc)
    _run(
        formats,
        dest_acc,
        mathop,
        math_fidelity,
        num_tiles=num_tiles,
        num_faces=num_faces,
        clear_dest=True,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=OPS,
    math_fidelity=lambda mathop: _fidelities(mathop),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_rmsnorm_bcast_scalar_dest_reuse_unpack_full_transpose(
    perf_report,
    formats,
    dest_acc,
    mathop,
    math_fidelity,
    run_types,
    loop_factor,
    is_perf,
):
    _skip_unsupported(formats, dest_acc)
    _run(
        formats,
        dest_acc,
        mathop,
        math_fidelity,
        num_tiles=1,
        clear_dest=True,
        unpack_full_transpose=True,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
