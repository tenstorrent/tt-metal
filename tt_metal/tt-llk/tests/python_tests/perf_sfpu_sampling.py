# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.llk_params import DestAccumulation, VectorMode
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_sampling import FORMATS, SAMPLING_OPS
from test_sfpu_sampling import test_sfpu_sampling as _test_sfpu_sampling


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    op=list(SAMPLING_OPS.keys()),
    vector_mode=[VectorMode.None_, VectorMode.C],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_sampling(
    perf_report, formats, dest_acc, op, vector_mode, run_types, loop_factor, is_perf
):
    _test_sfpu_sampling(
        formats,
        dest_acc,
        op,
        vector_mode,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
