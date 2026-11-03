# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_softmax_k import ALL_K, FORMATS, test_sfpu_softmax_k


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    k=ALL_K,
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_sfpu_softmax_k(
    perf_report, formats, dest_acc, k, run_types, loop_factor, is_perf
):
    test_sfpu_softmax_k(
        formats,
        dest_acc,
        k,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
