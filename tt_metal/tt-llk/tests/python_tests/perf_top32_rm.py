# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_top32_rm import (
    FORMATS,
    PRE_SORTED_FORMATS,
    PRE_SORTED_ROW_ELEMENTS,
    ROW_ELEMENTS,
)
from test_top32_rm import test_top32_rm as _test_top32_rm
from test_top32_rm import test_top32_rm_pre_sorted as _test_top32_rm_pre_sorted
from test_top32_rm import (
    test_top32_rm_pre_sorted_via_metal_wrappers as _test_top32_rm_pre_sorted_via_metal_wrappers,
)
from test_top32_rm import (
    test_top32_rm_via_metal_wrappers as _test_top32_rm_via_metal_wrappers,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    row_elements=ROW_ELEMENTS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_top32_rm(
    perf_report, formats, row_elements, dest_acc, run_types, loop_factor, is_perf
):
    _test_top32_rm(
        formats,
        row_elements,
        dest_acc,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    formats=PRE_SORTED_FORMATS,
    row_elements=PRE_SORTED_ROW_ELEMENTS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_top32_rm_pre_sorted(
    perf_report, formats, row_elements, run_types, loop_factor, is_perf
):
    _test_top32_rm_pre_sorted(
        formats,
        row_elements,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    row_elements=[64, 160],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_top32_rm_via_metal_wrappers(
    perf_report, formats, row_elements, dest_acc, run_types, loop_factor, is_perf
):
    _test_top32_rm_via_metal_wrappers(
        formats,
        row_elements,
        dest_acc,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    formats=PRE_SORTED_FORMATS,
    row_elements=[1024, 1088],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_top32_rm_pre_sorted_via_metal_wrappers(
    perf_report, formats, row_elements, run_types, loop_factor, is_perf
):
    _test_top32_rm_pre_sorted_via_metal_wrappers(
        formats,
        row_elements,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
