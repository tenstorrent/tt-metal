# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_sdpa import (
    CORRECTION_SCALE_BF16_VALUES,
    Precision,
    _dest_configurations,
    _variants,
    test_sfpu_sdpa,
    test_sfpu_sdpa_correction,
)


@pytest.mark.perf
@parametrize(
    variant=_variants(),
    precision=list(Precision),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_sdpa(
    perf_report, variant, precision, run_types, loop_factor, is_perf
):
    test_sfpu_sdpa(
        variant,
        precision,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(
    dest_config=_dest_configurations(),
    scale_bf16=list(CORRECTION_SCALE_BF16_VALUES),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_sdpa_correction(
    perf_report, dest_config, scale_bf16, run_types, loop_factor, is_perf
):
    test_sfpu_sdpa_correction(
        dest_config,
        scale_bf16,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
