# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_sdpa_fw import Precision, _variants, test_sfpu_sdpa_fw


@pytest.mark.perf
@parametrize(
    variant=_variants(),
    precision=list(Precision),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_sdpa_fw(
    perf_report, variant, precision, run_types, loop_factor, is_perf
):
    test_sfpu_sdpa_fw(
        variant,
        precision,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
