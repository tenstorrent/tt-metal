# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

# Import under a non-test name. pytest collects every test* binding in this module.
import test_topk_xl_parts as topk_parts
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES

pytestmark = [skip_for_wormhole, skip_for_quasar]

# The split sort is SFPU work on MATH: end to end, and with MATH isolated.
PERF_AXES = {
    **PERF_MEASUREMENT_AXES,
    "run_types": [[PerfRunType.L1_TO_L1, PerfRunType.MATH_ISOLATE]],
}


@pytest.mark.perf
@parametrize(parts=[2, 4], combine=[False, True], **PERF_AXES)
def test_perf_topk_xl_parts(
    perf_report, parts, combine, run_types, loop_factor, is_perf
):
    topk_parts._run(
        parts,
        combine=combine,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
