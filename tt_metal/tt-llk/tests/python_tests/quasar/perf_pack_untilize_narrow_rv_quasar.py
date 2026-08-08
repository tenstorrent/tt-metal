# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Perf harness for the NARROW-ROW RV_PACR untilize demo (Quasar).
#
# Profiles the end-to-end L1_TO_L1 path of pack_untilize_narrow_rv_quasar_test.cpp:
# unpack (unary operand) -> math (A2D datacopy) -> pack (RV_PACR narrow-row untilize).
# LOOP_FACTOR repeats the steady-state work so the INIT and TILE_LOOP profiler zones capture stable cycle counts.
#
# Only L1_TO_L1 is swept for now; the stage-isolate / congestion run types would need
# per-thread dvalid mock accounting for the custom RV_PACR pack loop (see the quasar-perf-test skill).

import pytest
from helpers.llk_params import PERF_LOOP_FACTOR_QUASAR, PerfRunType
from helpers.param_config import parametrize
from quasar.test_pack_untilize_narrow_rv_quasar import (
    CASES,
    NARROW_RV_FORMATS,
)
from quasar.test_pack_untilize_narrow_rv_quasar import (
    test_pack_untilize_narrow_rv_quasar as run_pack_untilize_narrow_rv,
)

# L1_TO_L1 only (end-to-end). Extend with UNPACK_ISOLATE / MATH_ISOLATE / PACK_ISOLATE /
# L1_CONGESTION once the per-thread dvalid mocks for the RV_PACR pack loop are wired up.
PERF_RUN_TYPES_NARROW_RV = [PerfRunType.L1_TO_L1]


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=NARROW_RV_FORMATS,
    case=CASES,
)
def test_perf_pack_untilize_narrow_rv_quasar(
    perf_report,
    formats,
    case,
):
    run_pack_untilize_narrow_rv(
        formats,
        case,
        run_types=PERF_RUN_TYPES_NARROW_RV,
        loop_factor=PERF_LOOP_FACTOR_QUASAR,
        is_perf=True,
        perf_report=perf_report,
    )
