# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    ApproximationMode,
    ImpliedMathFormat,
    MathOperation,
    PerfRunType,
)
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from quasar.test_eltwise_unary_sfpu_quasar import (
    DEST_SYNC_MODES,
    TENSOR_DIMS,
    _dest_acc_for_typecast_formats,
    run_eltwise_unary_sfpu_quasar,
    typecast_formats,
)


def _schema_columns_match_unary_helper():
    """Header-gate source for this module: same templates as the shared unary helper."""
    create_test_or_perf_config(
        is_perf=False,
        run_types=(),
        test_config_kwargs={
            "templates": [
                MATH_OP(mathop=MathOperation.Typecast),
                APPROX_MODE(),
                IMPLIED_MATH_FORMAT(),
                DATA_COPY_TYPE(),
                UNPACKER_ENGINE_SEL(),
                DEST_SYNC(),
                TYPECAST_FORMATS(),
            ],
            "runtimes": [
                TILE_COUNT(),
                NUM_FACES(),
                TEST_FACE_DIMS(),
                DEST_INDEX(),
                LOOP_FACTOR(),
            ],
        },
    )


@pytest.mark.quasar
@parametrize(
    formats=typecast_formats(),
    dest_acc=_dest_acc_for_typecast_formats,
    approx_mode=[ApproximationMode.No],
    dest_sync=list(DEST_SYNC_MODES),
    implied_math_format=[ImpliedMathFormat.No],
    input_dimensions=list(TENSOR_DIMS),
)
def test_eltwise_unary_typecast_quasar(
    formats,
    dest_acc,
    approx_mode,
    dest_sync,
    implied_math_format,
    input_dimensions,
    *,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor=1,
    is_perf=False,
    perf_report=None,
):
    """Typecast family: same unary SFPU kernel, paired with BH typecast module."""
    run_eltwise_unary_sfpu_quasar(
        formats,
        dest_acc,
        MathOperation.Typecast,
        approx_mode,
        input_dimensions,
        dest_sync,
        implied_math_format,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
