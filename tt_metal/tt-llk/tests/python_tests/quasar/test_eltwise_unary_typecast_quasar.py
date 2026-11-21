# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import ApproximationMode, MathOperation
from helpers.param_config import parametrize, runtime
from quasar.test_eltwise_unary_sfpu_quasar import (
    DEST_SYNC_MODES,
    TENSOR_DIMS,
    _dest_acc_for_typecast_formats,
    run_eltwise_unary_sfpu_quasar,
    typecast_formats,
)


@pytest.mark.quasar
@parametrize(
    mathop=[MathOperation.Typecast],
    formats=typecast_formats(),
    dest_acc=_dest_acc_for_typecast_formats,
    approx_mode=[ApproximationMode.No],
    dest_sync=list(DEST_SYNC_MODES),
    input_dimensions=runtime(list(TENSOR_DIMS)),
)
def test_eltwise_unary_typecast_quasar(
    mathop,
    formats,
    dest_acc,
    approx_mode,
    dest_sync,
    input_dimensions,
):
    """Typecast family: same unary SFPU kernel, paired with BH typecast module."""
    run_eltwise_unary_sfpu_quasar(
        formats,
        dest_acc,
        mathop,
        approx_mode,
        input_dimensions,
        dest_sync,
    )
