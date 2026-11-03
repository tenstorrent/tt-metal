# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sdpa_custom_mm_reuse_dest_srcb import FORMATS, KT_NT_DIMS, _run

pytestmark = [skip_for_wormhole, skip_for_quasar]


@pytest.mark.perf
@parametrize(
    math_fidelity=[MathFidelity.LoFi],
    formats=FORMATS,
    dest_acc=[DestAccumulation.No],
    kt_nt_dims=KT_NT_DIMS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sdpa_custom_mm_reuse_dest_srcb(
    perf_report,
    math_fidelity,
    formats,
    dest_acc,
    kt_nt_dims,
    run_types,
    loop_factor,
    is_perf,
):
    kt_dim, nt_dim = kt_nt_dims
    _run(
        math_fidelity,
        formats,
        dest_acc,
        kt_dim,
        nt_dim,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
