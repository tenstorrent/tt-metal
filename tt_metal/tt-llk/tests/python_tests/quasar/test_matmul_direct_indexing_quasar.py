# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_math_fidelities
from helpers.llk_params import DestSync, PerfRunType
from helpers.param_config import parametrize, quasar_dest_sync_modes, runtime
from quasar.test_matmul_quasar import (
    FULL_MATMUL_SHAPES,
    NON_MX_MATMUL_FORMATS,
    matmul_dest_acc_modes,
    matmul_exact_fill_ct_dims,
    matmul_exact_fill_rt_dims,
    matmul_transpose_modes,
)
from quasar.test_matmul_quasar import test_matmul as run_matmul

DIRECT_INDEXING_KT_DIMS = [1, 2]


def matmul_direct_indexing_ct_dims(dest_acc, dest_sync, rt_dim):
    """The exact Dest fill, plus a single tile when rt is 1."""
    ct_dims = {matmul_exact_fill_ct_dims(dest_acc, dest_sync, rt_dim)[0]}
    if rt_dim == 1:
        ct_dims.add(1)
    return sorted(ct_dims)


@pytest.mark.nightly
@pytest.mark.quasar
@parametrize(
    input_tile_dimensions=runtime(FULL_MATMUL_SHAPES),
    formats=NON_MX_MATMUL_FORMATS,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    dest_sync=lambda: quasar_dest_sync_modes(),
    dest_acc=matmul_dest_acc_modes,
    rt_dim=runtime(matmul_exact_fill_rt_dims),
    ct_dim=runtime(matmul_direct_indexing_ct_dims),
    kt_dim=runtime(DIRECT_INDEXING_KT_DIMS),
    register_format_hint=[None],
    unpack_transpose_faces=matmul_transpose_modes,
)
def test_matmul_direct_indexing(
    input_tile_dimensions,
    ct_dim,
    rt_dim,
    kt_dim,
    math_fidelity,
    dest_sync,
    dest_acc,
    formats,
    register_format_hint,
    unpack_transpose_faces,
):
    run_matmul(
        input_tile_dimensions,
        ct_dim,
        rt_dim,
        kt_dim,
        math_fidelity,
        dest_sync,
        dest_acc,
        formats,
        register_format_hint,
        enable_direct_indexing=True,
        unpack_transpose_faces=unpack_transpose_faces,
        run_types=[PerfRunType.L1_TO_L1],
        loop_factor=2 if dest_sync == DestSync.Half else 1,
    )
