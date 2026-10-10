# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_math_fidelities
from helpers.llk_params import PerfRunType
from helpers.param_config import parametrize, runtime
from quasar.test_matmul_quasar import (
    FULL_MATMUL_SHAPES,
    NON_MX_MATMUL_FORMATS,
    matmul_dest_acc_modes,
    matmul_dest_sync_modes,
    matmul_implied_math_formats,
    matmul_tile_dimensions,
    matmul_transpose_modes,
)
from quasar.test_matmul_quasar import test_matmul as run_matmul

DIRECT_INDEXING_KT_DIMS = (1, 2)


def matmul_direct_indexing_tile_dimensions(dest_acc, dest_sync_mode):
    """Single tile plus every dest-filling (ct, rt) block: reuse-A, reuse-B and strided-dest."""
    shapes = {(1, 1)} | {
        (ct_dim, rt_dim)
        for ct_dim, rt_dim, _ in matmul_tile_dimensions(
            dest_acc, dest_sync_mode, exact_dest_fill=True
        )
    }
    return [
        (ct_dim, rt_dim, kt_dim)
        for ct_dim, rt_dim in sorted(shapes)
        for kt_dim in DIRECT_INDEXING_KT_DIMS
    ]


@pytest.mark.nightly
@pytest.mark.quasar
@parametrize(
    input_tile_dimensions=runtime(FULL_MATMUL_SHAPES),
    format=NON_MX_MATMUL_FORMATS,
    math_fidelity=lambda format: get_valid_math_fidelities(format),
    dest_sync_mode=lambda: matmul_dest_sync_modes(),
    dest_acc=matmul_dest_acc_modes,
    matmul_tile_dims=runtime(matmul_direct_indexing_tile_dimensions),
    implied_math_format=lambda format: matmul_implied_math_formats(format),
    register_format_hint=[None],
    transpose=matmul_transpose_modes,
)
def test_matmul_direct_indexing(
    input_tile_dimensions,
    matmul_tile_dims,
    math_fidelity,
    dest_sync_mode,
    dest_acc,
    format,
    implied_math_format,
    register_format_hint,
    transpose,
):
    run_matmul(
        input_tile_dimensions,
        matmul_tile_dims,
        math_fidelity,
        dest_sync_mode,
        dest_acc,
        format,
        implied_math_format,
        register_format_hint,
        enable_direct_indexing=True,
        transpose=transpose,
        run_types=[PerfRunType.L1_TO_L1],
        loop_factor=1,
    )
