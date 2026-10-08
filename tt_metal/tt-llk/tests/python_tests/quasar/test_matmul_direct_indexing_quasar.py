# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.constraints import get_valid_math_fidelities
from helpers.llk_params import ImpliedMathFormat, PerfRunType
from helpers.param_config import parametrize, runtime
from quasar.test_matmul_quasar import (
    FULL_MATMUL_SHAPES,
    MATMUL_FORMAT,
    matmul_dest_acc_modes,
    matmul_dest_sync_modes,
    matmul_tile_dimensions,
    matmul_transpose_modes,
)
from quasar.test_matmul_quasar import test_matmul as run_matmul

# Direct indexing is full-tile only, and the MxFp4 2x path already sweeps it in
# test_matmul, so this covers the plain (non-2x) MVMULDI MOP.
DIRECT_INDEXING_FORMATS = [
    format
    for format in MATMUL_FORMAT
    if not format.input_format.is_mx_format()
    and not format.output_format.is_mx_format()
]
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
    format=DIRECT_INDEXING_FORMATS,
    math_fidelity=lambda format: get_valid_math_fidelities(format),
    dest_sync_mode=lambda: matmul_dest_sync_modes(),
    dest_acc=matmul_dest_acc_modes,
    matmul_tile_dims=runtime(matmul_direct_indexing_tile_dimensions),
    implied_math_format=[ImpliedMathFormat.Yes],
    register_format_hint=[None],
    enable_direct_indexing=[True, False],
    transpose=matmul_transpose_modes,
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
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
    enable_direct_indexing,
    transpose,
    run_types,
    loop_factor,
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
        enable_direct_indexing,
        transpose,
        run_types,
        loop_factor,
    )
