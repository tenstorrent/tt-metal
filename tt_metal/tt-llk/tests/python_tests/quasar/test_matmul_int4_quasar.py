# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Int8 activations (A) x Int4 weights (B) -> Int32, the integer core of a
# wint4_aint8 linear. The unpacker widens the Int4 weights to Int8 on the way
# into SrcA, so the FPU runs a plain int8 matmul. Int8 x Int8 rides the same
# kernel as a control.

import pytest
import torch
from helpers.chip_architecture import is_4row_arch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    PerfRunType,
    Transpose,
    format_dict,
)
from helpers.param_config import DEST_SYNC_TILE_LIMITS, parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    ENABLE_2X_FORMAT,
    ENABLE_DIRECT_INDEXING,
    IMPLIED_MATH_FORMAT,
    IN_FACE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tile_shape import construct_tile_shape
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

TILE = 32

# Value range the unpacker can see. Both are sign-magnitude, so -8 / -128
# have no encoding.
INT_RANGES = {
    DataFormat.Int4: (-7, 7),
    DataFormat.Int8: (-127, 127),
}

MATMUL_INT4_FORMATS = [
    InputOutputFormat(DataFormat.Int8, DataFormat.Int32),
    InputOutputFormat(
        DataFormat.Int8, DataFormat.Int32, input_format_B=DataFormat.Int4
    ),
]

# (ct, rt, kt). kt=4 is K=128, one quantization group.
MATMUL_TILE_DIMS = [(1, 1, 1), (1, 1, 4), (2, 2, 4), (4, 1, 2), (1, 4, 2)]


def matmul_tile_dims(dest_sync):
    # 32-bit Dest halves the tile capacity.
    max_tiles = DEST_SYNC_TILE_LIMITS[dest_sync] // 2
    return [dims for dims in MATMUL_TILE_DIMS if dims[0] * dims[1] <= max_tiles]


def generate_int_matrix(fmt: DataFormat, dimensions, seed: int):
    lo, hi = INT_RANGES[fmt]
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        lo, hi + 1, (dimensions[0] * dimensions[1],), generator=generator
    ).to(format_dict[fmt])


@pytest.mark.quasar
@parametrize(
    format=MATMUL_INT4_FORMATS,
    dest_sync_mode=[DestSync.Half],
    matmul_dims=lambda dest_sync_mode: matmul_tile_dims(dest_sync_mode),
    implied_math_format=[ImpliedMathFormat.Yes, ImpliedMathFormat.No],
)
def test_matmul_int4_quasar(format, dest_sync_mode, matmul_dims, implied_math_format):
    _run_matmul_int4(format, dest_sync_mode, matmul_dims, implied_math_format)


@pytest.mark.quasar
@pytest.mark.skipif(
    not is_4row_arch(),
    reason="Int8_2x register format exists only on the 4-row Quasar variant",
)
@parametrize(
    format=MATMUL_INT4_FORMATS,
    dest_sync_mode=[DestSync.Half],
    matmul_dims=lambda dest_sync_mode: matmul_tile_dims(dest_sync_mode),
    enable_direct_indexing=[True, False],
)
def test_matmul_int4_2x_quasar(
    format, dest_sync_mode, matmul_dims, enable_direct_indexing
):
    # Int4 weights unpack straight into the 2x-packed Int8 SrcA storage.
    format = InputOutputFormat(
        format.input_format,
        format.output_format,
        input_format_B=format.input_format_B,
        register_format_hint=DataFormat.Int8_2x,
    )
    _run_matmul_int4(
        format,
        dest_sync_mode,
        matmul_dims,
        ImpliedMathFormat.Yes,
        enable_direct_indexing=enable_direct_indexing,
    )


def _run_matmul_int4(
    format,
    dest_sync_mode,
    matmul_dims,
    implied_math_format,
    enable_direct_indexing=False,
):
    ct_dim, rt_dim, kt_dim = matmul_dims
    dest_acc = DestAccumulation.Yes
    tile_dimensions = [TILE, TILE]
    tile_shape = construct_tile_shape(tile_dimensions)

    # A = activations (M x K), B = weights (K x N).
    input_A_dimensions = [rt_dim * TILE, kt_dim * TILE]
    input_B_dimensions = [kt_dim * TILE, ct_dim * TILE]
    output_dimensions = [rt_dim * TILE, ct_dim * TILE]
    tile_cnt_A = rt_dim * kt_dim
    tile_cnt_B = kt_dim * ct_dim
    output_tile_cnt = rt_dim * ct_dim

    src_A = generate_int_matrix(format.input_format, input_A_dimensions, seed=1)
    src_B = generate_int_matrix(format.input_format_B, input_B_dimensions, seed=2)

    # Exact integer reference; |sum| <= 127 * 127 * K fits Int32 comfortably.
    golden = torch.matmul(
        src_A.to(torch.int64).view(input_A_dimensions),
        src_B.to(torch.int64).view(input_B_dimensions),
    ).to(torch.int32)

    golden_tensor = tilize_block(
        golden.flatten(),
        dimensions=output_dimensions,
        stimuli_format=format.output_format,
        tile_dimensions=tile_dimensions,
    ).flatten()

    tilized_A = tilize_block(
        src_A,
        dimensions=input_A_dimensions,
        stimuli_format=format.input_format,
        tile_dimensions=tile_dimensions,
    ).flatten()
    tilized_B = tilize_block(
        src_B,
        dimensions=input_B_dimensions,
        stimuli_format=format.input_format_B,
        tile_dimensions=tile_dimensions,
    ).flatten()

    num_faces = tile_shape.total_num_faces()
    test_config_kwargs = {
        "test_name": "sources/quasar/matmul_quasar_test.cpp",
        "formats": format,
        "templates": [
            MATH_FIDELITY(MathFidelity.LoFi),
            IMPLIED_MATH_FORMAT(implied_math_format),
            ENABLE_2X_FORMAT(format.register_format_hint is not None),
            ENABLE_DIRECT_INDEXING(enable_direct_indexing),
            DEST_SYNC(dest_sync_mode),
            UNPACK_TRANS_FACES(Transpose.No),
        ],
        "runtimes": [
            CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
            TILE_COUNT(output_tile_cnt * kt_dim),
            NUM_FACES(num_faces, num_faces, num_faces),
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim, tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim, tile_shape.num_faces_c_dim),
            LOOP_FACTOR(1),
        ],
        "variant_stimuli": StimuliConfig(
            tilized_A,
            format.input_format,
            tilized_B,
            format.input_format_B,
            format.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=output_tile_cnt,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
    }
    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=[PerfRunType.L1_TO_L1],
        test_config_kwargs=test_config_kwargs,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format.output_format])

    assert passed_test(
        golden_tensor, res_tensor, format.output_format
    ), "Assert against golden failed"
