# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional test of the reduce with a tilized operand A (tilizeA_B_reduce_init and unpack_tilizeA_B_block, the pool2d
configuration): a row-major block of tiles with face_r_dim rows per face, dense in L1 as the pool's input CB, is
tilized by the unpacker against a one-row scaler operand, reduced over its rows and packed one row per face. Row 0 of
faces 0 and 1 of every output tile is checked against the column max (MAX) or the scaled column sum (AVG) of the input
rows. The block is unpacked by num_blocks calls of the block unpack (num_blocks equal to the tile count is one tile per
call).
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    ReducePool,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test

BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
FP32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)


@skip_for_wormhole
@skip_for_quasar
@parametrize(
    formats_dest_acc=[
        (BF16, DestAccumulation.No),
        (BF16, DestAccumulation.Yes),
        (FP32, DestAccumulation.Yes),
    ],
    pool_type=[ReducePool.Max, ReducePool.Average],
    face_r_dim_num_faces=[(1, 2), (2, 2), (4, 2), (9, 2), (16, 2), (16, 4)],
    ct_dim_num_blocks=[(1, 1), (3, 1), (8, 1), (8, 2), (8, 8)],
)
def test_unpack_tilizeA_B(
    formats_dest_acc, pool_type, face_r_dim_num_faces, ct_dim_num_blocks
):
    formats, dest_acc = formats_dest_acc
    face_r_dim, num_faces = face_r_dim_num_faces
    ct_dim, num_blocks = ct_dim_num_blocks
    assert ct_dim % num_blocks == 0, "every block call takes the same number of tiles"
    if formats == FP32 and (ct_dim, num_blocks) not in ((3, 1), (8, 2)):
        pytest.skip("Float32 input: two block shapes are enough")

    torch_format = format_dict[formats.input_format]
    rows = face_r_dim * (2 if num_faces == 4 else 1)
    cols = ct_dim * 32
    tile_size = num_faces * face_r_dim * 16

    torch.manual_seed(face_r_dim * 1000 + num_faces * 100 + ct_dim * 10 + num_blocks)
    scale = 4.0 if pool_type == ReducePool.Max else 1.0
    src_A = ((torch.rand(rows, cols) * 2 - 1) * scale).to(torch_format)
    scaler = 1.0 if pool_type == ReducePool.Max else 1.0 / rows
    src_B = torch.full((tile_size,), scaler).to(torch_format)

    if pool_type == ReducePool.Max:
        reduced = src_A.to(torch.float32).max(dim=0).values
    else:
        reduced = (src_A.to(torch.float32) * src_B[0].to(torch.float32)).sum(dim=0)
    golden_tensor = reduced.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/unpack_tilizeA_B_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.ReduceColumn, pool_type=pool_type),
            MATH_FIDELITY(MathFidelity.HiFi4),
        ],
        runtimes=[
            generate_input_dim((32, cols), (32, cols)),
            TILE_COUNT(ct_dim),
            NUM_BLOCKS(num_blocks),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=16),
        ],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=ct_dim,
            tile_count_B=1,
            tile_count_res=ct_dim,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=[rows, 32],
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    # each output tile holds row 0 of its faces, face 0 then face 1
    res_tile_size = len(res_from_L1) // ct_dim
    res_rows = []
    for tile in range(ct_dim):
        res_rows.extend(res_from_L1[tile * res_tile_size : tile * res_tile_size + 32])
    res_tensor = torch.tensor(res_rows, dtype=format_dict[formats.output_format])

    assert len(res_tensor) == len(golden_tensor)
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
