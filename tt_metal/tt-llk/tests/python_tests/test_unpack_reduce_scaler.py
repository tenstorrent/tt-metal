# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single-operand reduce path (_llk_unpack_reduce_init_ / _llk_unpack_reduce_): the operand
tile goes to SrcA and the scaler tile, placed in its own L1 buffer, to SrcB. The math and
pack threads are the ones test_reduce.py uses, so a mismatch against the golden means the
scaler the unpacker delivered is not the one in the buffer.

REDUCE_ROW SUM/AVG is not a variant: _llk_unpack_reduce_init_ rejects it at compile time
(the math LLK runs that case with the operands swapped, which this unpack path does not
produce). REDUCE_ROW MAX pools SrcA only and is covered.

Scaler contract: the path unpacks only the first FACE_C_DIM datums of face 0 of the scaler
tile (its first row) into SrcB and reuses them for every face. The ``face0_row0`` fill puts
the scaler in that row only and 7x the scaler everywhere else; the golden uses the scaler for
all faces, so the variant passes under the documented contract and would fail if the path
stepped through the scaler tile face by face (as the AB reduce does). It is run for the
COL/SCALAR SUM/AVG variants only: MAX pools SrcA alone, so for MAX it would repeat ``uniform``.

Scaler format: ``scaler_format`` runs each data format with a scaler tile in the same format
and in the other member of its exponent family (Float32 data with a Float16_b scaler, the
usual tt-metal setup, and Float16_b data with a Float32 scaler). The SrcB unpacker formats
come from the scaler operand and the math SrcB format follows the scaler register format.
With the former derivation (SrcB L1 format fixed to Float32, register format taken from the
data operand) only the ``face0_row0`` SUM/AVG variants with a Float16_b scaler fail: a
uniform Float16_b tile read as Float32 still decodes to about the scaler value (every word is
the same bf16 pattern twice), so the ``uniform`` mixed-format variants are coverage of the
format plumbing rather than a fail/pass proof of it.
"""

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli, reduce_scaler_value
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    IN_FACE_DIMS,
    INPUT_TILE_CNT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
)
from helpers.tile_constants import FACE_C_DIM
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

pytestmark = [skip_for_quasar]

mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


def _pool_types_for_dim(reduce_dim):
    if reduce_dim == ReduceDimension.Row:
        return [ReducePool.Max]
    return [ReducePool.Max, ReducePool.Average, ReducePool.Sum]


def _scaler_fills_for_pool(pool_type):
    if pool_type == ReducePool.Max:
        return ["uniform"]
    return ["uniform", "face0_row0"]


def _scaler_formats_for(formats):
    other = (
        DataFormat.Float16_b
        if formats.input_format == DataFormat.Float32
        else DataFormat.Float32
    )
    return [formats.input_format, other]


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    scaler_format=_scaler_formats_for,
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=_pool_types_for_dim,
    scaler_fill=_scaler_fills_for_pool,
)
def test_unpack_reduce_scaler(
    formats, scaler_format, reduce_dim, pool_type, scaler_fill
):
    tile_dimensions = [32, 32]
    tile_shape = construct_tile_shape(tile_dimensions)
    input_dimensions = [64, 32]

    src_A, tile_cnt_A, _, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=scaler_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )

    scaler = reduce_scaler_value(pool_type, reduce_dim, tile_dimensions)
    src_B = torch.full((tile_shape.total_tile_size(),), scaler)
    if scaler_fill == "face0_row0":
        # Only face 0, row 0 carries the scaler; every other datum would corrupt the result if read.
        src_B = torch.full((tile_shape.total_tile_size(),), 7 * scaler)
        src_B[:FACE_C_DIM] = scaler

    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(
        src_A,
        reduce_dim,
        pool_type,
        formats.output_format,
        tile_cnt_A,
        reduce_to_one=False,
        tile_shape=tile_shape,
        input_format=formats.input_format,
    )

    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit() or scaler_format.is_32_bit()
        else DestAccumulation.No
    )
    test_formats = InputOutputFormat(
        formats.input_format, formats.output_format, input_format_B=scaler_format
    )

    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/unpack_reduce_scaler_test.cpp",
        test_formats,
        templates=[
            MATH_OP(mathop=mathop_mapping[reduce_dim], pool_type=pool_type),
            MATH_FIDELITY(MathFidelity.HiFi4),
        ],
        runtimes=[
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(tile_cnt_A),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            scaler_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"
