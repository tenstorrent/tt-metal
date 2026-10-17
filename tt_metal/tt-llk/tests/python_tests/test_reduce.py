# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from helpers.format_config import DataFormat, InputOutputFormat, is_dest_acc_needed
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
from helpers.stimuli_generator import generate_stimuli
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
    REDUCE_TO_ONE,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test, tolerances

# Helper dictionary to map reduce dimensions to math operations
mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


def _fidelities_for_format(formats):
    """LoFi fails for Float16_b/Float32 inputs — exclude at parametrize time."""
    if formats.input_format in [DataFormat.Float16_b, DataFormat.Float32]:
        return [MathFidelity.HiFi2, MathFidelity.HiFi3, MathFidelity.HiFi4]
    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


def _reduce_to_one_for_format(formats):
    """Bfp8_b accumulates error easily with reduce_to_one (#45143) — exclude at parametrize time."""
    if (
        formats.input_format == DataFormat.Bfp8_b
        or formats.output_format == DataFormat.Bfp8_b
    ):
        return [False]
    return [False, True]


@parametrize(
    tile_dimensions=[
        [1, 32],
        [2, 32],
        [4, 32],
        [8, 32],
        [16, 16],
        [16, 32],
        [32, 32],
        [32, 16],
    ],
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Average, ReducePool.Sum],
    math_fidelity=_fidelities_for_format,
    is_reduce_to_one=_reduce_to_one_for_format,
)
def test_reduce(
    formats,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    tile_dimensions,
):
    tile_shape = construct_tile_shape(tile_dimensions)

    if is_reduce_to_one:
        # Accumulating large tensors into a single value can lead to significant numerical errors
        # especially if the sum/average of the dimension is not enough to increment to the next representable value in the output format.
        # To mitigate this, we use smaller input values which reduces the chance of large accumulation errors.
        input_dimensions = [128, 32]
    else:
        # If not reducing to one, we can use larger input dimensions to better test the reduction operation without excessive numerical errors in the accumulation.
        input_dimensions = [256, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )

    if pool_type in [
        ReducePool.Max,
        ReducePool.Sum,
    ]:  # result in srcA should be divided by 1
        src_B = torch.full((tile_shape.total_tile_size(),), 1)
    else:
        # reduce average divides by length of elements in array we reduce
        if reduce_dim == ReduceDimension.Row:
            src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[1])
        elif reduce_dim == ReduceDimension.Column:
            src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[0])
        else:  # Scalar
            src_B = torch.full(
                (tile_shape.total_tile_size(),),
                1 / math.sqrt(tile_dimensions[0] * tile_dimensions[1]),
            )

    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(
        src_A,
        reduce_dim,
        pool_type,
        formats.output_format,
        tile_cnt_A,
        reduce_to_one=is_reduce_to_one,
        tile_shape=tile_shape,
        input_format=formats.input_format,
    )

    # Float32 golden uses full FP32 accumulation; match that in HW whenever we
    # pack Float32 from sub-32-bit float inputs (e.g. Float16_b), otherwise
    # HiFi column/reduce-to-one cases can drift below PCC thresholds.
    dest_acc = (
        DestAccumulation.Yes
        if (
            formats.input_format.is_32_bit()
            or is_dest_acc_needed(formats)
            or (
                formats.output_format == DataFormat.Float32
                and not formats.input_format.is_32_bit()
            )
        )
        else DestAccumulation.No
    )

    output_tile_count = 1 if is_reduce_to_one else tile_cnt_A

    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/reduce_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop_mapping[reduce_dim], pool_type=pool_type),
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(output_tile_count),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            REDUCE_TO_ONE(is_reduce_to_one),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=output_tile_count,
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

    if is_reduce_to_one:
        # Lower the threshold for reduce to one cases as they are more prone to numerical errors, especially in lower precision formats
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            tile_shape=tile_shape,
            custom_pcc_threshold=(
                0.90
                if formats.output_format is not DataFormat.Bfp8_b
                else pow(0.99, tile_cnt_A)
            ),
            custom_atol=tolerances[formats.output_format].atol * tile_cnt_A,
            custom_rtol=tolerances[formats.output_format].rtol * tile_cnt_A,
        ), "Assert against golden failed"
    else:
        # Use default target_pcc = 0.99
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            tile_shape=tile_shape,
        ), "Assert against golden failed"


@parametrize(
    formats=[
        fmt
        for fmt in input_output_formats(
            [
                DataFormat.Bfp4_b,
                DataFormat.Float16_b,
                DataFormat.Bfp8_b,
                DataFormat.Float32,
            ]
        )
        if fmt.input_format == DataFormat.Bfp4_b
        or fmt.output_format == DataFormat.Bfp4_b
    ],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Average, ReducePool.Sum],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    is_reduce_to_one=[False, True],
    tile_dimensions=[[32, 32]],
)
def test_reduce_bfp4_b(
    formats,
    reduce_dim,
    pool_type,
    math_fidelity,
    is_reduce_to_one,
    tile_dimensions,
):

    tile_shape = construct_tile_shape(tile_dimensions)

    if is_reduce_to_one:
        # Accumulating large tensors into a single value can lead to significant numerical errors
        # especially if the sum/average of the dimension is not enough to increment to the next representable value in the output format.
        # To mitigate this, we use smaller input values which reduces the chance of large accumulation errors.
        input_dimensions = [128, 32]
    else:
        # If not reducing to one, we can use larger input dimensions to better test the reduction operation without excessive numerical errors in the accumulation.
        input_dimensions = [256, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )

    if pool_type in [
        ReducePool.Max,
        ReducePool.Sum,
    ]:  # result in srcA should be divided by 1
        src_B = torch.full((tile_shape.total_tile_size(),), 1)
    else:
        # reduce average divides by length of elements in array we reduce
        if reduce_dim == ReduceDimension.Row:
            src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[1])
        elif reduce_dim == ReduceDimension.Column:
            src_B = torch.full((tile_shape.total_tile_size(),), 1 / tile_dimensions[0])
        else:  # Scalar
            src_B = torch.full(
                (tile_shape.total_tile_size(),),
                1 / math.sqrt(tile_dimensions[0] * tile_dimensions[1]),
            )

    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(
        src_A,
        reduce_dim,
        pool_type,
        formats.output_format,
        tile_cnt_A,
        reduce_to_one=is_reduce_to_one,
        tile_shape=tile_shape,
        input_format=formats.input_format,
    )

    # Float32 golden uses full FP32 accumulation; match that in HW whenever we
    # pack Float32 from sub-32-bit float inputs (e.g. Float16_b), otherwise
    # HiFi column/reduce-to-one cases can drift below PCC thresholds.
    dest_acc = (
        DestAccumulation.Yes
        if (
            formats.input_format.is_32_bit()
            or is_dest_acc_needed(formats)
            or (
                formats.output_format == DataFormat.Float32
                and not formats.input_format.is_32_bit()
            )
        )
        else DestAccumulation.No
    )

    output_tile_count = 1 if is_reduce_to_one else tile_cnt_A

    _, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/reduce_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop_mapping[reduce_dim], pool_type=pool_type),
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(output_tile_count),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            REDUCE_TO_ONE(is_reduce_to_one),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=output_tile_count,
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

    if is_reduce_to_one:
        # Lower the threshold for reduce to one cases as they are more prone to numerical errors, especially in lower precision formats
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            tile_shape=tile_shape,
            custom_pcc_threshold=(
                0.90
                if formats.output_format is not DataFormat.Bfp8_b
                else pow(0.99, tile_cnt_A)
            ),
            custom_atol=tolerances[formats.output_format].atol * tile_cnt_A,
            custom_rtol=tolerances[formats.output_format].rtol * tile_cnt_A,
        ), "Assert against golden failed"
    else:
        # Use default target_pcc = 0.99
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            tile_shape=tile_shape,
        ), "Assert against golden failed"


# ---------------------------------------------------------------------------
# REDUCE_SCALAR SUM/AVG with fp32 DEST accumulation must keep fp32 precision.
#
# The face pooling leaves 16 per-column partial sums in DEST as fp32 datums. The
# pre-fix, FPU-only tail then moved that row through SrcB (MOVD2B, TRNSPSRCB)
# and SrcA (MOVB2A) for the final GAPOOL; the Src registers hold 19-bit datums,
# so every partial was truncated to TF32 mid-reduction and the total came out
# low by the sum of those truncations. The default harness tolerance (5%) is far
# too loose to see a 2^-10 relative error, so this case builds inputs whose
# per-column sums need more mantissa bits than a Src datum carries, computes the
# exact total, and gates on a couple of fp32 ulps.
#
# Inputs are 1.0 + k * 2^-INPUT_MANTISSA_BITS: exact in every Src format and in
# every fidelity phase (SrcA contributes 4 + 5 mantissa bits over two phases), so
# the face pooling into fp32 DEST is exact and the only inexact step of the
# FPU-only tail is the Src round trip of the partial sums. The scaler row is
# either uniform (1.0 for SUM, an exact power of two for AVG on square tiles) or,
# for SUM, a per-column weight pattern that is also exact; the final GAPOOL is
# then exact too and the device total must equal the fp64 total bit for bit.
# With four tiles reduced to one the sum is still exact in fp32, which checks
# the accumulation carried across tiles at fp32 precision. Two more input
# patterns (SUM, uniform scaler): "sparse" puts 1, 2^-8 and 2^-16 in the same
# in-face column of three different faces (one value per face: a single 16-row
# face pool cannot carry terms 16 binades apart, the fp32 DEST accumulation
# across faces can), so that partial needs all three bf16-exact pieces and
# losing the low one (2^-16) exceeds the gate; "overflow" puts two -2^127
# values in one column so the face pooling overflows fp32 and the result must
# be -inf, as before.
# ---------------------------------------------------------------------------

FP32_EXACT_TILES = [[32, 32], [16, 32], [16, 16]]
INPUT_MANTISSA_BITS = 7
FACE_C_DIM = 16


def _fp32_exact_tiles_for_pool(pool_type):
    """AVG's scaler is 1/sqrt(rows*cols); keep it an exact power of two."""
    if pool_type == ReducePool.Average:
        return [t for t in FP32_EXACT_TILES if t[0] == t[1]]
    return FP32_EXACT_TILES


def _fp32_exact_scaler_rows_for_pool(pool_type):
    """The per-column pattern only makes sense where the test owns the scaler value (SUM)."""
    if pool_type == ReducePool.Average:
        return ["uniform"]
    return ["uniform", "per_column"]


def _fp32_exact_patterns(formats, pool_type, scaler_row):
    """The sparse, overflow and tiny patterns are SUM-only with a uniform scaler row; the
    subnormal-residual case (its partial needs 9 significant bits) is Float32 input only.
    """
    if pool_type == ReducePool.Sum and scaler_row == "uniform":
        patterns = ["random", "sparse", "overflow", "tiny_random"]
        if formats.input_format == DataFormat.Float32:
            patterns.append("subnormal_residual")
        return patterns
    return ["random"]


def _fp32_exact_tiles_for_pool_and_pattern(pool_type, pattern):
    """AVG keeps the scaler an exact power of two; the sparse pattern needs three faces."""
    tiles = _fp32_exact_tiles_for_pool(pool_type)
    if pattern == "sparse":
        return [t for t in tiles if construct_tile_shape(t).total_num_faces() >= 3]
    return tiles


def _truncate_significand(value: float, explicit_mantissa_bits: int) -> float:
    """Truncate a float toward zero to 1 + explicit_mantissa_bits significant bits."""
    if value == 0.0:
        return 0.0
    mantissa, exponent = math.frexp(
        abs(value)
    )  # abs(value) = mantissa * 2**exponent, mantissa in [0.5, 1)
    scale = 2.0 ** (explicit_mantissa_bits + 1)
    truncated = math.floor(mantissa * scale) / scale
    return math.copysign(math.ldexp(truncated, exponent), value)


FP32_MANTISSA_BITS = 23
FP32_EXACT_GATE_ULPS = 2
TINY_INPUT_EXPONENT = -110


@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
    ],
    pool_type=[ReducePool.Sum, ReducePool.Average],
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    num_tiles=[1, 4],
    scaler_row=_fp32_exact_scaler_rows_for_pool,
    pattern=_fp32_exact_patterns,
    tile_dimensions=_fp32_exact_tiles_for_pool_and_pattern,
)
def test_reduce_scalar_fp32_exact(
    formats, pool_type, math_fidelity, num_tiles, scaler_row, pattern, tile_dimensions
):
    # Every input value and scaler below is bf16-exact, so Float16_b input (the common
    # fp32-DEST case in tt-metal, where SrcA holds BF16 datums) shares the fp64 golden.
    tile_shape = construct_tile_shape(tile_dimensions)
    elements_per_tile = tile_shape.total_tile_size()
    num_elements = num_tiles * elements_per_tile
    is_reduce_to_one = num_tiles > 1

    # Deterministic, unstructured k so the per-column sums carry non-zero low
    # mantissa bits (a structured pattern can make every column sum TF32-exact).
    generator = torch.Generator().manual_seed(0x5EED)
    k = torch.randint(0, 2**INPUT_MANTISSA_BITS, (num_elements,), generator=generator)
    src_A = (1.0 + k.to(torch.float64) * 2.0**-INPUT_MANTISSA_BITS).to(torch.float32)
    if pattern == "tiny_random":
        # Same values near the bottom of the fp32 range: the partials sit far below
        # 2^-63, where a split would produce subnormal pieces.
        src_A = (src_A.to(torch.float64) * 2.0**TINY_INPUT_EXPONENT).to(torch.float32)
    elif pattern == "subnormal_residual":
        # One value (17/16) * 2^-126 in row 0, column 3 of face 0; with w[0] = 17/16 the
        # partial is (289/256) * 2^-126, normal, but its bits below the bf16 piece sit
        # under 2^-126 (the reviewer case).
        src_A = torch.zeros(num_elements, dtype=torch.float32)
        for tile in range(num_tiles):
            src_A[tile * elements_per_tile + 3] = (17.0 / 16.0) * 2.0**-126
    elif pattern != "random":
        # In-face column 3 carries the pattern, everything else is 0: "sparse" puts one value in row 0 of
        # faces 0, 1 and 2 (they pool into the same partial), "overflow" two values in rows 0 and 1 of face 0.
        src_A = torch.zeros(num_elements, dtype=torch.float32)
        face_elements = tile_shape.face_r_dim * FACE_C_DIM
        if pattern == "sparse":
            placements = [(face, 0, 2.0 ** (-8 * face)) for face in range(3)]
        else:
            placements = [(0, 0, -(2.0**127)), (0, 1, -(2.0**127))]
        for tile in range(num_tiles):
            for face, row, value in placements:
                src_A[
                    tile * elements_per_tile
                    + face * face_elements
                    + row * FACE_C_DIM
                    + 3
                ] = value

    # Scaler row w[c] for the 16 columns of a face. The kernel applies it twice:
    # the face pooling weights face row r by w[r] and the final pool weights the
    # per-column partial sums by w[c]. The scaler tile holds w[c] in every row of
    # every face (c = L1 index mod 16), so each face sees the same row.
    if pattern == "subnormal_residual":
        weights = [1.0] * FACE_C_DIM
        weights[0] = 17.0 / 16.0
        weights[3] = 256.0
    elif scaler_row == "per_column":
        weights = [1.0 + (c % 2) * 0.5 for c in range(FACE_C_DIM)]
    elif pool_type == ReducePool.Average:
        scaler = 1.0 / math.sqrt(tile_dimensions[0] * tile_dimensions[1])
        assert math.frexp(scaler)[0] == 0.5, "scaler must be an exact power of two"
        weights = [scaler] * FACE_C_DIM
    else:
        weights = [1.0] * FACE_C_DIM
    src_B = torch.tensor(
        [weights[i % FACE_C_DIM] for i in range(elements_per_tile)], dtype=torch.float32
    )

    # Exact reference: sum over every element of w[row] * w[col] * A, in fp64.
    values = src_A.to(torch.float64)
    index = torch.arange(num_elements)
    face_row = (index % (tile_shape.face_r_dim * FACE_C_DIM)) // FACE_C_DIM
    face_col = index % FACE_C_DIM
    w = torch.tensor(weights, dtype=torch.float64)
    weighted = values * w[face_row] * w[face_col]
    golden = float(weighted.sum().item())
    if pattern == "overflow":
        golden = (
            -math.inf
        )  # the fp32 face pooling overflows; the pre-fix tail carried that through

    # What the Src round trip of the pre-fix tail does, for the assertion message
    # only: per tile, each in-face column is pooled into one fp32 partial, every
    # partial is then truncated to a Src datum (TF32 keeps 10 explicit mantissa
    # bits; the multiplier ignores the last of them, so 9 are effective) before
    # the final GAPOOL weights and adds the 16 partials.
    row_weighted = (values * w[face_row]).reshape(num_tiles, -1)
    partials = [
        [float(row_weighted[t, c::FACE_C_DIM].sum().item()) for c in range(FACE_C_DIM)]
        for t in range(num_tiles)
    ]
    modelled = {
        bits: sum(
            weights[c] * _truncate_significand(partials[t][c], bits)
            for t in range(num_tiles)
            for c in range(FACE_C_DIM)
        )
        for bits in (10, 9, 7)
    }

    configuration = TestConfig(
        "sources/reduce_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.ReduceScalar, pool_type=pool_type),
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            INPUT_TILE_CNT(num_tiles),
            OUTPUT_TILE_CNT(1),
            NUM_TILES_IN_BLOCK(1),
            REDUCE_TO_ONE(is_reduce_to_one),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=num_tiles,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=DestAccumulation.Yes,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == elements_per_tile, "Result tensor has the wrong length"

    device = float(res_from_L1[0])
    if pattern == "overflow":
        assert device == golden, (
            f"REDUCE_SCALAR {pool_type.name} fp32 overflow result changed: device={device!r} "
            f"expected {golden!r} (tile={tile_dimensions}, num_tiles={num_tiles}, fidelity={math_fidelity.name})"
        )
        return
    error = device - golden
    if pattern == "tiny_random":
        # Below 2^-63 the fp32 tail keeps the partials whole, i.e. the pre-fix precision
        # (TF32 for Float32 input, BF16 for Float16_b input): no worse than that model.
        pre_fix_bits = 9 if formats.input_format == DataFormat.Float32 else 7
        allowed = abs(
            modelled[pre_fix_bits] - golden
        ) + FP32_EXACT_GATE_ULPS * math.ldexp(
            1.0, math.frexp(abs(golden))[1] - 1 - FP32_MANTISSA_BITS
        )
        assert abs(error) <= allowed, (
            f"REDUCE_SCALAR {pool_type.name} tiny partials worse than the pre-fix tail: "
            f"device={device!r} golden={golden!r} pre-fix model={modelled[pre_fix_bits]!r} "
            f"(tile={tile_dimensions}, num_tiles={num_tiles}, fidelity={math_fidelity.name})"
        )
        return
    # fp32 ulp of the golden: golden = m * 2**e with m in [0.5, 1), so its ulp is 2**(e - 1 - 23).
    tolerance = FP32_EXACT_GATE_ULPS * math.ldexp(
        1.0, math.frexp(abs(golden))[1] - 1 - FP32_MANTISSA_BITS
    )
    assert abs(error) <= tolerance, (
        f"REDUCE_SCALAR {pool_type.name} fp32 result lost precision: device={device!r} "
        f"golden={golden!r} error={error:+.6g} (allowed +-{tolerance:.3g}); "
        f"Src round-trip model of the pre-fix FPU-only tail (partials truncated to 9/10/7 "
        f"explicit mantissa bits): {modelled[9]!r} / {modelled[10]!r} / {modelled[7]!r} "
        f"(tile={tile_dimensions}, num_tiles={num_tiles}, scaler_row={scaler_row}, pattern={pattern}, "
        f"fidelity={math_fidelity.name})"
    )
