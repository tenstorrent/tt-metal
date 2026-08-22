# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SFPU reduce (column/row SUM/AVG/MAX/MIN) on Quasar.

Data layout and dimension sweep match ``test_sfpu_reduce.py``: L1 is tilized
(face layout), goldens are untilized, column reduce writes dest row 0, row
reduce writes dest column 0. The C++ harness is the Quasar unpack-to-dest +
RC_custom SFPU path.
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    ReducePool,
    UnpackerEngine,
    format_dict,
)
from helpers.logger import logger
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

max_tiles = 4

INT32_MAX = torch.iinfo(torch.int32).max
INT32_MIN = torch.iinfo(torch.int32).min
INT32_PAD_MIN = INT32_MIN + 1
UINT16_MAX = torch.iinfo(torch.uint16).max

dimension_combinations = [
    [m, n]
    for m in range(TILE_DIM, max_tiles * TILE_DIM + 1, TILE_DIM)
    for n in range(TILE_DIM, max_tiles * TILE_DIM + 1, TILE_DIM)
    if m * n <= max_tiles * TILE_DIM * TILE_DIM
]


def get_format_input_bounds(
    formats: InputOutputFormat, reduce_pool: ReducePool
) -> list[tuple[int, int]]:
    """Get valid stimuli bounds based on data format.

    UInt16 Sum folds up to ``max_tiles * TILE_DIM`` terms into a 16-bit result.
    Cap the exclusive upper bound so that product cannot overflow UInt16; Average
    and Max/Min stay at the unsigned (0, 1000) window.
    """
    if formats.input_format == DataFormat.UInt16:
        if reduce_pool == ReducePool.Sum:
            max_terms = max_tiles * TILE_DIM
            return [(0, UINT16_MAX // max_terms)]
        return [(0, 1000)]
    return [(-1000, 1000), (0, 1000), (-1000, 0)]


def get_supported_reduce_axioms(
    reduce_pool: ReducePool, formats: InputOutputFormat
) -> list[MathOperation]:
    # Row reduce supports SUM/MAX/MIN for every format and AVG for float formats only
    # (the row AVG divisor is the runtime column count, which only the float
    # reciprocal-multiply divides exactly; integer AVG stays column-only).
    if reduce_pool in (ReducePool.Sum, ReducePool.Max, ReducePool.Min):
        return [MathOperation.ReduceRow, MathOperation.ReduceColumn]
    if reduce_pool == ReducePool.Average and formats.input_format in (
        DataFormat.Float32,
        DataFormat.Float16_b,
    ):
        return [MathOperation.ReduceRow, MathOperation.ReduceColumn]
    return [MathOperation.ReduceColumn]


def use_int32_twos_complement(
    formats: InputOutputFormat, reduce_pool: ReducePool, mathop: MathOperation
) -> bool:
    """Whether Int32 stimuli/results use two's-complement L1 encoding.

    Quasar unpack-to-dest Int32 is two's-complement in Dest. SUM uses SFPIADD,
    MAX/MIN compare with SFPSWAP imm12=0 (int32 2's-complement), and column AVG
    does a toward-zero shift of that two's-complement word — so every Int32
    reduce path takes two's-complement operands.
    """
    del reduce_pool, mathop
    return formats.input_format == DataFormat.Int32


def get_reduce_pad_value(reduce_pool: ReducePool, input_format: DataFormat):
    """Identity fill for the padded (non-data) rows of a sub-tile column reduce."""
    if reduce_pool == ReducePool.Max:
        if input_format == DataFormat.Int32:
            return INT32_PAD_MIN
        if input_format.is_integer():
            return 0
        return -3.0e30
    if reduce_pool == ReducePool.Min:
        if input_format == DataFormat.Int32:
            return INT32_MAX
        if input_format == DataFormat.UInt16:
            return UINT16_MAX
        if input_format.is_integer():
            return INT32_MAX
        return 3.0e30
    return 0


def get_reduce_extents(
    mathop: MathOperation,
    reduce_pool: ReducePool,
    formats: InputOutputFormat,
    dimension_combinations: list[int],
) -> list[int]:
    """Number of real (unpadded) rows on the column-reduce axis."""
    full = [TILE_DIM]
    if (
        mathop != MathOperation.ReduceColumn
        or dimension_combinations != [TILE_DIM, TILE_DIM]
        or reduce_pool == ReducePool.Average
    ):
        return full
    if formats.input_format == DataFormat.Int32:
        return [1, 13, 15, 16, 17, 30, 31, TILE_DIM]
    return [15, TILE_DIM]


REDUCE_BASE_FORMATS = [
    DataFormat.Float32,
    DataFormat.Int32,
    DataFormat.UInt16,
    DataFormat.Float16_b,
]


def get_reduce_formats() -> list[InputOutputFormat]:
    """Input/output format pairs for the reduce suite.

    Every pool keeps input == output, including UInt16 Sum/Average. Overflow on
    those pools is avoided by shrinking the UInt16 Sum stimuli range in
    ``get_format_input_bounds`` rather than widening the pack format.
    """
    return [InputOutputFormat(fmt, fmt) for fmt in REDUCE_BASE_FORMATS]


_FLOAT_FORMAT_EPS = {
    DataFormat.Float16_b: 2.0**-8,
    DataFormat.Float16: 2.0**-11,
    DataFormat.Float32: 2.0**-24,
}


def get_reduce_sum_atol(
    output_format, reduce_pool, mathop, input_dimensions, input_bounds
):
    """Absolute tolerance for accumulating float reductions (Sum/Average)."""
    if reduce_pool not in (ReducePool.Sum, ReducePool.Average):
        return None

    eps = _FLOAT_FORMAT_EPS.get(output_format)
    if eps is None:
        return None

    max_term = max(abs(input_bounds[0]), abs(input_bounds[1]))
    num_terms = input_dimensions[1] if mathop == MathOperation.ReduceRow else TILE_DIM
    atol = 2.0 * max_term * eps * (num_terms**0.5)
    if reduce_pool == ReducePool.Average:
        atol /= num_terms
    return max(0.05, atol)


def is_valid_reduce_dimension(mathop, dest_acc, formats, dim):
    """Check if a dimension is valid for the given reduce operation."""
    try:
        num_blocks, _ = get_num_blocks_and_num_tiles_in_block(
            DestSync.Half,
            dest_acc,
            formats,
            dim,
            TILE_DIMENSIONS,
            BlocksCalculationAlgorithm.Standard,
        )
        if mathop == MathOperation.ReduceColumn:
            return True
        return num_blocks == 1  # ReduceRow needs the full matrix in one dest block
    except ValueError:
        return False


def _quasar_test_config(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_dimensions,
    tile_cnt,
    src_A,
    src_B,
):
    """Quasar unpack-to-dest + SFPU reduce harness. Layout of src_A is tilized."""
    num_faces = 4
    return TestConfig(
        "sources/quasar/sfpu_reduce_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
            twos_complement=use_int32_twos_complement(formats, reduce_pool, mathop),
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )


@pytest.mark.quasar
@parametrize(
    formats=get_reduce_formats,
    mathop=get_supported_reduce_axioms,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_bounds=get_format_input_bounds,
    reduce_pool=[ReducePool.Min, ReducePool.Max, ReducePool.Sum, ReducePool.Average],
    dimension_combinations=lambda mathop, dest_acc, formats: [
        dim
        for dim in dimension_combinations
        if is_valid_reduce_dimension(mathop, dest_acc, formats, dim)
    ],
    reduced_extent=get_reduce_extents,
)
def test_sfpu_reduce_quasar(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_bounds,
    dimension_combinations,
    reduced_extent,
):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip(
            reason="32-bit formats require DestAccumulation.Yes (HW cannot unpack into SrcA/SrcB)"
        )

    min_value, max_value = input_bounds
    input_dimensions = dimension_combinations
    torch_format = format_dict[formats.input_format]

    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    if formats.input_format.is_integer():
        src_A = torch.randint(
            low=min_value,
            high=max_value,
            size=stimuli_size,
            dtype=torch_format,
        )
    else:
        src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(
            min_value, max_value
        )
    src_B = torch.zeros_like(src_A)

    if mathop == MathOperation.ReduceColumn and reduced_extent < TILE_DIM:
        pad_value = get_reduce_pad_value(reduce_pool, formats.input_format)
        src_A = src_A.view(TILE_DIM, tile_cnt * TILE_DIM)
        src_A[reduced_extent:, :] = pad_value
        src_A = src_A.flatten()

    dst_dim = (
        [32, tile_cnt * 32]
        if mathop == MathOperation.ReduceColumn
        else input_dimensions
    )

    src_A = tilize_block(src_A, dst_dim, stimuli_format=formats.input_format).flatten()
    src_A_untilized = untilize_block(src_A, formats.input_format, dst_dim)

    golden_input = src_A_untilized
    if mathop == MathOperation.ReduceColumn and reduced_extent < TILE_DIM:
        golden_input = src_A_untilized[:reduced_extent]

    golden_tensor = get_golden_generator(UnarySFPUGolden)(
        mathop,
        golden_input,
        formats.output_format,
        dest_acc,
        formats.input_format,
        dst_dim,
        reduce_pool=reduce_pool,
    )

    configuration = _quasar_test_config(
        formats,
        dest_acc,
        mathop,
        reduce_pool,
        input_dimensions,
        tile_cnt,
        src_A,
        src_B,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, dst_dim)

    if mathop == MathOperation.ReduceColumn:
        golden_slice = golden_tensor[0]
        res_slice = res_tensor[0]
    elif mathop == MathOperation.ReduceRow:
        golden_slice = golden_tensor[:, 0]
        res_slice = res_tensor[:, 0]
    else:
        raise ValueError(f"Unsupported math operation: {mathop}")

    reduce_atol = get_reduce_sum_atol(
        formats.output_format, reduce_pool, mathop, input_dimensions, input_bounds
    )

    assert passed_test(
        golden_slice, res_slice, formats.output_format, custom_atol=reduce_atol
    )


def _run_int32_reduce(mathop, reduce_pool, injected_value, base_range=(-1000, 1000)):
    """Build a single 32x32 Int32 tile, inject `injected_value` at a few scattered
    positions, run the SFPU reduce on device, and return (golden_slice, device_slice).
    """
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [TILE_DIM, TILE_DIM]
    torch_format = format_dict[formats.input_format]

    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    torch.manual_seed(0)
    src_A = torch.randint(
        low=base_range[0], high=base_range[1], size=stimuli_size, dtype=torch_format
    )

    grid = src_A.view(TILE_DIM, TILE_DIM)
    inject_positions = [(0, 0), (5, 7), (13, 3), (20, 20), (31, 31), (7, 15)]
    for r, c in inject_positions:
        grid[r, c] = injected_value
    src_A = grid.flatten()

    dst_dim = (
        [32, tile_cnt * 32]
        if mathop == MathOperation.ReduceColumn
        else input_dimensions
    )

    src_A = tilize_block(src_A, dst_dim, stimuli_format=formats.input_format).flatten()
    src_A_untilized = untilize_block(src_A, formats.input_format, dst_dim)

    golden_tensor = get_golden_generator(UnarySFPUGolden)(
        mathop,
        src_A_untilized,
        formats.output_format,
        dest_acc,
        formats.input_format,
        dst_dim,
        reduce_pool=reduce_pool,
    )

    src_B = torch.zeros_like(src_A)
    configuration = _quasar_test_config(
        formats,
        dest_acc,
        mathop,
        reduce_pool,
        input_dimensions,
        tile_cnt,
        src_A,
        src_B,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, dst_dim)

    if mathop == MathOperation.ReduceColumn:
        return golden_tensor[0], res_tensor[0]
    return golden_tensor[:, 0], res_tensor[:, 0]


@pytest.mark.quasar
@pytest.mark.parametrize(
    "mathop", [MathOperation.ReduceColumn, MathOperation.ReduceRow]
)
@pytest.mark.parametrize("reduce_pool", [ReducePool.Min, ReducePool.Max])
@pytest.mark.parametrize(
    "injected_value", [INT32_MIN, INT32_MAX], ids=["INT32_MIN", "INT32_MAX"]
)
@pytest.mark.parametrize(
    "base_range",
    [(-1000, 1000), (-1000, -1), (1, 1000)],
    ids=["mixed", "all_negative", "all_positive"],
)
def test_int32_reduce_extreme_quasar(mathop, reduce_pool, injected_value, base_range):
    """INT32 SFPU reduce (min/max) must stay correct when the input contains
    INT32_MIN or INT32_MAX. Quasar compares as two's-complement, so both extremes
    must win the lanes they are injected into.
    """
    golden_slice, res_slice = _run_int32_reduce(
        mathop, reduce_pool, injected_value, base_range=base_range
    )

    golden = golden_slice.to(torch.int64)
    res = res_slice.to(torch.int64)

    mismatch = golden != res
    num_mismatch = int(mismatch.sum().item())

    if num_mismatch:
        idxs = torch.nonzero(mismatch).flatten().tolist()
        lines = [
            f"  idx={i}: golden={int(golden[i])} device={int(res[i])}"
            for i in idxs[:12]
        ]
        detail = "\n".join(lines)
        logger.info(
            "\n{} {} injected={}: {} mismatched lanes\n{}",
            reduce_pool,
            mathop,
            int(injected_value),
            num_mismatch,
            detail,
        )

    assert num_mismatch == 0, (
        f"{num_mismatch} mismatched reduction lanes for {reduce_pool} {mathop} "
        f"injected={int(injected_value)} (see stdout)"
    )
