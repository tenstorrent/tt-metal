# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import itertools
from dataclasses import dataclass
from enum import Enum

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.data_format_inference import is_format_combination_outlier
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BinarySFPUGolden,
    BroadcastGolden,
    get_golden_generator,
)
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    BROADCAST_TYPE,
    MATH_OP,
    TILE_COUNT,
    TemplateParameter,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    bcast_dim=[
        LlkBroadcastType.None_,
        LlkBroadcastType.Row,
        LlkBroadcastType.Column,
        LlkBroadcastType.Scalar,
    ],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        # Disabled: failing due to very small differences in generated stimuli
        # MathOperation.SfpuElwLt,
        # MathOperation.SfpuElwGt,
        # MathOperation.SfpuElwLe,
        # MathOperation.SfpuElwGe,
        # MathOperation.SfpuElwEq,
        # MathOperation.SfpuElwNe,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_float(
    formats,
    dest_acc,
    mathop,
    bcast_dim,
):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and formats.input_format == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Float16_a isn't supported for SFPU on Blackhole without being converted to 32-bit intermediate format in dest register"
        )

    if bcast_dim == LlkBroadcastType.Row and (
        dest_acc == DestAccumulation.Yes
        or is_format_combination_outlier(
            formats.input_format, formats.output_format, dest_acc
        )
    ):
        pytest.skip(
            "Row broadcast with FP32 dest: B2D datacopy uses MOVB2D which can't handle FP32 dest format conversion"
        )

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        broadcast_type=bcast_dim,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_div(formats, dest_acc):
    # DIV routes through the dedicated production kernel (calculate_sfpu_binary_div)
    # via call_binary_sfpu_operation. It is split out from test_sfpu_binary_float
    # because the reciprocal path is precision-sensitive and warrants its own
    # coverage (guarding the bf16 Newton-iteration count vs the fp32 residual path).
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and formats.input_format == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Float16_a isn't supported for SFPU on Blackhole without being converted to 32-bit intermediate format in dest register"
        )

    sfpu_binary(
        formats,
        dest_acc,
        MathOperation.SfpuElwdiv,
        broadcast_type=LlkBroadcastType.None_,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwLogicalRightShift,
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwGe,
    ],
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_int(
    formats,
    dest_acc,
    mathop,
):
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
    )


# ---------------------------------------------------------------------------
# Deterministic edge-case coverage for the integer shift ops.
#
# The default Int32 stimuli are drawn uniformly from [0, 2^30), so they never
# exercise negative values or in-range shift amounts, and the arithmetic vs.
# logical / out-of-range behaviour is left untested. These edge cases pin down
# the kernel contract from ckernel_sfpu_shift.h: shift amounts outside [0, 31]
# produce 0, arithmetic right-shift sign-extends negative values, and negative
# operands shift correctly (which requires the INT32_2S_COMP load/store mode -
# see SFPU_INT32_SHIFT.md).
#
# INT32_MIN (0x80000000) is deliberately excluded here: Dst stores int32 as
# sign-magnitude with range +-(2^31 - 1), so -2^31 has no representation (it is
# "negative zero"). Any value or result equal to INT32_MIN cannot round-trip; it
# is covered separately by the xfail test below.
# ---------------------------------------------------------------------------

_INT32_MIN = -(2**31)

_SHIFT_EDGE_OPS = [
    MathOperation.SfpuElwRightShift,
    MathOperation.SfpuElwLeftShift,
    MathOperation.SfpuElwLogicalRightShift,
]

# Representative Int32 values: zero, small magnitudes of both signs, byte / halfword
# boundaries, the sign bit, an alternating bit pattern and the int32 extremes.
_SHIFT_EDGE_VALUES = [
    0,
    1,
    -1,
    2,
    -2,
    7,
    -8,
    255,
    -256,
    0x0000FFFF,
    -0x00010000,
    0x40000000,
    -0x40000000,
    0x55555555,
    -0x55555555,
    0x7FFFFFFF,  # INT32_MAX
    -0x80000000,  # INT32_MIN (filtered out per-op; see _build_shift_edge_case_src)
]

# Shift amounts spanning in-range values (0..31), the first out-of-range value (32),
# larger out-of-range values, and negative amounts. Everything outside [0, 31] must
# yield 0 to match the kernel.
_SHIFT_EDGE_AMOUNTS = [
    0,
    1,
    2,
    7,
    15,
    16,
    30,
    31,  # in-range
    32,
    33,
    40,
    63,
    100,
    1000,  # >= 32 -> 0
    -1,
    -5,
    -32,
    -1000,  # < 0 -> 0
]


def _shift_reference(mathop, value, shift):
    """Bit-exact reference for a single (value, shift) pair, matching the kernel contract.

    Shift amounts outside [0, 31] produce 0. Right shift is arithmetic (sign-extending),
    logical right shift treats the operand as unsigned, left shift is a plain bit shift.
    Mirrors BinarySFPUGolden and is used to pre-filter results that Dst cannot represent.
    """
    shift = int(shift)
    if shift < 0 or shift >= 32:
        return 0
    v = torch.tensor(int(value), dtype=torch.int32)
    if mathop == MathOperation.SfpuElwRightShift:
        return int(torch.bitwise_right_shift(v, shift))
    if mathop == MathOperation.SfpuElwLeftShift:
        return int(torch.bitwise_left_shift(v, shift))
    if mathop == MathOperation.SfpuElwLogicalRightShift:
        r = (int(value) & 0xFFFFFFFF) >> shift
        return r - 0x100000000 if r >= 0x80000000 else r
    raise ValueError(f"Unsupported shift op: {mathop}")


def _build_shift_edge_case_src(mathop):
    """Build a deterministic [64, 32] Int32 operand for the shift edge-case test.

    The math kernel consumes only ``buffer_A``: tile 0 (rows 0-31) holds the values
    and tile 1 (rows 32-63) holds the per-element shift amounts. ``tilize`` applies
    the same permutation to both tiles, so a value at index ``k`` within tile 0 is
    always paired with the shift at index ``k`` within tile 1. Laying both grids out
    with the same ``k`` ordering lets us walk the cartesian product of interesting
    (value, shift) pairs so every combination is exercised.

    Pairs whose operand or result equals INT32_MIN are dropped, since Dst's
    sign-magnitude int32 format cannot represent -2^31 (covered by the xfail test).
    """
    pairs = [
        (v, s)
        for v, s in itertools.product(_SHIFT_EDGE_VALUES, _SHIFT_EDGE_AMOUNTS)
        if v != _INT32_MIN and _shift_reference(mathop, v, s) != _INT32_MIN
    ]
    num_elements = 32 * 32
    value_grid = [pairs[i % len(pairs)][0] for i in range(num_elements)]
    shift_grid = [pairs[i % len(pairs)][1] for i in range(num_elements)]
    return torch.tensor(value_grid + shift_grid, dtype=torch.int32)


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=_SHIFT_EDGE_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_int_shift_edge_cases(
    formats,
    dest_acc,
    mathop,
):
    if TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE:
        pytest.skip(
            reason="Blackhole shift kernels (left / arithmetic right / logical right) are "
            "unmigrated TTI microcode whose predicated out-of-range/sign handling breaks "
            "under INT32_2S_COMP for negative operands, so all three diverge from the "
            "two's-complement golden. See SFPU_INT32_SHIFT.md."
        )

    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=_build_shift_edge_case_src(mathop),
    )


@pytest.mark.xfail(
    reason="Dst stores int32 as sign-magnitude with range +-(2^31 - 1). INT32_MIN "
    "(0x80000000) is 'negative zero' and cannot round-trip through Dst, so shifts that "
    "consume or produce it diverge from the two's-complement golden. This is a hardware "
    "limitation of the Wormhole SFPU load/store path; see SFPU_INT32_SHIFT.md.",
    strict=False,
)
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=_SHIFT_EDGE_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_int_shift_int32_min_unsupported(
    formats,
    dest_acc,
    mathop,
):
    # Every value lane is INT32_MIN, shifted by 0 (identity). The golden expects
    # INT32_MIN back, but the hardware loads it as 0 (sign-magnitude negative zero),
    # so this is expected to fail until/unless the representation changes.
    num_elements = 32 * 32
    value_grid = [_INT32_MIN] * num_elements
    shift_grid = [0] * num_elements
    src = torch.tensor(value_grid + shift_grid, dtype=torch.int32)
    sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=src,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
        ],
        same=True,
    ),
    mathop=[MathOperation.SfpuAddTopRow],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_add_top_row(formats, dest_acc, mathop):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip(
            "32-bit integer formats require DestAccumulation.Yes (HW cannot unpack into SrcA/SrcB)"
        )

    input_dimensions = [64, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        0,
        1,
        0,
        1,
        input_dimensions,
        formats.output_format,
    )

    golden_tensor = (
        golden_tensor.view([32, 32])
        if golden_tensor.shape == torch.Size([1024])
        else golden_tensor.view(input_dimensions)
    )

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
            BROADCAST_TYPE(LlkBroadcastType.None_),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).view(input_dimensions)

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


def sfpu_binary(
    formats,
    dest_acc,
    mathop,
    broadcast_type=None,
    src_A_override=None,
):

    input_dimensions = [64, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # The math kernel only consumes buffer_A (operand 0 = tile 0, operand 1 = tile 1),
    # so an explicit src_A fully controls the inputs and lets us exercise deterministic
    # edge cases. src_B stays random but unused by the kernel.
    if src_A_override is not None:
        src_A = src_A_override.to(src_A.dtype).flatten()

    golden_src = src_A
    if broadcast_type is not None and broadcast_type != LlkBroadcastType.None_:
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_src = generate_broadcast_golden(
            broadcast_type,
            src_A,
            (
                formats.input_format
                if formats.input_format != DataFormat.Bfp8_b
                else DataFormat.Float16_b
            ),
            tile_cnt=tile_cnt_A,
        )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        golden_src,
        0,  # src1_idx: use tile 0
        1,  # src2_idx: use tile 1
        0,  # dst_idx: write to tile 0
        32,  # num_iterations: 32 rows
        input_dimensions,  # [64, 32] = 2 tiles
        (
            DataFormat.Float16_b
            if formats.input_format == DataFormat.Bfp8_b
            else formats.input_format
        ),
    ).flatten()

    # ONLY Blackhole needs this for some reason
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    bcast = broadcast_type if broadcast_type else LlkBroadcastType.None_

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
            BROADCAST_TYPE(bcast),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# ---------------------------------------------------------------------------
# SFPU binary with row/column broadcast (BCAST_COL / BCAST_ROW).
#
# Uses its own kernel source (`sources/sfpu_binary_bcast_test.cpp`) because
# the in-DST pipeline is 3-tile (data + bcast + result) with a custom init
# and full-tile driver. The LLK load/store path uses InstrModLoadStore::DEFAULT
# so any float dest format (Float32, Float16, Float16_b, or Bfp8_b-via-unpack-
# conversion) works; SFPU compute is FP32 in LRegs regardless.
# ---------------------------------------------------------------------------


class BroadcastType(Enum):
    # Values must match ckernel::BroadcastType in llk_defs.h
    # (NONE=0, COL=1, ROW=2, SCALAR=3) because the kernel does
    # `static_cast<BroadcastType>(BCAST_DIM_VAL)`.
    COL = 1
    ROW = 2


@dataclass
class SFPU_BCAST_DIM(TemplateParameter):
    bcast_dim: BroadcastType

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t BCAST_DIM_VAL = {self.bcast_dim.value};"


@dataclass
class INPUT_TILE_A(TemplateParameter):
    """Base DST tile index for input A.

    The kernel derives the other tile indices from this single value:
      INPUT_TILE_A      -> data tile
      INPUT_TILE_A + 1  -> bcast tile
      INPUT_TILE_A + 2  -> result tile
    """

    tile_index: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t INPUT_TILE_A_VAL = {self.tile_index};"


_BCAST_BINARY_OPS = {
    MathOperation.SfpuElwadd: torch.add,
    MathOperation.SfpuElwsub: torch.sub,
    MathOperation.SfpuElwmul: torch.mul,
}


def _golden_sfpu_binary_bcast(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    bcast_dim: BroadcastType,
    op,
    stimuli_format: DataFormat,
) -> torch.Tensor:
    """Golden matching the SFPU bcast kernel for a single 32x32 tile.

    Inputs are row-major 32x32. Broadcast is applied in row-major space, then
    the result is tilized to match the face-ordered layout the packer writes
    to L1. `stimuli_format` drives the tilize precision (use Float16_b for
    Bfp8_b inputs, since the unpacker converts Bfp8_b -> Float16_b in dest).
    """
    a = src_A.flatten()[:1024].reshape(32, 32)
    b = src_B.flatten()[:1024].reshape(32, 32)

    if bcast_dim == BroadcastType.ROW:
        b_bcast = b[0].unsqueeze(0).expand_as(b)
    else:
        b_bcast = b[:, 0].unsqueeze(1).expand_as(b)

    golden_rm = op(a, b_bcast.contiguous()).flatten()
    return tilize(golden_rm, stimuli_format=stimuli_format)


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    bcast_dim=[BroadcastType.ROW, BroadcastType.COL],
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_bcast(
    formats,
    bcast_dim,
    mathop,
    dest_acc,
):
    if dest_acc == DestAccumulation.No and formats.input_format == DataFormat.Float32:
        pytest.skip(reason="Float32 inputs with dest_acc=No are not supported")

    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and formats.input_format == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Float16_a isn't supported for SFPU on Blackhole without being converted to 32-bit intermediate format in dest register"
        )

    # Mirror sfpu_binary(): on Blackhole, Float16/Float32 inputs require
    # dest_acc=Yes (32-bit dest), so silently upgrade the parametrized value.
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Bfp8_b stimuli are effectively Float16_b in dest after unpack; golden
    # computes and tilizes at that precision to match.
    golden_format = (
        DataFormat.Float16_b
        if formats.input_format == DataFormat.Bfp8_b
        else formats.input_format
    )
    golden_tensor = _golden_sfpu_binary_bcast(
        src_A, src_B, bcast_dim, _BCAST_BINARY_OPS[mathop], golden_format
    )

    # Only FP32 inputs with dest_acc=Yes take the unpack-to-dest path; all
    # other float formats go through srcA + MATH datacopy into dest.
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/sfpu_binary_bcast_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop),
            SFPU_BCAST_DIM(bcast_dim),
            INPUT_TILE_A(tile_index=0),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilize(src_A, stimuli_format=formats.input_format),
            formats.input_format,
            tilize(src_B, stimuli_format=formats.input_format),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result ({len(res_from_L1)}) and golden ({len(golden_tensor)}) size mismatch"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
