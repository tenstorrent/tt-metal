# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
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
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_float(
    formats,
    dest_acc,
    mathop,
):
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
        and mathop == MathOperation.SfpuElwsub
    ):
        pytest.skip("Not currently supported in tests")

    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
        and mathop in [MathOperation.SfpuElwadd, MathOperation.SfpuElwmul]
        and dest_acc == DestAccumulation.No
        and formats.input_format == DataFormat.Float32
    ):
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1092")

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
        mathop,
    )


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=[
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwLogicalRightShift,
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
        sfpu=True,
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
):

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
        src_A,  # Contains tiles 0 and 1
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

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
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
