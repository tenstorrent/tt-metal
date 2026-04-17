# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum

import torch
from conftest import skip_for_blackhole, skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_OP,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize, untilize
from helpers.utils import passed_test


class SfpuBcastDim(Enum):
    BCAST_COL = 0
    BCAST_ROW = 1


@dataclass
class SFPU_BCAST_DIM(TemplateParameter):
    bcast_dim: SfpuBcastDim

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t BCAST_DIM_VAL = {self.bcast_dim.value};"


@dataclass
class INPUT_TILE_A(TemplateParameter):
    """Base DST tile index for input A.

    The kernel derives the other tile indices from this single value:
      INPUT_TILE_A  -> data tile
      INPUT_TILE_A + 1 -> bcast tile
      INPUT_TILE_A + 2 -> result tile
    """

    tile_index: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t INPUT_TILE_A_VAL = {self.tile_index};"


_FACE_R_DIM = 16
_FACE_C_DIM = 16

_BINARY_OPS = {
    MathOperation.SfpuElwadd: torch.add,
    MathOperation.SfpuElwsub: torch.sub,
    MathOperation.SfpuElwmul: torch.mul,
}


def _golden_sfpu_binary_bcast(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    bcast_dim: SfpuBcastDim,
    op,
) -> torch.Tensor:
    """Compute the golden result matching the SFPU binary bcast kernel.

    Both inputs are row-major 32x32.  The broadcast is applied across the
    full tile in row-major space, then the result is tilized to match the
    face-ordered layout that the packer writes to L1.

    BCAST_ROW: row 0 of src_B is replicated to all 32 rows.
    BCAST_COL: column 0 of src_B is replicated to all 32 columns.
    """
    a = src_A.flatten()[:1024].reshape(32, 32)
    b = src_B.flatten()[:1024].reshape(32, 32)

    if bcast_dim == SfpuBcastDim.BCAST_ROW:
        b_bcast = b[0].unsqueeze(0).expand_as(b)
    else:
        b_bcast = b[:, 0].unsqueeze(1).expand_as(b)

    golden_rm = op(a, b_bcast.contiguous()).flatten()
    return tilize(golden_rm, stimuli_format=DataFormat.Float32)


SUPPORTED_ELTWISE_OPS = [
    MathOperation.SfpuElwadd,
    MathOperation.SfpuElwsub,
    MathOperation.SfpuElwmul,
]

SUPPORTED_FORMATS = input_output_formats(
    [
        DataFormat.Float32,
    ]
)


@skip_for_blackhole
@skip_for_quasar
@parametrize(
    formats=SUPPORTED_FORMATS,
    bcast_dim=[SfpuBcastDim.BCAST_ROW, SfpuBcastDim.BCAST_COL],
    eltwise_op=SUPPORTED_ELTWISE_OPS,
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_bcast(
    formats: InputOutputFormat,
    bcast_dim: SfpuBcastDim,
    eltwise_op: MathOperation,
    dest_acc: DestAccumulation,
):
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Debug overrides: row-major src_A = 0, src_B[:, 0] = tile_row, rest = 0.
    # Golden operates on row-major inputs; StimuliConfig below tilizes them
    # before sending to L1.
    # src_A = torch.zeros(1024, dtype=src_A.dtype)
    # src_B = torch.zeros(1024, dtype=src_B.dtype)
    # src_B_rm = src_B.view(32, 32)
    # # For row bcasting: set src_B's row 0 to increasing values, rest zero
    # for c in range(32):
    #     src_B_rm[0, c] = float(c)

    # print("src_A (row-major view):")
    # print(src_A.view(32, 32))
    # print("src_B (row-major view, kernel does the broadcast):")
    # print(src_B.view(32, 32))

    op = _BINARY_OPS[eltwise_op]
    golden_tensor = _golden_sfpu_binary_bcast(src_A, src_B, bcast_dim, op)

    # print("golden_tensor (face-ordered):")
    # print(golden_tensor.view(32, 32))

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/sfpu_binary_bcast_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=eltwise_op),
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

    untlized_golden_tensor = untilize(
        golden_tensor, stimuli_format=formats.output_format
    )

    untilized_res_tensor = untilize(res_tensor, stimuli_format=formats.output_format)
    # print("untlized_golden_tensor:")
    # print(untlized_golden_tensor.view(32, 32))
    # print("untilized_res_tensor:")
    # print(untilized_res_tensor.view(32, 32))

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
