# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation

from ._spec import BinaryOpSpec, binary_input_output_formats


def _gt_int32_defines(_: FormatConfig) -> dict:
    # calculate_binary_comp_int32<APPROXIMATE, ITERATIONS, SfpuType::gt>(
    #     num_sfpu_iterations, in0_off, in1_off, out_off).
    # Int ops index by element OFFSET, not tile index; operands at tiles 0 and 1
    # (stride apart), result to tile 0.
    return {
        "SFPU_OP_CALL": "ckernel::sfpu::calculate_binary_comp_int32<false, 8, ckernel::SfpuType::gt>",
        "SFPU_BINARY_OPERANDS": "0, tile_stride, 0",
    }


SPEC = BinaryOpSpec(
    name="gt_int32",
    math_op=MathOperation.SfpuGtInt,
    include_header="sfpu/ckernel_sfpu_binary_comp.h",
    sfpu_defines=_gt_int32_defines,
    formats=binary_input_output_formats(
        [DataFormat.Int32], [DataFormat.Int32], [DataFormat.Int32]
    ),
)
