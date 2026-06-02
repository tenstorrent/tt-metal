# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation

from ._spec import BinaryOpSpec, binary_input_output_formats


def _mul_int32_defines(formats: FormatConfig) -> dict:
    # _mul_int32_<APPROXIMATE, ITERATIONS>(iterations, in0_off, in1_off, out_off).
    # Int ops index by element OFFSET, not tile index, so override the kernel's
    # default tile-index operands: operands at tiles 0 and 1 (stride apart),
    # result to tile 0.
    return {
        "SFPU_OP_CALL": "ckernel::sfpu::_mul_int32_<false, 8>",
        "SFPU_BINARY_OPERANDS": "0, tile_stride, 0",
    }


SPEC = BinaryOpSpec(
    name="mul_int32",
    math_op=MathOperation.SfpuElwmulInt,
    include_header="sfpu/ckernel_sfpu_mul_int32.h",
    sfpu_defines=_mul_int32_defines,
    # The only binary SFPU ops in the Quasar LLK tree today are Int32.
    formats=binary_input_output_formats(
        [DataFormat.Int32], [DataFormat.Int32], [DataFormat.Int32]
    ),
)
