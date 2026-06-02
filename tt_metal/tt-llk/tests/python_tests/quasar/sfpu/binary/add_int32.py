# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation

from ._spec import BinaryOpSpec, binary_input_output_formats


def _add_int32_defines(_: FormatConfig) -> dict:
    # _add_int_<false,8,0,false>(DataFormat src_format, int num_iter, in0, in1, out).
    # Wrap in a capturing lambda so src_format (a runtime variable in the kernel
    # scope) is forwarded via capture; the lambda signature matches the binary
    # dispatch: (num_sfpu_iterations, in0_off, in1_off, out_off).
    return {
        "SFPU_INIT": (
            "const auto _add_fn = [src_format](int n, int in0, int in1, int out) "
            "{ ckernel::sfpu::_add_int_<false, 8, 0, false>(src_format, n, in0, in1, out); };"
        ),
        "SFPU_OP_CALL": "_add_fn",
        "SFPU_BINARY_OPERANDS": "0, tile_stride, 0",
    }


SPEC = BinaryOpSpec(
    name="add_int32",
    math_op=MathOperation.SfpuElwadd,
    include_header="sfpu/ckernel_sfpu_add.h",
    sfpu_defines=_add_int32_defines,
    formats=binary_input_output_formats(
        [DataFormat.Int32], [DataFormat.Int32], [DataFormat.Int32]
    ),
)
