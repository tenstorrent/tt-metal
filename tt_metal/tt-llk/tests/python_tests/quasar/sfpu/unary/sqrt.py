# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _sqrt_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_sqrt_<true>"}


SPEC = UnaryOpSpec(
    name="sqrt",
    math_op=MathOperation.Sqrt,
    include_header="sfpu/ckernel_sfpu_sqrt.h",
    sfpu_defines=_sqrt_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # Default stimuli (0.1–1.1) are positive, which is the required domain
    # for sqrt; sqrt(0.1–1.1) ≈ 0.316–1.049, fits all output formats.
)
