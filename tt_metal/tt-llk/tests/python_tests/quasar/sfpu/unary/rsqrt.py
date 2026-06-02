# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _rsqrt_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_rsqrt_"}


SPEC = UnaryOpSpec(
    name="rsqrt",
    math_op=MathOperation.Rsqrt,
    include_header="sfpu/ckernel_sfpu_rsqrt.h",
    sfpu_defines=_rsqrt_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # Default stimuli (0.1–1.1) are positive, which is the required domain
    # for rsqrt; no custom range needed.
)
