# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _reciprocal_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_reciprocal_<true>"}


SPEC = UnaryOpSpec(
    name="reciprocal",
    math_op=MathOperation.Reciprocal,
    include_header="sfpu/ckernel_sfpu_recip.h",
    sfpu_defines=_reciprocal_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # Default stimuli (0.1–1.1) are already positive and away from zero,
    # so no custom range is needed — 1/x stays finite and in-range for all
    # three output formats.
)
