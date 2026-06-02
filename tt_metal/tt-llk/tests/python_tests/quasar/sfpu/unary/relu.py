# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _relu_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_relu_"}


SPEC = UnaryOpSpec(
    name="relu",
    math_op=MathOperation.Relu,
    include_header="sfpu/ckernel_sfpu_relu.h",
    sfpu_defines=_relu_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # Symmetric range so the zero-clipping branch (negative inputs) and the
    # identity branch (positive inputs) are both exercised.
    stimuli=StimuliSpec.uniform(low=-1.1, high=1.1),
)
