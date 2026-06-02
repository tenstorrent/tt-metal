# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _abs_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_abs_"}


SPEC = UnaryOpSpec(
    name="abs",
    math_op=MathOperation.Abs,
    include_header="experimental/ckernel_sfpu_abs.h",
    sfpu_defines=_abs_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # Include negative values so both the sign-clear path and the identity path
    # are exercised (random positive-only stimuli would only test the latter).
    stimuli=StimuliSpec.uniform(low=-1.1, high=1.1),
)
