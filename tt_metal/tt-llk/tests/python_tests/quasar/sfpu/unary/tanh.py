# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _tanh_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_tanh_<true>"}


SPEC = UnaryOpSpec(
    name="tanh",
    math_op=MathOperation.Tanh,
    include_header="sfpu/ckernel_sfpu_tanh.h",
    sfpu_defines=_tanh_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # tanh saturates to ±1 outside ≈ ±4; [-10, 10] covers saturation, steep
    # region, and the linear region near 0.
    stimuli=StimuliSpec.uniform(low=-10.0, high=10.0),
)
