# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _silu_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_silu_"}


SPEC = UnaryOpSpec(
    name="silu",
    math_op=MathOperation.Silu,
    include_header="sfpu/ckernel_sfpu_silu.h",
    sfpu_defines=_silu_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # SiLU = x * sigmoid(x). [-10, 10] exercises the negative (near-zero
    # output) and positive (linear-ish) branches without exp overflow.
    stimuli=StimuliSpec.uniform(low=-10.0, high=10.0),
)
