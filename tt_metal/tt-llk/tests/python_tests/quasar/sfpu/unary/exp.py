# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _exp_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_exp_<true>"}


SPEC = UnaryOpSpec(
    name="exp",
    math_op=MathOperation.Exp,
    include_header="sfpu/ckernel_sfpu_exp.h",
    sfpu_defines=_exp_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # Clamp to [-10, 10]: exp(10) ≈ 22026 < Float16 max (65504), so no
    # overflow in any output format.  Default (0.1–1.1) would only test the
    # small-positive branch.
    stimuli=StimuliSpec.uniform(low=-10.0, high=10.0),
)
