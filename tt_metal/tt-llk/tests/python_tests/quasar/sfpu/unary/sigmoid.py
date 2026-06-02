# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _sigmoid_defines(_: FormatConfig) -> dict:
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_sigmoid_"}


SPEC = UnaryOpSpec(
    name="sigmoid",
    math_op=MathOperation.Sigmoid,
    include_header="sfpu/ckernel_sfpu_sigmoid.h",
    sfpu_defines=_sigmoid_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    # [-10, 10] covers the full saturation transition: sigmoid(x) → 0 for x→-∞,
    # → 0.5 at x=0, → 1 for x→+∞.  Default (0.1–1.1) only tests the
    # near-linear region.
    stimuli=StimuliSpec.uniform(low=-10.0, high=10.0),
)
