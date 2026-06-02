# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]


def _gelu_defines(_: FormatConfig) -> dict:
    return {
        "SFPU_OP_CALL": "ckernel::sfpu::_calculate_gelu_",
        # _init_gelu_ loads BF16-quantised constants (polynomial coefficients)
        # into LREGs before the per-row loop; omitting it leaves stale LREG
        # state and produces wrong results.
        "SFPU_INIT": "ckernel::sfpu::_init_gelu_();",
    }


SPEC = UnaryOpSpec(
    name="gelu",
    math_op=MathOperation.Gelu,
    include_header="sfpu/ckernel_sfpu_gelu.h",
    sfpu_defines=_gelu_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    stimuli=StimuliSpec.uniform(low=-10.0, high=10.0),
)
