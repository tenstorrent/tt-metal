# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import float_binary_op_spec

SPEC = float_binary_op_spec(
    name="mul",
    math_op=MathOperation.SfpuElwmul,
    binop="MUL",
    stimuli=StimuliSpec.uniform(low=-4.0, high=4.0),
)
