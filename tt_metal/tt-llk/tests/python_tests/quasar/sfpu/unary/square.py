# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Canonical minimal unary op spec: a plain elementwise op with the standard
# math_op-keyed golden. The only required pieces are the op call, header, math_op,
# and the format list. See BaseOpSpec (.._spec) for the full field contract and the
# optional hooks (stimuli, prepare, golden, SFPU_INIT / SFPU_ADDITIONAL_ARGS).

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation

from ._spec import UnaryOpSpec, input_output_formats


def _square_defines(formats: FormatConfig) -> dict:
    # _calculate_square_(int iterations) — the dispatch forwards num_sfpu_iterations,
    # so no extra args and no op-specific init are needed.
    return {"SFPU_OP_CALL": "ckernel::sfpu::_calculate_square_"}


SPEC = UnaryOpSpec(
    name="square",
    math_op=MathOperation.Square,
    include_header="sfpu/ckernel_sfpu_square.h",
    sfpu_defines=_square_defines,
    formats=input_output_formats(
        [
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Float32,
        ]
    ),
)
