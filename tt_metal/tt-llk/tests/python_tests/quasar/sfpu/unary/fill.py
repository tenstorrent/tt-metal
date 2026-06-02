# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation, format_dict

from ._spec import UnaryOpSpec, input_output_formats

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
_FILL_CONST = 5.0


def _fill_defines(_: FormatConfig) -> dict:
    # The unary path passes (num_sfpu_iterations) to SFPU_OP_CALL.  Wrap
    # _calculate_fill_<8> in a lambda that ignores the iterations arg and
    # broadcasts the compile-time constant instead; no SFPU_ADDITIONAL_ARGS needed.
    return {
        "SFPU_INIT": (
            f"static const auto _fill_fn = [](float) "
            f"{{ ckernel::sfpu::_calculate_fill_<8>({_FILL_CONST}f); }};"
        ),
        "SFPU_OP_CALL": "_fill_fn",
    }


def _fill_golden(*, src, io, **_):
    """Every output lane = FILL_CONST regardless of input."""
    n = src.flatten().numel()
    return torch.full((n,), _FILL_CONST, dtype=format_dict[io.output_format])


SPEC = UnaryOpSpec(
    name="fill",
    math_op=MathOperation.Fill,
    include_header="experimental/ckernel_sfpu_fill.h",
    sfpu_defines=_fill_defines,
    formats=input_output_formats(_FLOAT_FORMATS),
    golden=_fill_golden,
)
