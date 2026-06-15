# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation, format_dict

from ._spec import BinaryInputOutputFormat, BinaryOpSpec

_FLOAT_FORMATS = [
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Float32,
    DataFormat.MxFp8R,
    DataFormat.MxFp8P,
]
_ELEMENTS_PER_TILE = 1024


def _min_defines(formats: FormatConfig) -> dict:
    df = (
        "DataFormat::Int32"
        if formats.input_format.is_integer()
        else "DataFormat::Float32"
    )
    return {
        "SFPU_INIT": (
            f"ckernel::sfpu::_init_binary_max_min_(); "
            f"static const auto _min_fn = [](int, unsigned in0, unsigned in1, unsigned out) "
            f"{{ ckernel::sfpu::calculate_binary_max_min<{df}, false, 8>(in0, in1, out); }};"
        ),
        "SFPU_OP_CALL": "_min_fn",
    }


def _min_golden(*, src, io, **_):
    """Element-wise min of tile0 and tile1; result at tile0."""
    flat = src.flatten()
    in0 = flat[:_ELEMENTS_PER_TILE].to(torch.float32)
    in1 = flat[_ELEMENTS_PER_TILE : 2 * _ELEMENTS_PER_TILE].to(torch.float32)
    return torch.minimum(in0, in1).to(format_dict[io.output_format])


SPEC = BinaryOpSpec(
    name="min",
    math_op=MathOperation.Abs,  # placeholder; custom golden is used
    include_header="experimental/ckernel_sfpu_binary_max_min.h",
    sfpu_defines=_min_defines,
    formats=[BinaryInputOutputFormat(f, f, f) for f in _FLOAT_FORMATS],
    tile_layouts=[(0, 1, 0)],
    golden=_min_golden,
)
