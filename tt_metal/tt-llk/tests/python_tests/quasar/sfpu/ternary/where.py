# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation, format_dict

from ._spec import TernaryInputOutputFormat, TernaryOpSpec

# All three operands share the same format; sweep the full input × output matrix.
_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
_ELEMENTS_PER_TILE = 1024


def _where_defines(formats: FormatConfig) -> dict:
    # calculate_where<APPROXIMATE, ITERATIONS=8>(in0, in1, in2, out); the ternary
    # dispatch forwards the four tile indices, so only the call, its SfpuType-templated
    # init, and the CC-stack prime (init_where) are needed.
    return {
        "SFPU_OP_CALL": "ckernel::sfpu::calculate_where<false>",
        "SFPU_TYPE": "ckernel::SfpuType::where",
        "SFPU_INIT": "ckernel::sfpu::init_where();",
    }


def _alternating_condition(src, *, src_tiles):
    """Overwrite the condition tile with an alternating 0/1 pattern so both select
    branches are exercised (random stimuli are almost all non-zero -> true branch only).
    ``src_tiles`` is (cond_tile, true_tile, false_tile); only the cond tile is touched.
    """
    cond_tile = src_tiles[0]
    flat = src.flatten().clone()
    start = cond_tile * _ELEMENTS_PER_TILE
    pattern = torch.arange(_ELEMENTS_PER_TILE) % 2
    flat[start : start + _ELEMENTS_PER_TILE] = pattern.to(flat.dtype)
    return flat.reshape(src.shape)


def _where_golden(*, src, io, **_):
    """Element-wise ternary select matching WhereGolden.

    Operates directly on the face-sequential stimulus tensor (cond | true_val | false_val
    concatenated), which is the native order generate_stimuli returns.  Bypasses the
    TernarySFPUGolden tilize path that omits the untilize step."""
    flat = src.flatten()
    cond = flat[:_ELEMENTS_PER_TILE]
    true_val = flat[_ELEMENTS_PER_TILE : 2 * _ELEMENTS_PER_TILE]
    false_val = flat[2 * _ELEMENTS_PER_TILE : 3 * _ELEMENTS_PER_TILE]
    mask = cond.to(torch.float32) != 0.0
    return torch.where(mask, true_val, false_val).to(format_dict[io.output_format])


SPEC = TernaryOpSpec(
    name="where",
    math_op=MathOperation.SfpuWhere,
    include_header="llk_sfpu/ckernel_sfpu_where.h",
    sfpu_defines=_where_defines,
    formats=[
        TernaryInputOutputFormat(i, i, i, o)
        for i in _FLOAT_FORMATS
        for o in _FLOAT_FORMATS
    ],
    prepare=_alternating_condition,
    golden=_where_golden,
)
