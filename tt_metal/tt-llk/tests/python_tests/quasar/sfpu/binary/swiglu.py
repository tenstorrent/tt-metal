# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import MathOperation, format_dict

from ._spec import BinaryInputOutputFormat, BinaryOpSpec

_FLOAT_FORMATS = [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
_ELEMENTS_PER_TILE = 1024

# GPT-OSS swiglu hyperparameters, matching ckernel_sfpu_swiglu.h SwiGLUConfigGPTOSS.
_SWIGLU_ALPHA = 1.702
_SWIGLU_CLAMP = 7.0


def _swiglu_defines(_: FormatConfig) -> dict:
    # The binary path calls SFPU_OP_CALL(num_sfpu_iterations, SFPU_BINARY_OPERANDS).
    # _calculate_swiglu_ takes (n, gate_off, up_off, out_off).  Wrap it in a lambda
    # that ignores the three trailing tile-layout args from SFPU_BINARY_OPERANDS and
    # supplies the fixed offsets (gate=tile0, up=tile1, out=tile2 = 0/64/128 rows).
    # _init_swiglu_ hoists three LREG constants and can be called before sfpu_start.
    return {
        "SFPU_INIT": (
            "ckernel::sfpu::_init_swiglu_(); "
            "static const auto _sw_fn = "
            "[](std::uint32_t n, std::uint32_t, std::uint32_t, std::uint32_t) "
            "{ ckernel::sfpu::_calculate_swiglu_(n, 0u, 64u, 128u); };"
        ),
        "SFPU_OP_CALL": "_sw_fn",
    }


def _swiglu_golden(*, src, io, **_):
    """GPT-OSS swiglu: (up_c + 1) * gate_c * sigmoid(alpha * gate_c)."""
    flat = src.flatten()
    gate = flat[:_ELEMENTS_PER_TILE].to(torch.float32)
    up = flat[_ELEMENTS_PER_TILE : 2 * _ELEMENTS_PER_TILE].to(torch.float32)
    gate_c = torch.minimum(gate, torch.tensor(_SWIGLU_CLAMP))
    up_c = torch.clamp(up, -_SWIGLU_CLAMP, _SWIGLU_CLAMP)
    sig = torch.sigmoid(_SWIGLU_ALPHA * gate_c)
    return ((up_c + 1.0) * gate_c * sig).to(format_dict[io.output_format])


SPEC = BinaryOpSpec(
    name="swiglu",
    math_op=MathOperation.SfpuSwiGLU,
    include_header="experimental/ckernel_sfpu_swiglu.h",
    sfpu_defines=_swiglu_defines,
    formats=[BinaryInputOutputFormat(f, f, f) for f in _FLOAT_FORMATS],
    # gate=tile0, up=tile1; output written to tile2 by the kernel.
    # DST_INDEX=2 (from dst_idx=2) so PACK reads the output tile directly.
    tile_layouts=[(0, 1, 2)],
    golden=_swiglu_golden,
)
