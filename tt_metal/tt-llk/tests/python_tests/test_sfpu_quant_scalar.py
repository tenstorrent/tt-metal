# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SFPU quantization tests (quant, requant, dequant) with a per-tensor scale, in the two LLK forms of
the scale (see perf_sfpu_quant_scalar.py), bit for bit against one host reference: whole-number
results (``exact``) and fractions, ties and both saturation ends. Int32 buffers are two's complement.
"""

import struct

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import QUANT_SCALAR_CFG, TILE_COUNT
from helpers.tilize_untilize import tilize_block, untilize_block

_QUANT_FORMATS = {
    "quant": (DataFormat.Float32, DataFormat.Int32),
    "requant": (DataFormat.Int32, DataFormat.Int32),
    "dequant": (DataFormat.Int32, DataFormat.Float32),
}

_ZERO_POINT = 3.0
_SCALE = 0.25


def _float_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _stimuli(quant_op: str, exact: bool, seed: int) -> torch.Tensor:
    """Input tile A: whole-number results with ``exact``, else fractions, ties and saturation."""
    g = torch.Generator().manual_seed(seed)
    if exact:
        # x * 0.25 + 3.0 is a whole number for x a multiple of 4; keep it inside [-120, 120].
        values = torch.randint(-120, 121, (ELEMENTS_PER_TILE,), generator=g) * 4
    else:
        values = torch.randint(-3000, 3001, (ELEMENTS_PER_TILE,), generator=g)
    if quant_op == "quant":
        x = values.to(torch.float32)
        if not exact:
            x = x + torch.randint(0, 4, (ELEMENTS_PER_TILE,), generator=g).to(torch.float32) * 0.5
        return x
    return values.to(torch.int32)


def _reference(quant_op: str, x: torch.Tensor) -> torch.Tensor:
    if quant_op == "dequant":
        return (x.to(torch.float32) - _ZERO_POINT) * _SCALE
    # x * scale + zero_point in float32 as the SFPMAD computes it, then the SFPSTOCHRND float to int8
    # rounding: nearest, ties away from zero, clamp to plus or minus 127 (exact in float64).
    y = (x.to(torch.float32) * _SCALE + _ZERO_POINT).to(torch.float64)
    q = torch.sign(y) * torch.floor(y.abs() + 0.5)
    return torch.clamp(q, -127, 127).to(torch.int32)


def _run(quant_op: str, scale_form: str, src_A: torch.Tensor) -> torch.Tensor:
    in_fmt, out_fmt = _QUANT_FORMATS[quant_op]
    formats = InputOutputFormat(in_fmt, out_fmt)
    dims = [TILE_DIM, TILE_DIM]
    # Tile B, the scale tile of the tile form, travels as the input format's bits.
    src_B = torch.full((ELEMENTS_PER_TILE,), _SCALE, dtype=torch.float32)
    if in_fmt == DataFormat.Int32:
        src_B = src_B.view(torch.int32)
    configuration = TestConfig(
        "sources/sfpu_quant_scalar_test.cpp",
        formats,
        templates=[
            QUANT_SCALAR_CFG(quant_op=quant_op, scale_form=scale_form, zero_point=_ZERO_POINT, scale=_SCALE),
        ],
        runtimes=[TILE_COUNT(1)],
        variant_stimuli=StimuliConfig(
            tilize_block(src_A, dims, stimuli_format=in_fmt).flatten(),
            in_fmt,
            tilize_block(src_B, dims, stimuli_format=in_fmt).flatten(),
            in_fmt,
            out_fmt,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            twos_complement=True,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res = torch.tensor(configuration.run().result, dtype=format_dict[out_fmt])
    return untilize_block(res, out_fmt, dims).flatten()


@parametrize(
    quant_op=["quant", "requant", "dequant"],
    scale_form=["tile", "scalar"],
    exact=[True, False],
)
def test_sfpu_quant_scalar(quant_op, scale_form, exact):
    src_A = _stimuli(quant_op, exact, seed=7 if exact else 11)
    res = _run(quant_op, scale_form, src_A)
    golden = _reference(quant_op, src_A)
    if golden.dtype == torch.float32:
        diff = (res.view(torch.int32).to(torch.int64) - golden.view(torch.int32).to(torch.int64)).abs()
    else:
        diff = (res.to(torch.int64) - golden.to(torch.int64)).abs()
    assert int((diff != 0).sum()) == 0, (
        f"{quant_op} ({scale_form} scale, exact={exact}): {int((diff != 0).sum())} lanes differ from the host "
        f"reference; first result {res[diff != 0][:8].tolist()} golden {golden[diff != 0][:8].tolist()}"
    )
