# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LLK SFPU quantization tests: quant (Float32 -> Int32), requant (Int32 -> Int32) and dequant (Int32 -> Float32)
with a per-tensor scale, in the two LLK forms of the scale (see perf_sfpu_quant_scalar.py): the scale as a DEST
tile (``tile``) and the scale loaded once by the init (``scalar``). Both forms run on the same stimuli and must
produce the same bits; the tile form is also checked against a host reference on values where the arithmetic is
exact, so the check does not depend on the rounding and saturation contracts of the kernels.

Int32 buffers use two's complement in L1 (twos_complement=True), the encoding the quant kernels read and write.
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
    """Input tile A. With ``exact`` every result is a whole number well inside the quantized range, so the host
    reference is exact whatever the rounding mode; otherwise the inputs cover fractions, ties and both saturation
    ends, and only the two forms are compared with each other."""
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
    q = torch.round(x.to(torch.float32) * _SCALE + _ZERO_POINT)
    return torch.clamp(q, -127, 127).to(torch.int32)


def _run(quant_op: str, scale_form: str, src_A: torch.Tensor) -> torch.Tensor:
    in_fmt, out_fmt = _QUANT_FORMATS[quant_op]
    formats = InputOutputFormat(in_fmt, out_fmt)
    dims = [TILE_DIM, TILE_DIM]
    # Tile B is the scale tile of the tile form: every datum the scale, as the binary_ng writer fills it. It
    # travels as the input format's bits (an Int32 buffer carries the fp32 bit pattern).
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
    exact=[True, False],
)
def test_sfpu_quant_scalar(quant_op, exact):
    if isinstance(quant_op, tuple):
        (quant_op,) = quant_op
    src_A = _stimuli(quant_op, exact, seed=7 if exact else 11)

    res_tile = _run(quant_op, "tile", src_A)
    res_scalar = _run(quant_op, "scalar", src_A)

    # The scalar-scale form is the tile-scale form without the per-row scale load: same bits, every lane.
    if res_tile.dtype == torch.float32:
        same = res_tile.view(torch.int32) == res_scalar.view(torch.int32)
    else:
        same = res_tile == res_scalar
    assert bool(same.all()), (
        f"{quant_op}: the scalar-scale form differs from the tile-scale form in {int((~same).sum())} lanes; "
        f"first tile {res_tile[~same][:8].tolist()} scalar {res_scalar[~same][:8].tolist()}"
    )

    if exact:
        golden = _reference(quant_op, src_A)
        if golden.dtype == torch.float32:
            diff = (res_tile - golden).abs()
        else:
            diff = (res_tile.to(torch.int64) - golden.to(torch.int64)).abs()
        assert float(diff.max()) == 0.0, (
            f"{quant_op}: {int((diff != 0).sum())} lanes differ from the host reference; "
            f"first result {res_tile[diff != 0][:8].tolist()} golden {golden[diff != 0][:8].tolist()}"
        )
