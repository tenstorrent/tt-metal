# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GGUF weight decoders on the SFPU and the DEST-fed matmul (Blackhole), through sources/gguf_decode_mm_test.cpp."""

import math

import numpy as np
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.gguf_records import (
    FORMATS,
    dequantize,
    pages,
    random_blocks,
    records,
    tiles_kn,
)
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, GGUF_DECODE_MM, MATH_FIDELITY

pytestmark = blackhole_only

TILE = 32
TILE_BYTES_32BIT = 4096
TILE_BYTES_BF16 = 2048
FIDELITY_PHASES = {
    MathFidelity.LoFi: 1,
    MathFidelity.HiFi2: 2,
    MathFidelity.HiFi3: 3,
    MathFidelity.HiFi4: 4,
}


def _k_unit(fmt):
    """The smallest K that holds whole groups and whole blocks."""
    return math.lcm(TILE * fmt.group_tiles, fmt.block_elems)


def _decode_shapes(fmt):
    """[N, K]: one N tile and one group, and two N tiles over at least eight K tiles."""
    unit = _k_unit(fmt)
    return [(TILE, unit), (2 * TILE, max(2 * unit, 8 * TILE))]


def _tilize(matrix: np.ndarray) -> np.ndarray:
    """[32, 32] -> the four 16x16 faces in L1 order."""
    return matrix.reshape(2, 16, 2, 16).transpose(0, 2, 1, 3).reshape(-1)


def _untilize(flat: torch.Tensor, tiles: int) -> torch.Tensor:
    return (
        flat.reshape(tiles, 2, 2, 16, 16)
        .permute(0, 1, 3, 2, 4)
        .reshape(tiles, TILE, TILE)
    )


def _weight_pages(fmt, blocks):
    """All pages of the weight, N tile by N tile, as one L1 image."""
    return b"".join(
        b"".join(tile_pages) for tile_pages in pages(fmt, records(fmt, blocks))
    )


def _in0_tiles(in0: np.ndarray) -> bytes:
    """[rows, K] bf16 values -> K / 32 tiles of 32 rows (rows past `rows` zero), 4 faces each."""
    rows, k = in0.shape
    padded = np.zeros((TILE, k), dtype=np.float32)
    padded[:rows] = in0
    bits = torch.from_numpy(padded).to(torch.bfloat16).view(torch.int16).numpy()
    return b"".join(
        _tilize(bits[:, TILE * kt : TILE * (kt + 1)]).tobytes()
        for kt in range(k // TILE)
    )


def _run(
    fmt_id,
    kt,
    nt,
    a_image,
    a_format,
    output_tiles,
    matmul=False,
    in0_rows=32,
    in0_image=None,
    fidelity=MathFidelity.HiFi2,
    page_rows=0,
):
    in0_faces = 2 if in0_rows <= 16 else 4
    in0_image = in0_image if in0_image is not None else bytes(TILE_BYTES_BF16)
    configuration = TestConfig(
        "sources/gguf_decode_mm_test.cpp",
        InputOutputFormat(a_format, DataFormat.Float32),
        templates=[
            GGUF_DECODE_MM(
                fmt_id=fmt_id,
                kt=kt,
                nt=nt,
                matmul=matmul,
                in0_rows=in0_rows,
                in0_faces=in0_faces,
                page_rows=page_rows,
            ),
            MATH_FIDELITY(fidelity),
            DEST_SYNC(DestSync.Full),
        ],
        # Prepacked L1 images: the tile counts only reserve each operand's region.
        variant_stimuli=StimuliConfig(
            buffer_A=a_image,
            stimuli_A_format=a_format,
            tile_count_A=-(-len(a_image) // TILE_BYTES_32BIT),
            buffer_B=in0_image,
            stimuli_B_format=DataFormat.Float16_b,
            tile_count_B=len(in0_image) // TILE_BYTES_BF16,
            stimuli_res_format=DataFormat.Float32,
            tile_count_res=output_tiles,
        ),
        dest_acc=DestAccumulation.Yes,
        disable_format_inference=True,
    )
    result = torch.tensor(configuration.run().result, dtype=torch.float32)
    return _untilize(result, output_tiles)


def _src_fields(x: torch.Tensor):
    """Sign, exponent and top 10 mantissa bits of float32 values: the Src datum (TF32 by truncation)."""
    bits = x.contiguous().view(torch.int32).numpy().astype(np.int64) & 0xFFFFFFFF
    return bits >> 31, (bits >> 23) & 0xFF, (bits >> 13) & 0x3FF


def _round_shift(x, n):
    """x / 2^n to nearest, ties towards +infinity."""
    return (x + np.where(n > 0, 1 << np.maximum(n - 1, 0), 0)) >> n


def _mvmul_golden(
    in0: torch.Tensor, in1: torch.Tensor, fidelity: MathFidelity
) -> torch.Tensor:
    """in0 [M, K] (bf16 values, SrcB) @ in1 [K, N] (Float32 in DEST, moved to SrcA as TF32), bit for bit as the FPU
    computes it into a 32-bit DEST: tt-isa-documentation's MatrixUnitArithmetic.md, finite values. Per in1 tile, per
    fidelity phase, per 16-deep K half, one MVMUL: each lane of 8 products sums them in fixed point with 10 fractional
    bits of its largest exponent; the two lane sums and the DEST value are aligned to their common exponent, added,
    and rounded to Float32 in DEST; a result exponent of 255 or more is an infinity of the result's sign.
    """
    sa, ea, ma = _src_fields(in1)
    sb, eb, mb = _src_fields(in0)
    shape = (in0.shape[0], in1.shape[1])
    d_sign, d_exp, d_man = (np.zeros(shape, np.int64) for _ in range(3))
    for k_tile in range(0, in0.shape[1], TILE):
        for phase in range(FIDELITY_PHASES[fidelity]):
            for half in (k_tile, k_tile + 16):
                lanes = []
                for lane in (half, half + 8):
                    k = slice(lane, lane + 8)
                    zero = (ea[k][None] == 0) | (eb[:, k][:, :, None] == 0)
                    exp_sum = np.where(zero, 0, ea[k][None] + eb[:, k][:, :, None])
                    top = exp_sum.max(axis=1)
                    a = (ma[k] >> 1) & 0x1F if phase & 1 else 0x10 | ma[k] >> 6
                    b = (mb[:, k] & 0xF) << 3 if phase & 2 else 0x40 | mb[:, k] >> 4
                    shift = np.minimum(top[:, None] - exp_sum, 16)
                    aligned = (((a[None] * b[:, :, None]) << 1 >> shift) + 1) >> 1
                    aligned = np.where(zero | (shift >= 16), 0, aligned)
                    negative = sa[k][None] != sb[:, k][:, :, None]
                    exp = top - 127 - 5 * (phase & 1) - 7 * (phase >> 1)
                    total = np.where(negative, -aligned, aligned).sum(axis=1)
                    lanes.append(
                        (np.where(exp > 0, exp, 0), np.where(exp > 0, total, 0))
                    )
                exp = np.maximum(np.maximum(lanes[0][0], lanes[1][0]), d_exp)
                acc = sum(
                    _round_shift(s << 13, np.minimum(exp - e, 31)) for e, s in lanes
                )
                dest = _round_shift(
                    np.where(d_exp > 0, 0x800000 | d_man, 0),
                    np.minimum(exp - d_exp, 31),
                )
                acc = acc + np.where(d_sign == 1, -dest, dest)
                mag = np.abs(acc)
                msb = np.frexp(mag.astype(np.float64))[1] - 1
                rounded = ((mag << np.maximum(28 - msb, 0) >> 4) + 1) >> 1
                carry = rounded >> 24
                result_exp = exp + msb - 23 + carry
                live = (exp > 0) & (mag > 0) & (result_exp > 0)
                finite = result_exp < 255
                d_sign = np.where(live, acc < 0, 0)
                d_exp = np.where(live, np.minimum(result_exp, 255), 0)
                d_man = np.where(live & finite, (rounded >> carry) & 0x7FFFFF, 0)
    bits = (d_sign << 31) | (d_exp << 23) | d_man
    return torch.from_numpy(bits.astype(np.uint32).view(np.float32))


def _assert_matmul(
    device: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, fidelity: MathFidelity
):
    """device: the output tiles [nt][32][32]; in0 [rows, K]; in1 [K, N]. Bit-exact."""
    rows = in0.shape[0]
    nt = in1.shape[1] // TILE
    golden = torch.zeros(TILE, nt * TILE)
    golden[:rows] = _mvmul_golden(in0, in1, fidelity)
    result = device.permute(1, 0, 2).reshape(TILE, nt * TILE)
    wrong = torch.nonzero(result != golden)
    assert not len(wrong), (
        f"{len(wrong)} values differ, first at {wrong[:4].tolist()}: device {result[result != golden][:4].tolist()},"
        f" golden {golden[result != golden][:4].tolist()}"
    )


@parametrize(fmt=list(FORMATS), shape=lambda fmt: _decode_shapes(FORMATS[fmt]))
def test_gguf_decode(fmt, shape):
    """Random ggml blocks -> records -> pages -> decoded Float32 [K][N] tiles, equal to ggml's dequantization
    (gguf-py) value for value."""
    f = FORMATS[fmt]
    n, k = shape
    nt, kt = n // TILE, k // TILE
    blocks = random_blocks(f, n, k, np.random.default_rng(1000 + f.id))
    device = _run(
        f.id,
        kt,
        nt,
        _weight_pages(f, blocks),
        DataFormat.UInt32,
        nt * kt,
    )

    weights = dequantize(f, blocks)
    for t in range(nt):
        for j in range(kt):
            golden = torch.from_numpy(np.ascontiguousarray(tiles_kn(weights, t, j)))
            # Equal values; a zero may differ in sign (q * S + M against ggml's (q - m) * d).
            assert torch.equal(device[t * kt + j], golden), (
                f"{fmt}: tile (n {t}, k {j}) differs at {torch.nonzero(device[t * kt + j] != golden)[:8].tolist()}:"
                f" device {device[t * kt + j][device[t * kt + j] != golden][:8].tolist()},"
                f" golden {golden[device[t * kt + j] != golden][:8].tolist()}"
            )


@parametrize(fmt=list(FORMATS), rows=[1, 13, 32])
def test_gguf_decode_mm(fmt, rows):
    """The same weights decoded in DEST and multiplied with bf16 in0 rows; bit-exact against the FPU's arithmetic."""
    f = FORMATS[fmt]
    n, k = 2 * TILE, max(_k_unit(f), 8 * TILE)
    nt, kt = n // TILE, k // TILE
    rng = np.random.default_rng(2000 + 7 * f.id + rows)
    blocks = random_blocks(f, n, k, rng, wide_scales=False)
    in0 = torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(rows, k)).astype(np.float32)
    ).to(torch.bfloat16)
    in0_rows = 8 if rows <= 8 else 16 if rows <= 16 else 32
    device = _run(
        f.id,
        kt,
        nt,
        _weight_pages(f, blocks),
        DataFormat.UInt32,
        nt,
        matmul=True,
        in0_rows=in0_rows,
        in0_image=_in0_tiles(in0.to(torch.float32).numpy()),
    )
    in1 = torch.from_numpy(dequantize(f, blocks)).T.contiguous()
    _assert_matmul(device, in0.to(torch.float32), in1, MathFidelity.HiFi2)


@parametrize(fmt=["Q4_K", "Q5_K", "Q3_K", "Q2_K", "Q4_0"])
def test_gguf_decode_mm_page_rows(fmt):
    """The decode (and the group step, where the format has one) with a DEST row offset: rows = 64 one tile lower."""
    (fmt,) = fmt
    f = FORMATS[fmt]
    n, k = TILE, max(_k_unit(f), 8 * TILE)
    nt, kt = n // TILE, k // TILE
    rng = np.random.default_rng(4000 + f.id)
    blocks = random_blocks(f, n, k, rng, wide_scales=False)
    in0 = torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(TILE, k)).astype(np.float32)
    ).to(torch.bfloat16)
    device = _run(
        f.id,
        kt,
        nt,
        _weight_pages(f, blocks),
        DataFormat.UInt32,
        nt,
        matmul=True,
        in0_image=_in0_tiles(in0.to(torch.float32).numpy()),
        page_rows=64,
    )
    in1 = torch.from_numpy(dequantize(f, blocks)).T.contiguous()
    _assert_matmul(device, in0.to(torch.float32), in1, MathFidelity.HiFi2)


@parametrize(
    fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4],
    rows=[5, 16, 32],
    kt=[1, 3, 8],
)
def test_dest_srca_mm(fidelity, rows, kt):
    """The matmul alone, on Float32 in1 tiles unpacked to DEST; bit-exact against the FPU's arithmetic."""
    nt = 2
    rng = np.random.default_rng(3000 + 31 * kt + rows)
    # Full fp32 mantissas, so that the TF32 truncation of in1 matters.
    in1 = torch.from_numpy(
        rng.standard_normal((kt * TILE, nt * TILE)).astype(np.float32)
    )
    in0 = torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(rows, kt * TILE)).astype(np.float32)
    ).to(torch.bfloat16)
    image = b"".join(
        _tilize(
            in1[TILE * j : TILE * (j + 1), TILE * t : TILE * (t + 1)].numpy()
        ).tobytes()
        for t in range(nt)
        for j in range(kt)
    )
    in0_rows = 8 if rows <= 8 else 16 if rows <= 16 else 32
    device = _run(
        0,
        kt,
        nt,
        image,
        DataFormat.Float32,
        nt,
        matmul=True,
        in0_rows=in0_rows,
        in0_image=_in0_tiles(in0.to(torch.float32).numpy()),
        fidelity=fidelity,
    )
    _assert_matmul(device, in0.to(torch.float32), in1, fidelity)


@parametrize(fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4])
def test_dest_srca_mm_overflow(fidelity):
    """Finite operands whose sums overflow Float32: infinities of the result's sign, also with an infinity in DEST."""
    (fidelity,) = fidelity
    nt = 2
    in1 = torch.empty(TILE, nt * TILE)
    # 32 products of 2**124 overflow in the first K half; 2**123 at the second (2**127, then 2**128); 2**120 stays finite.
    for c, v in enumerate([2.0**124, -(2.0**124), 2.0**123, 2.0**120]):
        in1[:, 16 * c : 16 * (c + 1)] = v
    in0 = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    image = b"".join(
        _tilize(in1[:, TILE * t : TILE * (t + 1)].numpy()).tobytes() for t in range(nt)
    )
    device = _run(
        0,
        1,
        nt,
        image,
        DataFormat.Float32,
        nt,
        matmul=True,
        in0_image=_in0_tiles(in0.to(torch.float32).numpy()),
        fidelity=fidelity,
    )
    golden = _mvmul_golden(in0.to(torch.float32), in1, fidelity)
    assert (
        torch.isposinf(golden[:, :16]).all() and torch.isneginf(golden[:, 16:32]).all()
    )
    assert torch.isposinf(golden[:, 32:48]).all() and (golden[:, 48:] == 2.0**125).all()
    _assert_matmul(device, in0.to(torch.float32), in1, fidelity)
