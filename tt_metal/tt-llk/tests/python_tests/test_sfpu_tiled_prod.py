# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact regression for Wormhole's ``calculate_tiled_prod`` (metal ``llk_sfpu``).

The kernel keeps one running fp32 product per SFPU lane through the 32 elements the lane sees in
faces 0-3 and stores every partial product in place. It records a two-element body in the replay
buffer and replays it, and seeds the product with 1.0 so that the first element still goes through
an SFPMUL. The kernel runs on two tiles, called back to back, and the check is word for word
against a host model of the same chain over both tiles and over a guard tile that follows them in
Dest and must come back unchanged:

- a replay window or replay count that does not match the body sfpi emits stores the wrong
  products, or none, in part of a tile, or stores past it into the next tile;
- sfpi records repeated instruction sequences into the replay buffer on its own; a recording of
  its own that spans the kernel's ``lltt::record`` (back-to-back calls give it the repetition)
  nests one recording in another and hangs or corrupts the second tile;
- dropping the first SFPMUL (sfpi folds a literal ``1.0f * v`` to ``v``) stores a -0.0 first
  element as -0.0 instead of +0, which only the Float32 cells can see: the bf16 unpack hands the
  SFPU +0 for -0.0;
- NaN, infinity, denormal and over/underflowing products follow Wormhole's SFPMUL, modelled from
  ``fma_model_wh`` in tt-isa-documentation (Miscellaneous/FMA/fma.c).

The shared ``UnarySFPUGolden._tiled_prod`` is not used: it compares through tilize/untilize and
fp32 floats, which cannot express NaN payloads.

Wormhole only. Blackhole's kernel is a separate sfpi loop with a different SFPMUL (canonical NaN);
Quasar has its own implementation, covered by the fuser YAMLs.
"""

import struct

import pytest
import torch
from conftest import skip_for_blackhole, skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import ApproximationMode, DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import APPROX_MODE, generate_input_dim

pytestmark = [skip_for_blackhole, skip_for_quasar]

ELEMENTS_PER_TILE = TILE_DIM * TILE_DIM
FACES = 4
FACE_DIM = 16
STEPS_PER_FACE = 8  # ITERATIONS: each step covers two Dest rows
SFPU_LANE_ROWS = 4
SFPU_LANE_COLS = 8
FP32_ONE = 0x3F800000
SEED = 1234
GUARD_SEED = 4321
SFPU_TILE_COUNT = 2  # tiles the kernel runs on, back to back
TILE_COUNT = SFPU_TILE_COUNT + 1  # then the guard tile

# (input/output format, Dest accumulation). Float32 needs a 32-bit Dest.
FORMAT_CELLS = [
    (DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Float16_b, DestAccumulation.Yes),
    (DataFormat.Float32, DestAccumulation.Yes),
]

# Lanes whose first element is -0.0 in the "specials" stimuli.
SIGNED_ZERO_FIRST_LANES = range(SFPU_LANE_COLS)


def _lane_elements() -> list[list[int]]:
    """Face-order index of every element of each SFPU lane, in the order the kernel visits them.

    Lane (r, c) reads row r of each four-row group, column 2c on even steps and 2c + 1 on odd
    steps; a step advances two Dest rows, so two steps make one four-row group.
    """
    lanes = []
    for lane_row in range(SFPU_LANE_ROWS):
        for lane_col in range(SFPU_LANE_COLS):
            lane = []
            for face in range(FACES):
                for step in range(STEPS_PER_FACE):
                    row = SFPU_LANE_ROWS * (step // 2) + lane_row
                    col = 2 * lane_col + (step & 1)
                    lane.append(face * FACE_DIM * FACE_DIM + row * FACE_DIM + col)
            lanes.append(lane)
    return lanes


LANE_ELEMENTS = _lane_elements()


def _sfpmul_wh(x: int, y: int) -> int:
    """fp32 bits of Wormhole's SFPMUL x * y, i.e. fma_model_wh(x, y, +0.0).

    Denormal operands and results flush to +0, a zero product is +0 (the +0 addend), and a NaN
    result is ``product_sign | 0x7f800001`` with product mantissa bits OR-ed in when they do not
    carry out.
    """

    def unpack(v):
        exp = (v >> 23) & 0xFF
        man = 0 if exp == 0 else (v & 0x7FFFFF) | 0x800000
        return exp, man

    x_e, x_m = unpack(x)
    y_e, y_m = unpack(y)
    p_sign = (x ^ y) & 0x80000000
    p_m = (x_m * y_m) << 3  # three guard bits
    p_m = (p_m >> 23) | int(p_m & 0x7FFFFF != 0)
    p_e = x_e + y_e - 127

    nan_result = 0
    if x_e == 0xFF or y_e == 0xFF or p_e >= 0xFF:
        if (x_e == 0xFF and (x_m != 0x800000 or y_m == 0)) or (
            y_e == 0xFF and (y_m != 0x800000 or x_m == 0)
        ):
            nan_result = p_sign | 0x7F800001
        else:
            return p_sign | 0x7F800000
        p_e = min(p_e, 0xFF)
    if p_m == 0 or p_e < 0:
        if not nan_result:
            return 0
        p_m, p_e = 0, 0
    if p_m == 0:
        return nan_result

    shift = p_m.bit_length() - 27  # normalise to 1 integer bit, 26 fraction bits
    r_e = p_e + shift
    if r_e >= 0xFF:
        return nan_result or (p_sign | 0x7F800000)
    if r_e < 0:
        return nan_result
    p_m = p_m << -shift if shift <= 0 else (p_m >> shift) | (p_m & 1)
    r = (r_e << 23) + ((p_m >> 3) & 0x7FFFFF)
    r += int((p_m & 7) + (r & 1) > 4)  # round to nearest even
    if r >> 23 == 0:
        return nan_result
    return (nan_result or p_sign) | r


def _expected_words(in_words: list[int], fmt: DataFormat, dest_acc) -> list[int]:
    """Packed output words for one tile of input words, both in face order."""
    if fmt == DataFormat.Float32:
        operands = list(in_words)
    else:
        # bf16 reaches the SFPU as the top half of an fp32; the unpack turns -0.0 and denormals
        # into +0.
        operands = [0 if (w >> 7) & 0xFF == 0 else w << 16 for w in in_words]

    products = [0] * ELEMENTS_PER_TILE
    for lane in LANE_ELEMENTS:
        product = FP32_ONE
        for i in lane:
            product = _sfpmul_wh(product, operands[i])
            products[i] = product
    if fmt == DataFormat.Float32:
        return products

    out = []
    for p in products:
        if (p >> 23) & 0xFF == 0xFF:
            out.append(
                (p >> 16) & 0xFF80
            )  # a 16-bit result keeps a NaN as the signed infinity
        elif dest_acc == DestAccumulation.No:
            out.append(p >> 16)  # the bf16 Dest store truncates
        else:
            out.append(
                ((p + 0x8000) >> 16) & 0xFFFF
            )  # the fp32 -> bf16 pack rounds ties away
    return out


def _stimuli_words(fmt: DataFormat, kind: str, seed: int = SEED) -> list[int]:
    """One tile of input bit patterns in face order."""
    g = torch.Generator().manual_seed(seed)
    wide = fmt == DataFormat.Float32
    if kind == "near_one":
        # Every partial product stays finite and distinct, so each stored word checks one element.
        mag = torch.exp2(
            torch.empty(ELEMENTS_PER_TILE).uniform_(-1.5, 1.5, generator=g)
        )
        sign = torch.where(torch.rand(ELEMENTS_PER_TILE, generator=g) < 0.5, -1.0, 1.0)
        values = mag * sign
    elif kind == "random_bits":
        bits = 32 if wide else 16
        return torch.randint(0, 2**bits, (ELEMENTS_PER_TILE,), generator=g).tolist()
    else:
        specials = (
            [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000]
            + [0x7F800001, 0xFFA5A5A5, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000]
            + [0x80800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x5F800000]
            + [0x1F800000, 0x47800000, 0x37800000]
        )
        if not wide:
            specials = sorted({w >> 16 for w in specials} | {0x0001, 0x8001, 0x007F})
        pick = torch.randint(
            0, len(specials), (ELEMENTS_PER_TILE,), generator=g
        ).tolist()
        plain = (torch.rand(ELEMENTS_PER_TILE, generator=g) * 4 - 2).tolist()
        use_plain = (torch.rand(ELEMENTS_PER_TILE, generator=g) < 0.5).tolist()
        words = [
            _float_bits(plain[i], wide) if use_plain[i] else specials[pick[i]]
            for i in range(ELEMENTS_PER_TILE)
        ]
        negative_zero = 0x80000000 if wide else 0x8000
        for lane in SIGNED_ZERO_FIRST_LANES:
            words[LANE_ELEMENTS[lane][0]] = negative_zero
        return words
    return [_float_bits(v, wide) for v in values.tolist()]


def _float_bits(value: float, wide: bool) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if wide:
        return bits
    return torch.tensor([value], dtype=torch.bfloat16).view(torch.int16).item() & 0xFFFF


def _to_tensor(words: list[int], fmt: DataFormat) -> torch.Tensor:
    if fmt == DataFormat.Float32:
        return (
            torch.tensor(words, dtype=torch.int64).to(torch.int32).view(torch.float32)
        )
    return torch.tensor(words, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


@pytest.mark.parametrize(
    "fmt, dest_acc",
    FORMAT_CELLS,
    ids=[f"{f.name}-dest_acc_{d.name}" for f, d in FORMAT_CELLS],
)
@pytest.mark.parametrize("kind", ["near_one", "random_bits", "specials"])
def test_sfpu_tiled_prod_bit_exact(fmt, dest_acc, kind):
    formats = InputOutputFormat(fmt, fmt)
    in_words = [
        w
        for tile in range(SFPU_TILE_COUNT)
        for w in _stimuli_words(fmt, kind, SEED + tile)
    ]
    # Normal values only, so the guard tile goes through unpack, Dest and pack unchanged.
    guard_words = _stimuli_words(fmt, "near_one", GUARD_SEED)
    src_A = _to_tensor(in_words + guard_words, fmt)
    src_B = torch.zeros_like(src_A)

    configuration = TestConfig(
        "sources/sfpu_tiled_prod_test.cpp",
        formats,
        templates=[
            generate_input_dim([TILE_DIM, TILE_DIM], [TILE_DIM, TILE_DIM]),
            APPROX_MODE(ApproximationMode.No),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=TILE_COUNT,
            tile_count_B=TILE_COUNT,
            tile_count_res=TILE_COUNT,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=fmt.is_32_bit(),
        compile_time_formats=True,
    )
    configuration.run()

    word_bytes = 4 if fmt == DataFormat.Float32 else 2
    raw = configuration.variant_stimuli.collect_raw_result_bytes(
        TestConfig.TENSIX_LOCATION
    )
    words = TILE_COUNT * ELEMENTS_PER_TILE
    got = list(
        struct.unpack(
            f"<{words}{'I' if word_bytes == 4 else 'H'}", raw[: words * word_bytes]
        )
    )
    want = [
        w
        for tile in range(SFPU_TILE_COUNT)
        for w in _expected_words(
            in_words[tile * ELEMENTS_PER_TILE : (tile + 1) * ELEMENTS_PER_TILE],
            fmt,
            dest_acc,
        )
    ] + guard_words
    inputs = in_words + guard_words

    wrong = [i for i in range(words) if got[i] != want[i]]
    width = 2 * word_bytes
    assert not wrong, (
        f"tiled_prod {fmt.name} dest_acc={dest_acc.name} {kind}: {len(wrong)}/{words} words "
        f"differ from the host model, per tile {[sum(i // ELEMENTS_PER_TILE == t for i in wrong) for t in range(TILE_COUNT)]} "
        f"(the last is the guard tile, which the kernel must not touch). Face-order index: "
        f"input, got, expected:\n"
        + "\n".join(
            f"  [{i}] 0x{inputs[i]:0{width}X}: 0x{got[i]:0{width}X}, 0x{want[i]:0{width}X}"
            for i in wrong[:16]
        )
    )
