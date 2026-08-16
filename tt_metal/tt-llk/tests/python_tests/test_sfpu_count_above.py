# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness suite for the Blackhole SFPGT threshold-count kernel.

Drives ``sources/sfpu_count_above_test.cpp`` -- the checkable twin of
``ARM_COUNT_D1`` in ``sources/sfpu_count_above_perf.cpp`` -- so the perf numbers
come from a loop known to compute the right answer.

Two things shape every case below.

1. **SFPGT does not order like IEEE.** It uses the total order
   ``-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`` (tt-isa BlackholeA0
   ``SFPGT.md:3``). A golden written with ``numpy``/``torch`` ``>`` is wrong for
   signed zero and for every NaN, so the golden here transcribes
   ``SignMagIsSmaller`` instead (see ``sign_magnitude_order_key``). Several
   cases exist purely to pin that divergence down, and
   ``test_golden_disagrees_with_ieee`` asserts the two really do differ -- if a
   future refactor quietly swapped in an IEEE golden, that test fails first.

2. **A dropped accumulate is silent.** ``SFPLOADMACRO.md:149``: "If an
   instruction scheduled via SFPLOADMACRO arrives at a sub-unit on the same
   cycle as software issues a regular Vector Unit (SFPU) instruction to that
   sub-unit, then the scheduled instruction takes priority and the regular
   instruction is silently discarded." No fault, no watcher trip. A random
   half-above stimulus checked with PCC would look entirely plausible while
   dropping every second accumulate. The tripwire is an all-above stimulus
   compared for **exact** equality against ``count == N`` -- see
   ``all_above_one_tile`` / ``all_above_four_tiles``.

Everything is checked with exact integer equality, never PCC. The kernel
produces an integer count, so any tolerance at all would hide a real defect.
This follows ``test_sfpu_reduce.py::test_int32_reduce_extreme``, which likewise
bypasses ``passed_test`` and compares with an explicit ``!=`` mismatch count.

The device result is the sum of the *whole* packed output tile: the kernel
writes 32 per-lane partials with one SFPSTORE (which lands on 4 Dst rows x 8
even-or-odd columns) after zeroing every one of the tile's 1024 elements. A sum
is permutation-invariant, so the host never has to model the SFPLOAD lane map.
For the same reason the stimuli are *not* tilized: the count over a tile does
not depend on the element order within it.
"""

import random

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    CountAboveGolden,
    get_golden_generator,
    sign_magnitude_order_key,
)
from helpers.llk_params import DestAccumulation
from helpers.logger import logger
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    SFPU_UNARY_SCALAR,
)

# --- fp32 bit patterns, named -------------------------------------------------
# Everything is expressed as raw uint32 so that -0.0, the infinities and the
# NaNs are exact; going through a Python float would lose the NaN payload and
# collapse -0.0 in some paths.
POS_ZERO = 0x00000000  # +0.0
NEG_ZERO = 0x80000000  # -0.0
HALF = 0x3F000000  # 0.5
ONE = 0x3F800000  # 1.0
ONE_ULP_UP = 0x3F800001  # nextafter(1.0, +inf)
TWO = 0x40000000  # 2.0
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
POS_NAN = 0x7FC00000  # quiet +NaN
NEG_NAN = 0xFFC00000  # quiet -NaN

# Dest capacity: SyncHalf with a 32-bit dest holds 4 tiles
# (DEST_REGISTER_HALF_SIZE 512 rows >> 1 for fp32 >> 6 rows per tile).
MAX_DEST_TILES = 4


def _repeat(pattern: int, count: int) -> list[int]:
    return [pattern] * count


def _chunk_ramp(tile_count: int = 1) -> list[int]:
    """Chunk c of 32 elements gets exactly c elements above the threshold.

    Every one of the 32 chunks in a tile has a *distinct* hit count, so the
    total (0 + 1 + ... + 31 = 496 per tile) only comes out right if the Dst walk
    visits each 32-datum group exactly once. A stride bug that skips a group,
    revisits one, or runs off the end of the tile lands on a different total.
    """
    values: list[int] = []
    for _ in range(tile_count):
        for chunk in range(TILE_DIM):
            values.extend(_repeat(TWO, chunk))
            values.extend(_repeat(HALF, TILE_DIM - chunk))
    return values


def _single_hit() -> list[int]:
    """One element above, in the middle of the tile.

    SFPGT's mask is -1 per hit (SFPGT.md:29), not -(2**31 - 1) or -0x7FFFFFFF,
    so exactly one hit must read out as exactly 1. Any other mask magnitude
    shows up here immediately instead of being averaged away.
    """
    values = _repeat(HALF, ELEMENTS_PER_TILE)
    values[ELEMENTS_PER_TILE // 2] = TWO
    return values


def _equal_and_one_ulp_above() -> list[int]:
    """Half the tile exactly at the threshold, half one ULP above it.

    Pins ``>`` against ``>=``: a ``>=`` kernel returns 1024 here, a correct one
    returns 512.
    """
    half = ELEMENTS_PER_TILE // 2
    return _repeat(ONE, half) + _repeat(ONE_ULP_UP, half)


def _random_bits(tile_count: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.getrandbits(32) for _ in range(tile_count * ELEMENTS_PER_TILE)]


# --- stimulus matrix ----------------------------------------------------------
#
# (test id, threshold bits, values, tile count, LOOP_FACTOR, expected total)
#
# `expected` is written out by hand so the golden is itself under test: the
# device is checked against the golden AND the golden against these numbers, so
# a bug in the transcription of SignMagIsSmaller cannot cancel against a bug in
# the kernel.
STIMULI = [
    # Baseline: nothing exceeds the threshold. Catches a kernel that counts
    # everything (e.g. an inverted compare, or a stuck LaneEnabled mask).
    ("all_below", ONE, _repeat(HALF, ELEMENTS_PER_TILE), 1, 1, 0),
    # THE SILENT-DISCARD CATCHER. Every element is above, so the count must be
    # exactly the element count. One dropped SFPIADD anywhere in the 32-load
    # walk shows up as 1024 - 32k.
    ("all_above_one_tile", HALF, _repeat(ONE, ELEMENTS_PER_TILE), 1, 1, 1024),
    (
        "all_above_four_tiles",
        HALF,
        _repeat(ONE, MAX_DEST_TILES * ELEMENTS_PER_TILE),
        MAX_DEST_TILES,
        1,
        4096,
    ),
    # `>` and not `>=`.
    ("exactly_equal", ONE, _repeat(ONE, ELEMENTS_PER_TILE), 1, 1, 0),
    ("equal_vs_one_ulp_above", ONE, _equal_and_one_ulp_above(), 1, 1, 512),
    # Mask magnitude is -1, not -(2**31 - 1).
    ("single_hit", ONE, _single_hit(), 1, 1, 1),
    # Dst row/column addressing: 32 distinct per-chunk counts summing to 496.
    ("positional_ramp", ONE, _chunk_ramp(), 1, 1, 496),
    # SFPGT ranks -0.0 strictly below +0.0; IEEE calls them equal, so an IEEE
    # golden says 0 here and the hardware says 1024.
    ("neg_zero_threshold", NEG_ZERO, _repeat(POS_ZERO, ELEMENTS_PER_TILE), 1, 1, 1024),
    # +NaN sits above +Inf in the total order; IEEE says every NaN comparison is
    # false, so an IEEE golden says 0 and the hardware says 1024.
    ("pos_inf_vs_pos_nan", POS_INF, _repeat(POS_NAN, ELEMENTS_PER_TILE), 1, 1, 1024),
    # -NaN sits below -Inf, so nothing is above the threshold.
    ("neg_inf_vs_neg_nan", NEG_INF, _repeat(NEG_NAN, ELEMENTS_PER_TILE), 1, 1, 0),
    # ...and the other direction, which is what makes the pair discriminating:
    # with the operands swapped the same two values give the opposite answer.
    ("neg_nan_vs_neg_inf", NEG_NAN, _repeat(NEG_INF, ELEMENTS_PER_TILE), 1, 1, 1024),
    # Full-range bit patterns: infinities, NaNs of both signs, denormals and
    # both zeros all appear. This is the case a torch `>` golden gets wrong.
    ("mixed_random_bits", ONE, _random_bits(1, seed=0), 1, 1, None),
    # ACCUMULATOR WIDTH. A per-lane count is at most 32 per tile pass, so the
    # only way past 2**16 is to re-walk the resident tiles: 4 tiles x 1024
    # passes x 32 = 131072 per lane, well clear of a 16-bit wrap anywhere in
    # the SFPIADD -> SFPSTORE -> pack path. The total is 32 x that.
    (
        "accumulator_width_over_16_bits",
        HALF,
        _repeat(ONE, MAX_DEST_TILES * ELEMENTS_PER_TILE),
        MAX_DEST_TILES,
        1024,
        4 * 1024 * 1024,
    ),
]


def _run_count_above(
    values_bits: list[int], threshold_bits: int, tile_count: int, repeat: int
) -> int:
    """Run the kernel and return the sum of the whole packed output tile."""
    formats = InputOutputFormat(DataFormat.UInt32, DataFormat.UInt32)

    # UInt32 in and out, so the harness's packer/unpacker move raw 32-bit words
    # (pack_uint32/unpack_uint32) and every bit pattern survives the round trip
    # exactly. Int32 would route through the sign-magnitude packer unless
    # twos_complement were set, and neither encoding is what we want here: the
    # stimuli are fp32 bit patterns, not integers.
    src_A = torch.tensor(values_bits, dtype=torch.int64)
    assert src_A.numel() == tile_count * ELEMENTS_PER_TILE
    src_B = torch.zeros_like(src_A)

    configuration = TestConfig(
        "sources/sfpu_count_above_test.cpp",
        formats,
        # The threshold is a raw fp32 bit pattern, which is exactly what
        # SFPU_UNARY_SCALAR emits (`constexpr std::uint32_t SFPU_UNARY_SCALAR`).
        # It has to be a compile-time constant because the kernel loads it into
        # L_THR before recording the replay body.
        templates=[SFPU_UNARY_SCALAR(threshold_bits)],
        runtimes=[
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(tile_count),
            LOOP_FACTOR(repeat),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=1,
            # Only Dest tile 0 is packed out; it carries the 32 lane partials
            # plus 992 zeros.
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.Yes,  # 32-bit formats cannot unpack to SrcA/SrcB
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    return sum(int(v) for v in res_from_L1)


@pytest.mark.parametrize(
    "threshold_bits, values_bits, tile_count, repeat, expected",
    [pytest.param(*case[1:], id=case[0]) for case in STIMULI],
)
def test_sfpu_count_above(threshold_bits, values_bits, tile_count, repeat, expected):
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            reason="SFPGT (and its mask-as-value SET_VD form) is new in Blackhole; "
            "there is no Wormhole/Quasar equivalent for this kernel to target."
        )

    assert tile_count <= MAX_DEST_TILES, (
        f"{tile_count} tiles exceeds the {MAX_DEST_TILES}-tile SyncHalf 32-bit Dest "
        "section; the kernel keeps every input tile resident for the whole run."
    )

    # The device run comes first on purpose: under ``--compile-producer`` it
    # compiles and then skips, and ``get_golden_generator`` has been swapped for
    # ``dummy_golden_generator`` (which returns a zero tensor, not an int). Doing
    # the golden afterwards keeps the compile-only pass out of that code.
    device = _run_count_above(values_bits, threshold_bits, tile_count, repeat)

    golden = get_golden_generator(CountAboveGolden)(values_bits, threshold_bits, repeat)

    if expected is not None:
        # Cross-check the golden against the hand-derived count, so a defect in
        # the SignMagIsSmaller transcription cannot silently agree with a defect
        # in the kernel.
        assert golden == expected, (
            f"golden disagrees with the hand-derived expectation: "
            f"golden={golden} expected={expected}"
        )

    if device != golden:
        logger.info(
            "\nthreshold=0x{:08X} tiles={} repeat={}: device={} golden={} (delta {})",
            threshold_bits,
            tile_count,
            repeat,
            device,
            golden,
            device - golden,
        )

    assert device == golden, (
        f"count mismatch for threshold=0x{threshold_bits:08X}, "
        f"{tile_count} tile(s), repeat={repeat}: device={device} golden={golden}"
    )


def test_sign_magnitude_key_anchors():
    """Anchor the golden's order key against the four values that define it.

    Host-only; needs no device. ``sign_magnitude_order_key`` is a transcription
    of ``SignMagIsSmaller`` (SFPGT.md:55-66), and these four patterns are the
    ones where a transcription slip (wrong shift, unsigned instead of
    arithmetic, mask off by a bit) shows up.
    """
    anchors = {
        POS_ZERO: 0,  # +0.0 sits at the origin of the order
        NEG_ZERO: -1,  # -0.0 is immediately below +0.0
        0xFFFFFFFF: torch.iinfo(torch.int32).min,  # most negative -NaN: lowest
        0x7FFFFFFF: torch.iinfo(torch.int32).max,  # largest +NaN: highest
    }
    keys = sign_magnitude_order_key(torch.tensor(list(anchors), dtype=torch.int64))
    for (bits, want), got in zip(anchors.items(), keys.tolist()):
        assert got == want, f"key(0x{bits:08X}) = {got}, expected {want}"

    # And the full documented total order, end to end.
    total_order = [
        0xFFFFFFFF,  # -NaN (largest payload)
        NEG_NAN,
        NEG_INF,
        0xBF800000,  # -1.0
        NEG_ZERO,
        POS_ZERO,
        ONE,
        POS_INF,
        POS_NAN,
        0x7FFFFFFF,  # +NaN (largest payload)
    ]
    ordered = sign_magnitude_order_key(
        torch.tensor(total_order, dtype=torch.int64)
    ).tolist()
    assert ordered == sorted(ordered), (
        "sign_magnitude_order_key must be monotone over "
        "-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN"
    )


def test_golden_disagrees_with_ieee():
    """Guard: the golden must NOT be reducible to a torch/numpy ``>``.

    If someone ever "simplifies" ``CountAboveGolden`` into an IEEE comparison,
    the special-value cases in ``STIMULI`` would start passing against a wrong
    reference. This test states the disagreement explicitly. Host-only.

    The generator is instantiated directly rather than fetched through
    ``get_golden_generator``: under ``--compile-producer`` that lookup is
    rebound to ``dummy_golden_generator``, and this test has no device run to
    skip on.
    """
    golden = CountAboveGolden()

    def ieee_count(values_bits, threshold_bits):
        values = torch.tensor(values_bits, dtype=torch.int64).to(torch.int32)
        threshold = torch.tensor([threshold_bits], dtype=torch.int64).to(torch.int32)
        return int(
            (values.view(torch.float32) > threshold.view(torch.float32)).sum().item()
        )

    # -0.0 threshold, +0.0 data: IEEE says they are equal (0 above), SFPGT ranks
    # -0.0 strictly lower (all above).
    data = _repeat(POS_ZERO, ELEMENTS_PER_TILE)
    assert golden(data, NEG_ZERO) == ELEMENTS_PER_TILE
    assert ieee_count(data, NEG_ZERO) == 0

    # +Inf threshold, +NaN data: IEEE says every NaN comparison is false,
    # SFPGT ranks +NaN above +Inf.
    data = _repeat(POS_NAN, ELEMENTS_PER_TILE)
    assert golden(data, POS_INF) == ELEMENTS_PER_TILE
    assert ieee_count(data, POS_INF) == 0

    # -NaN threshold, -Inf data: IEEE false again, SFPGT ranks -Inf above -NaN.
    data = _repeat(NEG_INF, ELEMENTS_PER_TILE)
    assert golden(data, NEG_NAN) == ELEMENTS_PER_TILE
    assert ieee_count(data, NEG_NAN) == 0
