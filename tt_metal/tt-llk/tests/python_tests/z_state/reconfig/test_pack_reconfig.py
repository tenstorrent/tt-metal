# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import product

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import (
    DestAccumulation,
)
from helpers.param_config import parametrize
from helpers.tensix import TensixState
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CONFIGURE_TEST_RUN_IDX

_BFP_FORMATS = {
    DataFormat.Bfp8,
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
    DataFormat.Bfp2_b,
}
_FP8_FORMATS = {
    f
    for f in (getattr(DataFormat, "Fp8_e4m3", None), getattr(DataFormat, "Lf8", None))
    if f is not None
}
_INT8_FORMATS = {DataFormat.Int8, DataFormat.UInt8}

# These may vary run to run. Excluded so that tests don't fail spuriously.
_IGNORED_GROUPS = ("address_counters", "register_window_counters")


def _exp_section_size_required(dst: DataFormat) -> bool:
    return dst in _BFP_FORMATS or dst in _INT8_FORMATS or dst in _FP8_FORMATS


def _drop_key(state, key):
    if isinstance(state, dict):
        return {k: _drop_key(v, key) for k, v in state.items() if k != key}
    if isinstance(state, list):
        return [_drop_key(v, key) for v in state]
    return state


def generate_valid_formats(
    formats: list[DataFormat],
) -> list[tuple[DataFormat, DataFormat, DataFormat, DataFormat]]:
    groups = [
        [f for f in formats if f.is_exponent_A()],
        [f for f in formats if f.is_exponent_B()],
        [f for f in formats if f.is_integer()],
    ]

    return [
        combo
        for prev_group in groups
        for next_group in groups
        for combo in product(prev_group, prev_group, next_group, next_group)
    ]


def get_valid_dest_acc(
    formats: tuple[DataFormat, DataFormat, DataFormat, DataFormat],
) -> list[DestAccumulation]:
    # The 16/32-bit integer formats need the 32-bit Dest. Int8/UInt8 also pack from a 16-bit Dest,
    # where the Dest-read control selects the raw 8-bit integer read (PCK_DEST_RD_CTRL_Read_int8);
    # cover both Dest modes for a reconfig onto or away from them so the dumped control compares
    # the reconfig against the full configure in both directions. The Round_10b_mant rows are
    # 32-bit Dest only (the bit is set only when an fp32 Dest packs from Float16).
    if any(f.is_integer() and f not in _INT8_FORMATS for f in formats):
        return [DestAccumulation.Yes]
    if formats in ROUND_10B_MANT_FORMATS:
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


# Float16 is out of the sweep below, but PCK_DEST_RD_CTRL_Round_10b_mant is set only when an fp32
# dest packs from Float16, and no functional test can see a 10-bit-mantissa rounding difference.
# These rows keep the reconfig RMW of that bit covered, so they only run with dest_acc=Yes.
ROUND_10B_MANT_FORMATS = [
    (
        DataFormat.Float16,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
    ),  # clears the bit
]
# Sets the bit. Blackhole only: Wormhole infers pack_src Float32 for a Float16 output, so the bit
# would stay clear there.
if get_chip_architecture() != ChipArchitecture.WORMHOLE:
    ROUND_10B_MANT_FORMATS.append(
        (
            DataFormat.Float16_b,
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float16,
        )
    )


@parametrize(
    formats=generate_valid_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Bfp8,
            DataFormat.Bfp8_b,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Tf32,
            DataFormat.UInt32,
            DataFormat.UInt16,
            DataFormat.Int8,
            DataFormat.UInt8,
        ]
    )
    + ROUND_10B_MANT_FORMATS,
    dest_acc=lambda formats: get_valid_dest_acc(formats),
)
def test_pack_reconfig(
    formats,
    dest_acc,
):
    prev_src, prev_dst, next_src, next_dst = formats

    configuration = TestConfig(
        "sources/state/reconfig/pack_reconfig_test.cpp",
        FormatConfig(
            prev_src, prev_dst, next_src, next_dst, DataFormat.Float32
        ),  # slot overload: unpack_A_src/dst = prev, pack_src/dst = next
        runtimes=[
            CONFIGURE_TEST_RUN_IDX(0),
        ],
        dest_acc=dest_acc,
    )
    if dest_acc == DestAccumulation.No and (
        prev_src in _INT8_FORMATS or next_src in _INT8_FORMATS
    ):
        # With a 16-bit Dest the format inference keeps pack_src at the previous (unpack) format,
        # which would give both runs the same Dest-read control (and an unsupported Int8 -> float
        # pack in run 0). Pack the next format's own register representation so the control
        # changes between the previous and the next format.
        for fmt_config in configuration.formats_config:
            fmt_config.pack_src = next_src

    configuration.run()
    expected = TensixState.fetch(TestConfig.TENSIX_LOCATION)

    # Only the runtime parameter changes between runs.
    configuration.runtimes = [CONFIGURE_TEST_RUN_IDX(1)]

    configuration.run()
    actual = TensixState.fetch(TestConfig.TENSIX_LOCATION)

    for group in _IGNORED_GROUPS:
        expected.pop(group, None)
        actual.pop(group, None)

    if not _exp_section_size_required(next_dst):
        expected = _drop_key(expected, "exp_section_size")
        actual = _drop_key(actual, "exp_section_size")

    TensixState.assert_equal(expected, actual)
