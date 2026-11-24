# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import flat_axes, parametrize
from helpers.reconfig_formats import NEXT_TILE_SIZE, RECONFIG_FORMATS
from helpers.tensix import TensixState, TensixThread
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


def _pack_dest_acc(
    prev_src: DataFormat, next_dst: DataFormat
) -> list[DestAccumulation]:
    # The 16/32-bit integer formats need the 32-bit Dest. Int8/UInt8 also pack from a 16-bit Dest,
    # where the Dest-read control selects the raw 8-bit integer read (PCK_DEST_RD_CTRL_Read_int8);
    # cover both Dest modes for a reconfig onto or away from them so the dumped control compares
    # the reconfig against the full configure in both directions.
    if any(f.is_integer() and f not in _INT8_FORMATS for f in (prev_src, next_dst)):
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


def _format_group(fmt: DataFormat) -> tuple[bool, bool, bool]:
    return (fmt.is_exponent_A(), fmt.is_exponent_B(), fmt.is_integer())


def _pack_next_src(
    prev_src: DataFormat, next_dst: DataFormat, dest_acc: DestAccumulation
) -> list[DataFormat | None]:
    """``pack_src`` overrides to sweep; ``None`` keeps the inferred one."""
    if dest_acc == DestAccumulation.Yes or not {prev_src, next_dst} & _INT8_FORMATS:
        return [None]
    # With a 16-bit Dest the format inference keeps pack_src at the previous (unpack) format,
    # which would give both runs the same Dest-read control (and an unsupported Int8 -> float
    # pack in run 0). Pack every format of next_dst's group instead, so the control changes and
    # converting packs such as UInt8 -> Int8 or Float32 -> Bfp8_b stay covered.
    return [
        f
        for f in RECONFIG_FORMATS
        if _format_group(f) == _format_group(next_dst)
        and (not f.is_integer() or f in _INT8_FORMATS)
    ]


# Float16 is out of the sweep below, but PCK_DEST_RD_CTRL_Round_10b_mant is set only when an fp32
# dest packs from Float16, and no functional test can see a 10-bit-mantissa rounding difference.
# These (prev_src, next_dst) rows keep the reconfig RMW of that bit covered, so they only run with
# dest_acc=Yes.
ROUND_10B_MANT_FORMATS = [(DataFormat.Float16, DataFormat.Float16_b)]  # clears the bit
# Sets the bit. Blackhole only: Wormhole infers pack_src Float32 for a Float16 output, so the bit
# would stay clear there.
if get_chip_architecture() != ChipArchitecture.WORMHOLE:
    ROUND_10B_MANT_FORMATS.append((DataFormat.Float16_b, DataFormat.Float16))


@parametrize(
    **flat_axes(
        ("prev_src", "next_dst", "dest_acc", "next_src"),
        [
            (prev_src, next_dst, dest_acc, next_src)
            for prev_src in RECONFIG_FORMATS
            for next_dst in RECONFIG_FORMATS
            for dest_acc in _pack_dest_acc(prev_src, next_dst)
            for next_src in _pack_next_src(prev_src, next_dst, dest_acc)
        ]
        + [
            (prev_src, next_dst, DestAccumulation.Yes, None)
            for prev_src, next_dst in ROUND_10B_MANT_FORMATS
        ],
    )
)
def test_pack_reconfig(
    prev_src,
    next_dst,
    dest_acc,
    next_src,
):
    configuration = TestConfig(
        "sources/state/reconfig/pack_reconfig_test.cpp",
        # The kernel takes prev_dst from the inferred unpack_A_dst and next_src from pack_src.
        InputOutputFormat(prev_src, next_dst),
        runtimes=[
            CONFIGURE_TEST_RUN_IDX(0),
        ],
        dest_acc=dest_acc,
    )
    if next_src is not None:
        for fmt_config in configuration.formats_config:
            fmt_config.pack_src = next_src

    configuration.run()
    expected = TensixState.fetch(TestConfig.TENSIX_LOCATION)
    if not TestConfig.TEST_TARGET.run_simulator:
        # configure_pack stores the tile size in the pack thread's TILE_HEADER GPR.
        assert expected["gpr"][TensixThread.PACK]["tile_header"] == NEXT_TILE_SIZE

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
