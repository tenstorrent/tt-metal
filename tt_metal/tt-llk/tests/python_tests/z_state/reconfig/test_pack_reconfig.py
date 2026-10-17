# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, InputOutputFormat
from helpers.param_config import parametrize
from helpers.reconfig_formats import (
    NEXT_TILE_SIZE,
    RECONFIG_FORMATS,
    reconfig_dest_acc,
)
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


@parametrize(
    prev_src=RECONFIG_FORMATS,
    next_dst=RECONFIG_FORMATS,
    dest_acc=lambda prev_src, next_dst: reconfig_dest_acc(prev_src, next_dst),
)
def test_pack_reconfig(
    prev_src,
    next_dst,
    dest_acc,
):
    configuration = TestConfig(
        "sources/state/reconfig/pack_reconfig_test.cpp",
        # The kernel takes prev_dst and next_src from the inferred unpack_A_dst and pack_src.
        InputOutputFormat(prev_src, next_dst),
        runtimes=[
            CONFIGURE_TEST_RUN_IDX(0),
        ],
        dest_acc=dest_acc,
    )

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
