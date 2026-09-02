# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import (
    DestAccumulation,
)
from helpers.param_config import parametrize
from helpers.tensix import TensixState
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CONFIGURE_TEST_RUN_IDX, TO_FROM_INT8

# These may vary run to run. Excluded so that tests don't fail spuriously.
_IGNORED_GROUPS = ("address_counters", "register_window_counters")

FORMATS = [
    (
        DataFormat.Float16,
        DataFormat.Float16,
        DataFormat.Bfp8,
        DataFormat.Bfp8,
    ),  # expA->expA
    (
        DataFormat.Float32,
        DataFormat.Float32,
        DataFormat.Float16,
        DataFormat.Float16,
    ),  # expB->expA
    (
        DataFormat.Float16,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
    ),  # expA->expB
    (
        DataFormat.Bfp8,
        DataFormat.Bfp8,
        DataFormat.Bfp8_b,
        DataFormat.Bfp8_b,
    ),  # expA->expB
    (
        DataFormat.Float16,
        DataFormat.Float16,
        DataFormat.Int8,
        DataFormat.Int8,
    ),  # expA->int
    (
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float16,
    ),  # expB->expA
    (
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
        DataFormat.Bfp8_b,
    ),  # expB->expB
    (
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        DataFormat.Int8,
        DataFormat.Int8,
    ),  # expB->int
    (
        DataFormat.Int8,
        DataFormat.Int8,
        DataFormat.Float16,
        DataFormat.Float16,
    ),  # int->expA
    (
        DataFormat.Int8,
        DataFormat.Int8,
        DataFormat.Bfp8_b,
        DataFormat.Bfp8_b,
    ),  # int->expB
    (
        DataFormat.Int8,
        DataFormat.Int8,
        DataFormat.Int32,
        DataFormat.Int32,
    ),  # int->int
    (
        DataFormat.Bfp8_b,
        DataFormat.Bfp8_b,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
    ),  # leave BFP
]


@parametrize(
    formats=FORMATS,
    to_from_int8=lambda formats: any(f.is_integer() for f in formats),
    dest_acc=DestAccumulation.Yes,
)
def test_unpack_reconfig(
    formats,
    to_from_int8,
    dest_acc,
):
    prev_src, prev_dst, next_src, next_dst = formats

    configuration = TestConfig(
        "sources/state/reconfig/unpack_reconfig_test.cpp",
        FormatConfig(
            prev_src, prev_dst, next_src, next_dst, DataFormat.Float32
        ),  # slot overload: unpack_A_src/dst = prev, pack_src/dst = next
        templates=[
            TO_FROM_INT8(to_from_int8),
        ],
        runtimes=[
            CONFIGURE_TEST_RUN_IDX(0),
        ],
        dest_acc=dest_acc,
    )

    configuration.run()
    expected = TensixState.fetch(TestConfig.TENSIX_LOCATION)

    # Only the runtime parameter changes between runs.
    configuration.runtimes = [CONFIGURE_TEST_RUN_IDX(1)]

    configuration.run()
    actual = TensixState.fetch(TestConfig.TENSIX_LOCATION)

    for group in _IGNORED_GROUPS:
        expected.pop(group, None)
        actual.pop(group, None)

    TensixState.assert_equal(expected, actual)
