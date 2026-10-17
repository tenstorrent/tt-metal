# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, FormatConfig, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
)
from helpers.param_config import parametrize
from helpers.reconfig_formats import RECONFIG_FORMATS, reconfig_dest_acc
from helpers.tensix import TensixState
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CONFIGURE_TEST_RUN_IDX


@parametrize(
    prev_a=RECONFIG_FORMATS,
    next_b=RECONFIG_FORMATS,
    dest_acc=lambda prev_a, next_b: reconfig_dest_acc(prev_a, next_b),
)
def test_math_reconfig(
    prev_a,
    next_b,
    dest_acc,
):
    # tt-metal#34499: reconfig now always re-derives INT8_math_enabled from the new format, so a reconfig
    # (run idx 1) must land in the same ALU state as a fresh hw_configure for the new formats (run idx 0),
    # with no to_from_int8 opt-in flag -- including across an int8 boundary.
    configuration = TestConfig(
        "sources/state/reconfig/math_reconfig_test.cpp",
        # The kernel takes prev_b and next_a from the inferred unpack_A_dst and pack_src.
        InputOutputFormat(prev_a, next_b),
        runtimes=[
            CONFIGURE_TEST_RUN_IDX(0),
        ],
        dest_acc=dest_acc,
    )

    configuration.run()
    expected = TensixState.fetch(TestConfig.TENSIX_LOCATION)

    # We needn't instance the TestConfig object once again, because we're changing only the runtime parameters
    configuration.runtimes = [CONFIGURE_TEST_RUN_IDX(1)]

    configuration.run()
    actual = TensixState.fetch(TestConfig.TENSIX_LOCATION)

    TensixState.assert_equal(expected, actual)


def generate_int8_boundary_formats() -> (
    list[tuple[DataFormat, DataFormat, DataFormat, DataFormat]]
):
    # Reconfigs that cross the INT8_math_enabled boundary: a float side (bit = 0) <-> an Int8/Int32 side (bit = 1).
    # These are the cases where skip_int8 is observable -- it leaves the bit stale instead of re-deriving it.
    floats = [DataFormat.Float16_b, DataFormat.Float16]
    ints = [DataFormat.Int8, DataFormat.Int32]
    return [(f, f, i, i) for f in floats for i in ints] + [
        (i, i, f, f) for f in floats for i in ints
    ]


@parametrize(
    formats=generate_int8_boundary_formats(),
    dest_acc=lambda formats: [
        DestAccumulation.Yes
    ],  # int8/int32 math requires FP32 dest accumulation
)
def test_math_reconfig_skip_int8(
    formats,
    dest_acc,
):
    prev_a, prev_b, next_a, next_b = formats

    # tt-metal#34499: the _skip_int8 path (run idx 2) leaves INT8_math_enabled untouched, whereas the default
    # path (run idx 1) re-derives it. Across an int8 boundary the two must therefore land in different ALU
    # states -- this is the coverage that pins the skip_int8=true branch and its "do not re-derive" contract.
    configuration = TestConfig(
        "sources/state/reconfig/math_reconfig_test.cpp",
        FormatConfig(prev_a, prev_b, next_a, next_b, DataFormat.Float32),
        runtimes=[CONFIGURE_TEST_RUN_IDX(1)],
        dest_acc=dest_acc,
    )

    configuration.run()
    derived = TensixState.fetch(TestConfig.TENSIX_LOCATION)

    configuration.runtimes = [CONFIGURE_TEST_RUN_IDX(2)]
    configuration.run()
    skipped = TensixState.fetch(TestConfig.TENSIX_LOCATION)

    TensixState.assert_not_equal(derived, skipped)
