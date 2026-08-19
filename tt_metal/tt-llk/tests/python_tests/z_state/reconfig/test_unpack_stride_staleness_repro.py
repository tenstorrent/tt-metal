# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Regression test (tt-llk#1161 follow-up): a format-only reconfig_data_format that changes datum size
# (fp16 -> fp32) must re-commit the format-derived ch1 Z-stride. The kernel reads the stride back and
# LLK_ASSERTs it matches the new format's canonical stride. run_idx 0 (direct fp32 configure) and
# run_idx 1 (fp16 then format-only reconfig) must both pass; before the fix run_idx 1 fired the assert.

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CONFIGURE_TEST_RUN_IDX, TO_FROM_INT8


@parametrize(
    # unpack_A slots = prev (fp16); pack slots = next (fp32).
    formats=[
        (DataFormat.Float16, DataFormat.Float16, DataFormat.Float32, DataFormat.Float32)
    ],
    # run_idx 0 = CONTROL (direct fp32 configure, asserts must pass); 1 = BUGGY (fp16 then format-only reconfig).
    run_idx=[0, 1],
    dest_acc=DestAccumulation.Yes,
)
def test_unpack_stride_staleness_repro(formats, run_idx, dest_acc):
    prev_src, prev_dst, next_src, next_dst = formats

    configuration = TestConfig(
        "sources/state/reconfig/unpack_stride_staleness_repro_test.cpp",
        FormatConfig(prev_src, prev_dst, next_src, next_dst, DataFormat.Float32),
        templates=[
            TO_FROM_INT8(False),
        ],
        runtimes=[
            CONFIGURE_TEST_RUN_IDX(run_idx),
        ],
        dest_acc=dest_acc,
    )

    # run_idx 0 (control) must pass. run_idx 1 (buggy) fires the in-kernel LLK_ASSERT if the stride is stale.
    configuration.run()
