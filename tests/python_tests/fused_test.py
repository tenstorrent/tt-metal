# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from helpers.device import (
    collect_pipeline_results,
    write_pipeline_operands_to_l1,
)
from helpers.format_config import DataFormat, FormatConfig
from helpers.fused_golden import FusedGolden
from helpers.fused_pipeline import create_fuse_pipeline
from helpers.param_config import parametrize
from helpers.test_config import BootMode, ProfilerBuild, TestConfig


@parametrize(
    test_name="fused_test",
)
def test_fused(test_name):
    pipeline = create_fuse_pipeline()

    write_pipeline_operands_to_l1(pipeline)

    config = TestConfig(
        test_name="fused_test",
        formats=FormatConfig(
            unpack_A_src=DataFormat.Float16,
            unpack_A_dst=DataFormat.Float16,
            pack_src=DataFormat.Float16,
            pack_dst=DataFormat.Float16,
            math=DataFormat.Float16,
        ),
        templates=set(),
        runtimes=set(),
        variant_stimuli=None,
        boot_mode=BootMode.DEFAULT,
        profiler_build=ProfilerBuild.No,
    )

    config.run_fused(pipeline)

    collect_pipeline_results(pipeline)

    golden = FusedGolden()
    assert golden.check_pipeline(pipeline)
