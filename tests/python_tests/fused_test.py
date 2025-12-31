# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from helpers.device import (
    collect_pipeline_results,
    write_pipeline_operands_to_l1,
)
from helpers.format_config import DataFormat, FormatConfig
from helpers.fused_generator import FUSED_TESTS_DIR
from helpers.fused_golden import FusedGolden
from helpers.fused_pipeline import create_fuse_pipeline
from helpers.test_config import BootMode, ProfilerBuild, TestConfig

FUSER_CONFIG_DIR = Path(__file__).parent / "fuser_config"
yaml_files = sorted(FUSER_CONFIG_DIR.glob("*.yaml"))
test_names = [f.stem for f in yaml_files]


@pytest.mark.parametrize("test_name", test_names, ids=test_names)
def test_fused(test_name, regenerate_cpp):
    yaml_path = FUSER_CONFIG_DIR / f"{test_name}.yaml"
    cpp_path = FUSED_TESTS_DIR / f"{test_name}.cpp"

    pipeline = create_fuse_pipeline(str(yaml_path))

    write_pipeline_operands_to_l1(pipeline)

    config = TestConfig(
        test_name=cpp_path,
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

    config.run_fused(pipeline, regenerate_cpp)

    collect_pipeline_results(pipeline)

    golden = FusedGolden()
    assert golden.check_pipeline(pipeline)
