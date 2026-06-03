# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fuser.fuser_config_parser import FUSER_CONFIG_DIR, FuserConfigSchema
from helpers.chip_architecture import ChipArchitecture
from helpers.test_config import TestConfig

yaml_files = sorted(FUSER_CONFIG_DIR.glob("*.yaml"))
_all_test_names = [f.stem for f in yaml_files]
test_names = (
    _all_test_names
    if TestConfig.CHIP_ARCH != ChipArchitecture.QUASAR and not TestConfig.WITH_COVERAGE
    else []
)


@pytest.mark.parametrize("test_name", test_names, ids=test_names)
def test_fuser(
    test_name,
    regenerate_cpp,
):
    config = FuserConfigSchema.load(test_name)
    config.global_config.regenerate_cpp = regenerate_cpp
    config.run_regular_test()
