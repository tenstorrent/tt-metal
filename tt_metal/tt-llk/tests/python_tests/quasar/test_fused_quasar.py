# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole, skip_for_coverage, skip_for_wormhole
from fuser.config_parser import FUSER_CONFIG_DIR, FuserConfigSchema
from fuser.sweep import collect_fuser_cases

yaml_files = sorted(FUSER_CONFIG_DIR.glob("*.yaml"))
yaml_files += sorted((FUSER_CONFIG_DIR / "quasar").glob("*.yaml"))
test_cases = collect_fuser_cases(yaml_files)


@skip_for_blackhole
@skip_for_wormhole
@skip_for_coverage
@pytest.mark.parametrize("test_name, config_dict", test_cases)
def test_fuser(
    test_name,
    config_dict,
    regenerate_cpp,
):
    config = FuserConfigSchema.load(test_name, config_dict)
    config.global_config.regenerate_cpp = regenerate_cpp
    config.run_regular_test()
