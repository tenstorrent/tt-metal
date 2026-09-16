# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole, skip_for_quasar, skip_for_wormhole
from fuser.config_parser import FUSER_CONFIG_DIR, FuserConfigSchema
from fuser.sweep import collect_fuser_cases

yaml_files = sorted(FUSER_CONFIG_DIR.glob("*.yaml"))
test_cases = collect_fuser_cases(yaml_files)


# https://github.com/tenstorrent/tt-llk/issues/1584
@skip_for_blackhole
@skip_for_wormhole
@skip_for_quasar
@pytest.mark.perf
@pytest.mark.parametrize("test_name, config_dict", test_cases)
def test_fuser(
    test_name,
    config_dict,
    regenerate_cpp,
    worker_id,
):
    config = FuserConfigSchema.load(test_name, config_dict)
    config.global_config.regenerate_cpp = regenerate_cpp
    config.run_perf_test(worker_id=worker_id)
