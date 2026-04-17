# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole, skip_for_quasar, skip_for_wormhole
from fuser.fuser_config_parser import FUSER_CONFIG_DIR, FuserConfigSchema

yaml_files = sorted(FUSER_CONFIG_DIR.glob("*.yaml"))
test_names = [f.stem for f in yaml_files]


# https://github.com/tenstorrent/tt-llk/issues/1584
@skip_for_blackhole
@skip_for_wormhole
@skip_for_quasar
@pytest.mark.perf
@pytest.mark.parametrize("test_name", test_names, ids=test_names)
def test_fuser(
    test_name,
    regenerate_cpp,
    worker_id,
):
    config = FuserConfigSchema.load(test_name)
    config.global_config.regenerate_cpp = regenerate_cpp
    config.run_perf_test(worker_id=worker_id)
