# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.fuser_config_parser import FUSER_CONFIG_DIR, load_fuser_config

yaml_files = sorted(FUSER_CONFIG_DIR.glob("*.yaml"))
test_names = [f.stem for f in yaml_files]


@pytest.mark.parametrize("test_name", test_names, ids=test_names)
def test_fuser(test_name, regenerate_cpp, workers_tensix_coordinates):
    config = load_fuser_config(test_name)
    config.global_config.regenerate_cpp = regenerate_cpp
    config.run_regular_test(location=workers_tensix_coordinates)
