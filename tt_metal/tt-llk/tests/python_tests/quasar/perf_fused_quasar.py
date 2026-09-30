# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_blackhole, skip_for_coverage, skip_for_wormhole
from fuser.config_parser import FUSER_CONFIG_DIR
from fuser.sweep import collect_fuser_cases
from helpers.llk_params import PERF_RUN_TYPES_QUASAR

yaml_files = sorted(FUSER_CONFIG_DIR.glob("*.yaml"))
yaml_files += sorted((FUSER_CONFIG_DIR / "quasar").glob("*.yaml"))
case_configs = collect_fuser_cases(yaml_files)


@skip_for_blackhole
@skip_for_wormhole
@skip_for_coverage
@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize(
    "run_type", PERF_RUN_TYPES_QUASAR[0], ids=lambda mode: mode.name
)
@pytest.mark.parametrize("case_name", case_configs)
def test_fuser(
    case_name,
    run_type,
    regenerate_cpp,
    testrun_uid,
):
    config = case_configs[case_name]
    config.global_config.regenerate_cpp = regenerate_cpp
    config.run_perf_test(run_type, session_id=testrun_uid)
