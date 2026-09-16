# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

FUSER_TEST_FILES = {
    "test_fused.py",
    "perf_fused.py",
    "test_fused_quasar.py",
    "perf_fused_quasar.py",
}


def expand_fuser_selector(selector):
    filename, separator, test = selector.partition("::test_fuser[")
    if (
        not separator
        or Path(filename).name not in FUSER_TEST_FILES
        or not test.endswith("]")
    ):
        return [selector]

    from .config_parser import FuserConfigSchema
    from .sweep import expand_fuser_configs

    test_name = test[:-1]
    try:
        definition = FuserConfigSchema.load_definition(test_name)
    except FileNotFoundError:
        return [selector]

    return [
        f"{filename}::test_fuser[{case_name}]"
        for case_name, _ in expand_fuser_configs(test_name, definition)
    ]


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    config.args[:] = [
        expanded
        for selector in config.args
        for expanded in expand_fuser_selector(selector)
    ]
