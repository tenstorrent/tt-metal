# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import dataclasses
import pprint
from types import ModuleType


from loguru import logger
import pytest

import ttnn


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, ModuleType):
        val = val.__name__
    return f"{argname}={val}"


@pytest.fixture(autouse=True)
def pre_and_post():
    try:
        ttnn._tt_lib.operations.clear_operation_history()
    except:
        ...
    ttnn.load_config_from_json_file(ttnn.CONFIG_PATH)
    ttnn.load_config_from_dictionary(json.loads(ttnn.CONFIG_OVERRIDES))
    logger.debug(f"ttnn.CONFIG:\n{pprint.pformat(dataclasses.asdict(ttnn.CONFIG))}")
    if ttnn.CONFIG.delete_reports_on_start:
        ttnn.database.delete_reports()
    yield
    ttnn.tracer.disable_tracing()
