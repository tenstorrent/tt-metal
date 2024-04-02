# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

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
    logger.debug(f"ttnn.CONFIG:\n{pprint.pformat(dataclasses.asdict(ttnn.CONFIG))}")
    ttnn.database.delete_reports()
    yield
