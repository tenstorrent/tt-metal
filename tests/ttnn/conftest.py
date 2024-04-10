# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import json
import dataclasses
import pprint
from types import ModuleType

from loguru import logger

from typing import Dict
from pytest import StashKey, CollectReport

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


call_report_key = StashKey[CollectReport]()


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    rep = yield

    # store test result for only call phase of a call
    # otherphases: "setup", "call", "teardown"
    if rep.when == "call":
        item.stash[call_report_key] = rep

    return rep


def pytest_addoption(parser):
    parser.addoption(
        "--enable-gihub-issue-creation",
        action="store_const",
        const=True,
        help="Enable creation of github issues for marked tests",
    )


# import sys
from tests.ttnn.github_automatic_issue import create_or_update_pytest_issue


@pytest.fixture
def github_issue(request):
    gh_issue = {"full_test_name": request.node.nodeid}

    yield gh_issue
    # captured = capsys.readouterr()
    # gh_issue["body"] = f"caplog.txt:\n```{caplog.txt}\n```\nstdout:\n```\n{captured.out}\n```\nstderr:\n```\n{captured.err}\n```"
    # gh_issue["body"] = f"```\nstdout:\n```\n{captured.out}\n```\nstderr:\n```\n{captured.err}\n```"
    if (call_report_key not in request.node.stash) or request.node.stash[call_report_key].failed:
        if request.config.getoption("--enable-gihub-issue-creation"):
            create_or_update_pytest_issue(gh_issue)
        else:
            logger.warning(
                "WARNING: pytest failure of github_issue marked test without --enable-gihub-issue-creation flag. No github issues will be created or updated"
            )
