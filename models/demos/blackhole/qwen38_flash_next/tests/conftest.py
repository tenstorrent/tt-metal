# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import re
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _expect_error(expected_exception, match=None):
    try:
        yield
    except expected_exception as exception:
        if match is not None and re.search(match, str(exception)) is None:
            raise AssertionError(f"Exception message did not match {match!r}: {exception}") from exception
    else:
        raise AssertionError(f"Expected {expected_exception} to be raised")


@pytest.fixture
def expect_error():
    return _expect_error


@pytest.fixture
def pcc_thresholds():
    """The PCC floors of the tiered unit tests, keyed by test function name (pcc_thresholds.json)."""

    return json.loads(Path(__file__).with_name("pcc_thresholds.json").read_text())


@pytest.fixture
def tp_harness(mesh_device, tmp_path):
    """The device-side harness on the root ``mesh_device`` fixture; imported here so no-device tests load without ttnn."""

    from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import Qwen38TPHarness

    return Qwen38TPHarness(mesh_device, tmp_path)
