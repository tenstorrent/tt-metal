# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
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


def _link_into_hub_blobs(snapshot: Path) -> None:
    """Lay ``snapshot`` (``<repo>/snapshots/<revision>``) out as the hub cache does: every file moves to
    ``<repo>/blobs/<sha256>`` and leaves a relative link in its place."""

    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir(exist_ok=True)
    for path in sorted(snapshot.iterdir()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        os.replace(path, blobs / digest)
        path.symlink_to(Path("../../blobs") / digest)


@pytest.fixture
def link_into_hub_blobs():
    return _link_into_hub_blobs


@pytest.fixture
def pcc_thresholds():
    """The PCC floors of the tiered unit tests, keyed by test function name (pcc_thresholds.json)."""

    return json.loads(Path(__file__).with_name("pcc_thresholds.json").read_text())


@pytest.fixture
def tp_harness(mesh_device, tmp_path):
    """The device-side harness on the root ``mesh_device`` fixture; imported here so no-device tests load without ttnn."""

    from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import Qwen38TPHarness

    return Qwen38TPHarness(mesh_device, tmp_path)
