# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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


@pytest.fixture
def pcc_thresholds():
    """The PCC floors of the tiered unit tests, keyed by test function name (pcc_thresholds.json)."""

    return json.loads(Path(__file__).with_name("pcc_thresholds.json").read_text())


def _checkpoint_root(unset) -> Path:
    """The checkpoint directory ``MODEL_WEIGHTS_DIR`` or ``QWEN38_CHECKPOINT`` names, or ``HF_MODEL``'s snapshot in the
    local Hugging Face cache (offline).  A named path that is not a directory fails the test; with no variable set,
    ``unset`` (``pytest.skip`` or ``pytest.fail``) gets the message."""

    for name in ("MODEL_WEIGHTS_DIR", "QWEN38_CHECKPOINT"):
        if os.environ.get(name):
            root = Path(os.environ[name])
            if not root.is_dir():
                pytest.fail(f"{name}={os.environ[name]} is not a checkpoint directory")
            return root
    if os.environ.get("HF_MODEL"):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(os.environ["HF_MODEL"], local_files_only=True))
    unset("set QWEN38_CHECKPOINT (or MODEL_WEIGHTS_DIR) to the checkpoint directory, or HF_MODEL to its id")


@pytest.fixture(scope="session")
def checkpoint_dir() -> Path:
    """The checkpoint of the no-device tests that read one: skipped when no variable names it."""

    return _checkpoint_root(pytest.skip)


@pytest.fixture(scope="session")
def device_checkpoint_dir() -> Path:
    """The checkpoint of the device tests: failed when no variable names it."""

    return _checkpoint_root(pytest.fail)


@pytest.fixture
def tp_harness(mesh_device, tmp_path, device_checkpoint_dir):
    """The device-side harness on the root ``mesh_device`` fixture; imported here so no-device tests load without ttnn."""

    from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import Qwen38TPHarness

    return Qwen38TPHarness(mesh_device, tmp_path, device_checkpoint_dir)
