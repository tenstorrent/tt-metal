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
    """The checkpoint directory ``MODEL_WEIGHTS_DIR`` or ``QWEN38_CHECKPOINT`` names, or ``HF_MODEL``'s snapshot at the
    pinned revision in the local Hugging Face cache (offline).  A named path that is not a directory fails the test;
    with no variable set, ``unset`` (``pytest.skip`` or ``pytest.fail``) gets the message."""

    for name in ("MODEL_WEIGHTS_DIR", "QWEN38_CHECKPOINT"):
        if os.environ.get(name):
            root = Path(os.environ[name])
            if not root.is_dir():
                pytest.fail(f"{name}={os.environ[name]} is not a checkpoint directory")
            return root
    if os.environ.get("HF_MODEL"):
        from huggingface_hub import snapshot_download

        from models.demos.blackhole.qwen38_flash_next.checkpoint import PINNED_CHECKPOINT_REVISION

        return Path(
            snapshot_download(os.environ["HF_MODEL"], revision=PINNED_CHECKPOINT_REVISION, local_files_only=True)
        )
    unset("set QWEN38_CHECKPOINT (or MODEL_WEIGHTS_DIR) to the checkpoint directory, or HF_MODEL to its id")


@pytest.fixture(scope="session")
def checkpoint_dir() -> Path:
    """The checkpoint of the no-device tests that read one: skipped when no variable names it."""

    return _checkpoint_root(pytest.skip)


@pytest.fixture(scope="session")
def device_checkpoint_dir() -> Path:
    """The checkpoint of the device tests: failed when no variable names it."""

    return _checkpoint_root(pytest.fail)


@pytest.fixture(scope="module")
def release_checkpoint(checkpoint_dir):
    from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint

    return Qwen38Checkpoint(checkpoint_dir)


@pytest.fixture(scope="module")
def release_placement(release_checkpoint):
    from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement

    return Qwen38Placement(release_checkpoint.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))


@pytest.fixture(scope="module")
def chat_template(checkpoint_dir):
    from models.demos.blackhole.qwen38_flash_next.chat import Qwen38OfficialChatTemplate

    return Qwen38OfficialChatTemplate(checkpoint_dir)


@pytest.fixture
def rows_fake(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import install_rows_fake

    return install_rows_fake(monkeypatch)


@pytest.fixture
def verify_fake(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import install_verify_fake

    return install_verify_fake(monkeypatch)


@pytest.fixture(scope="module")
def gdn_weights():
    """The oracle GDN weights and their device form (helpers/torch_ttnn.py)."""

    from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import device_gdn_weights, gdn_oracle_weights

    oracle = gdn_oracle_weights()
    return oracle, device_gdn_weights(oracle)


@pytest.fixture
def prefill_fakes(monkeypatch):
    from models.demos.blackhole.qwen38_flash_next.tests.helpers.prefill import install_prefill_fakes

    return install_prefill_fakes(monkeypatch)


@pytest.fixture
def tp_harness(mesh_device, tmp_path, device_checkpoint_dir):
    """The device-side harness on the root ``mesh_device`` fixture; imported here so no-device tests load without ttnn."""

    from models.demos.blackhole.qwen38_flash_next.tests.helpers.tp_harness import Qwen38TPHarness

    return Qwen38TPHarness(mesh_device, tmp_path, device_checkpoint_dir)
