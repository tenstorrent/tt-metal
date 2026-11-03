# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint identity and public server configuration; no device or weights required."""

import pytest

from models.experimental.qwen_image_2_1.common.config import HF_REVISION, snapshot_dir
from models.experimental.qwen_image_2_1.server.app import load_config


def test_missing_pinned_snapshot_does_not_select_another_revision(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_IMAGE_SNAPSHOT", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snapshots = tmp_path / "models--Qwen--Qwen-Image-2.1" / "snapshots"
    (snapshots / "unrelated-revision").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=HF_REVISION):  # allow-pytest.raises: runs without repository conftest.
        snapshot_dir()
    expected = snapshots / HF_REVISION
    expected.mkdir()
    assert snapshot_dir() == str(expected)


def test_explicit_snapshot_must_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_IMAGE_SNAPSHOT", str(tmp_path / "missing"))
    with pytest.raises(  # allow-pytest.raises: runs without repository conftest.
        FileNotFoundError, match="QWEN_IMAGE_SNAPSHOT"
    ):
        snapshot_dir()
    monkeypatch.setenv("QWEN_IMAGE_SNAPSHOT", str(tmp_path))
    assert snapshot_dir() == str(tmp_path)


@pytest.mark.parametrize(
    "name,value",
    [
        ("QWEN_IMAGE_DIT_DTYPE", "bf61"),
        ("QWEN_IMAGE_TE_DTYPE", "int8"),
        ("QWEN_IMAGE_EDITING", "yes"),
        ("QWEN_IMAGE_ETH_DISPATCH", "1"),
        ("QWEN_IMAGE_STEPS", "0"),
        ("QWEN_IMAGE_STEPS", "1"),
        ("QWEN_IMAGE_STEPS", "101"),
        ("QWEN_IMAGE_SIZE", "0"),
        ("TT_WEIGHTS_REVISION", "unrelated-revision"),
    ],
)
def test_invalid_server_configuration_fails_before_device_open(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):  # allow-pytest.raises: no repository conftest.
        load_config()


@pytest.mark.parametrize("editing", ["0", "1"])
def test_server_defaults_to_tensix_dispatch(editing, monkeypatch):
    monkeypatch.delenv("QWEN_IMAGE_ETH_DISPATCH", raising=False)
    monkeypatch.setenv("QWEN_IMAGE_EDITING", editing)
    assert load_config()["eth_dispatch"] is False


def test_checkpoint_rejects_ambiguous_safetensors_indices(tmp_path):
    from models.experimental.qwen_image_2_1.common.weights import LazyCheckpoint

    folder = tmp_path / "vae"
    folder.mkdir()
    for name in ("first", "second"):
        (folder / f"{name}.safetensors.index.json").write_text('{"weight_map": {}}')
    with pytest.raises(ValueError, match="expected one safetensors index"):  # allow-pytest.raises: host-only test.
        LazyCheckpoint("vae", str(tmp_path))
