# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from models.demos.wormhole.patchtst.scripts.download_assets import _resolve_dataset_path, _unlink_dataset_path


def test_resolve_dataset_path_allows_descendants(tmp_path: Path):
    destination = _resolve_dataset_path(tmp_path, Path("Heartbeat/Heartbeat_TRAIN.ts"))
    assert destination == (tmp_path / "Heartbeat/Heartbeat_TRAIN.ts").resolve()


def test_resolve_dataset_path_rejects_parent_traversal(tmp_path: Path):
    with pytest.raises(RuntimeError, match="escapes dataset root"):
        _resolve_dataset_path(tmp_path, Path("../outside.txt"))


def test_unlink_dataset_path_removes_descendant_file(tmp_path: Path):
    archive_path = tmp_path / "heartbeat_cls.zip"
    archive_path.write_bytes(b"payload")

    _unlink_dataset_path(tmp_path, Path("heartbeat_cls.zip"))

    assert not archive_path.exists()


def test_unlink_dataset_path_rejects_parent_traversal(tmp_path: Path):
    with pytest.raises(RuntimeError, match="escapes dataset root"):
        _unlink_dataset_path(tmp_path, Path("../outside.zip"))
