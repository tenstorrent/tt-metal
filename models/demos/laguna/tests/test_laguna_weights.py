# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks that checkpoint resolution stays local and happens once per process."""
from __future__ import annotations

import json

import huggingface_hub
import pytest

from models.demos.laguna.tests import laguna_weights as W


@pytest.fixture(autouse=True)
def _fresh_cache():
    W._snapshot_dir.cache_clear()
    W._weight_map.cache_clear()
    yield
    W._snapshot_dir.cache_clear()
    W._weight_map.cache_clear()


def _fake_snapshot(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s1"}}))
    return str(tmp_path)


def test_cached_checkpoint_resolves_locally_once(monkeypatch, tmp_path):
    calls = []

    def fake(repo_id, allow_patterns=None, local_files_only=False):
        calls.append(local_files_only)
        return _fake_snapshot(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    for _ in range(48):  # one call per decoder layer during a full load
        d, wm = W._index()
    assert (d, wm) == (str(tmp_path), {"a": "s1"})
    assert calls == [True], "a cached checkpoint must resolve once, without a network request"


def test_missing_checkpoint_falls_back_to_one_download(monkeypatch, tmp_path):
    calls = []

    def fake(repo_id, allow_patterns=None, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only:
            raise huggingface_hub.errors.LocalEntryNotFoundError("not cached")
        return _fake_snapshot(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    W._index()
    W._index()
    assert calls == [True, False]
