# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

from models.experimental.glm4_moe_lite.tt import weights as glm_weights


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_find_missing_shards_reports_missing(tmp_path: Path) -> None:
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir(parents=True, exist_ok=True)

    shard_a = "model-00001-of-00002.safetensors"
    shard_b = "model-00002-of-00002.safetensors"
    _write_index(model_dir, {"w1": shard_a, "w2": shard_b})

    # Only create shard_a.
    safetensors.torch.save_file({"w1": torch.zeros(1)}, str(model_dir / shard_a))

    missing = glm_weights.find_missing_shards(model_dir)
    assert missing == [shard_b]


def test_resolve_best_effort_snapshot_dir_prefers_snapshot_with_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create a fake HF cache with two snapshots for the same model id.
    hf_home = tmp_path / "hf"
    model_id = "zai-org/GLM-4.7-Flash"
    repo_dir = hf_home / "hub" / "models--zai-org--GLM-4.7-Flash"
    snapshots_dir = repo_dir / "snapshots"
    snap_a = snapshots_dir / "aaaa"
    snap_b = snapshots_dir / "bbbb"
    snap_a.mkdir(parents=True, exist_ok=True)
    snap_b.mkdir(parents=True, exist_ok=True)

    # Snapshot A: index-only (no weights)
    _write_index(snap_a, {"w1": "model-00001-of-00001.safetensors"})

    # Snapshot B: index + one shard file
    shard = "model-00001-of-00001.safetensors"
    _write_index(snap_b, {"w1": shard})
    safetensors.torch.save_file({"w1": torch.zeros(1)}, str(snap_b / shard))

    monkeypatch.setenv("HF_HOME", str(hf_home))
    resolved = glm_weights.resolve_best_effort_snapshot_dir(model_id)
    assert resolved == snap_b


def test_load_glm_lazy_state_dict_applies_layer_filter(tmp_path: Path) -> None:
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir(parents=True, exist_ok=True)

    shard = model_dir / "model-00001-of-00001.safetensors"
    keys = {
        "model.layers.0.foo": torch.randn(1),
        "model.layers.1.foo": torch.randn(1),
        "model.layers.2.foo": torch.randn(1),
        "lm_head.weight": torch.randn(2, 2),
    }
    safetensors.torch.save_file(keys, str(shard))
    _write_index(model_dir, {k: shard.name for k in keys.keys()})

    view = glm_weights.load_glm_lazy_state_dict(model_dir, num_layers=2)
    assert "model.layers.0.foo" in view
    assert "model.layers.1.foo" in view
    assert "lm_head.weight" in view
    assert "model.layers.2.foo" not in view

    _ = view["model.layers.0.foo"]
    _ = view["lm_head.weight"]
