# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Loads MoE tensors from the local Mistral Small 4 snapshot (requires shard ``safetensors`` files)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.demos.mistral_small_4_119B.tt.moe.moe import (
    TtMistral4MoE,
    load_ttmistral4_moe_from_sharded_safetensors,
    mistral4_text_config_from_snapshot,
)


@pytest.fixture(scope="session")
def snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "mistral_small_4"


def test_mistral4_text_config_from_snapshot(snapshot_dir: Path):
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    assert snapshot_dir.joinpath("config.json").is_file()
    cfg = mistral4_text_config_from_snapshot(snapshot_dir)
    assert cfg.hidden_size > 0
    assert cfg.n_routed_experts > 0


def test_load_moe_layer0_from_sharded_safetensors(snapshot_dir: Path):
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.skip("No model.safetensors.index.json (snapshot incomplete)")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    gate_key = "language_model.model.layers.0.mlp.gate.weight"
    shard_name = weight_map.get(gate_key)
    if shard_name is None:
        pytest.skip("Checkpoint index missing layer-0 MoE gate key")
    if not (snapshot_dir / shard_name).is_file():
        pytest.skip(f"Shard file not present: {shard_name} (download full weights first)")

    cfg = mistral4_text_config_from_snapshot(snapshot_dir)
    moe = TtMistral4MoE(cfg)
    incomp = load_ttmistral4_moe_from_sharded_safetensors(moe, snapshot_dir, layer_idx=0, strict=False)
    assert incomp.missing_keys == []
    w = moe.gate.weight
    assert w.shape[0] == cfg.n_routed_experts
    assert w.shape[1] == cfg.hidden_size
