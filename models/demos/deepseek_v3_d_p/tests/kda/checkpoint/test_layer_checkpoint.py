# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for loading one KDA layer from an indexed safetensor checkpoint."""

import json
from pathlib import Path

from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import required_kda_weight_names
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import (
    kda_layer_prefix,
    load_kda_layer_state_dict,
    resolve_kda_layer_shards,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_config, random_weights


def _write_indexed_layer(checkpoint_dir: Path, layer_idx: int, config: KDAConfig) -> Path:
    shard_name = "model-00001-of-00001.safetensors"
    prefix = kda_layer_prefix(layer_idx)
    weights = {f"{prefix}{name}": tensor.contiguous() for name, tensor in random_weights(config).items()}
    save_file(weights, checkpoint_dir / shard_name)
    index = {"weight_map": {name: shard_name for name in weights}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return checkpoint_dir / shard_name


def test_loads_one_indexed_full_rank_kda_layer(tmp_path: Path) -> None:
    config = make_config(use_full_rank_gate=True)
    shard = _write_indexed_layer(tmp_path, layer_idx=1, config=config)

    assert resolve_kda_layer_shards(tmp_path, 1, config) == (shard,)
    actual = load_kda_layer_state_dict(tmp_path, 1, config)

    assert set(actual) == set(required_kda_weight_names(config))
    assert "g_proj.weight" in actual
    assert "g_a_proj.weight" not in actual
    assert actual["A_log"].shape == (1, 1, config.num_heads, 1)


def test_rejects_incomplete_checkpoint_shard_set(tmp_path: Path, expect_error) -> None:
    config = make_config(use_full_rank_gate=True)
    _write_indexed_layer(tmp_path, layer_idx=1, config=config).unlink()

    with expect_error(FileNotFoundError, "missing complete KDA checkpoint shard"):
        resolve_kda_layer_shards(tmp_path, 1, config)


def test_rejects_index_missing_required_kda_weight(tmp_path: Path, expect_error) -> None:
    config = make_config(use_full_rank_gate=True)
    _write_indexed_layer(tmp_path, layer_idx=1, config=config)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"][f"{kda_layer_prefix(1)}g_proj.weight"]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with expect_error(ValueError, "g_proj.weight"):
        resolve_kda_layer_shards(tmp_path, 1, config)
