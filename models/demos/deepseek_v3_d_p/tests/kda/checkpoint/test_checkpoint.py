# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for indexed KDA checkpoint loading."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import (
    kda_layer_prefix,
    load_kda_layer_state_dict,
    resolve_kda_layer_shards,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import assert_equal, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.weight_schema import (
    normalize_kda_state_dict,
    required_kda_weight_names,
    validate_kda_weights,
)


def _full_rank_config(*, num_heads: int = 2) -> KDAConfig:
    return KDAConfig(
        hidden_size=64,
        num_heads=num_heads,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
        use_full_rank_gate=True,
        gate_lower_bound=-5.0,
    )


def _write_indexed_layer(checkpoint_dir: Path, layer_idx: int, config: KDAConfig) -> Path:
    shard_name = "model-00001-of-00001.safetensors"
    prefix = kda_layer_prefix(layer_idx)
    weights = {f"{prefix}{name}": tensor.contiguous() for name, tensor in random_weights(config).items()}
    save_file(weights, checkpoint_dir / shard_name)
    index = {"weight_map": {name: shard_name for name in weights}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return checkpoint_dir / shard_name


def test_loads_one_indexed_full_rank_kda_layer(tmp_path: Path) -> None:
    config = _full_rank_config()
    shard = _write_indexed_layer(tmp_path, layer_idx=1, config=config)

    assert resolve_kda_layer_shards(tmp_path, 1, config) == (shard,)
    actual = load_kda_layer_state_dict(tmp_path, 1, config)

    assert set(actual) == set(required_kda_weight_names(config))
    assert "g_proj.weight" in actual
    assert "g_a_proj.weight" not in actual
    assert actual["A_log"].shape == (1, 1, config.num_heads, 1)


def test_rejects_incomplete_checkpoint_shard_set(tmp_path: Path, expect_error) -> None:
    config = _full_rank_config()
    _write_indexed_layer(tmp_path, layer_idx=1, config=config).unlink()

    with expect_error(FileNotFoundError, "missing complete KDA checkpoint shard"):
        resolve_kda_layer_shards(tmp_path, 1, config)


def test_rejects_index_missing_required_kda_weight(tmp_path: Path, expect_error) -> None:
    config = _full_rank_config()
    _write_indexed_layer(tmp_path, layer_idx=1, config=config)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"][f"{kda_layer_prefix(1)}g_proj.weight"]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with expect_error(ValueError, "g_proj.weight"):
        resolve_kda_layer_shards(tmp_path, 1, config)


def test_weight_validation_reports_exact_name_and_shape(expect_error) -> None:
    config = _full_rank_config()
    weights = random_weights(config)
    weights["q_proj.weight"] = torch.empty(config.q_dim, config.hidden_size + 1)

    with expect_error(ValueError, r"q_proj\.weight shape .* !="):
        validate_kda_weights(weights, config)


def test_normalize_state_dict_trims_kimi_k3_padded_a_log() -> None:
    config = _full_rank_config(num_heads=96)
    state_dict = random_weights(config)
    padded = torch.arange(128, dtype=torch.float32)
    state_dict["A_log"] = padded

    normalized = normalize_kda_state_dict(state_dict, config)

    assert normalized["A_log"].shape == (1, 1, config.num_heads, 1)
    assert_equal(padded[: config.num_heads], normalized["A_log"].reshape(-1), name="trimmed A_log")


def test_normalize_state_dict_rejects_unsupported_a_log_padding(expect_error) -> None:
    config = _full_rank_config(num_heads=96)
    state_dict = random_weights(config)
    state_dict["A_log"] = torch.arange(127, dtype=torch.float32)

    with expect_error(ValueError, "A_log has 127 entries"):
        normalize_kda_state_dict(state_dict, config)
