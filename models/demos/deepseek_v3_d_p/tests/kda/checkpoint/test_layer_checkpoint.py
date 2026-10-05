# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for loading one KDA layer from an indexed safetensor checkpoint."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.glm_5_3_flash_config import (
    glm_5_3_flash_kda_config,
    glm_5_3_flash_model_config,
)
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import required_kda_weight_names
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import (
    GLM_5_3_FLASH_FIRST_KDA_LAYER,
    GLM_5_3_FLASH_LAYER_0_SHA256,
    GLM_5_3_FLASH_ROOT,
    KIMI_K3_BARE_ROOT,
    KIMI_K3_WRAPPED_ROOT,
    kda_layer_prefix,
    kda_state_dict_sha256,
    load_kda_layer_state_dict,
    resolve_kda_layer_shards,
    resolve_model_root,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_small_kda_test_config, random_weights


def _write_indexed_layer(
    checkpoint_dir: Path, layer_idx: int, config: KDAConfig, model_root: str = KIMI_K3_WRAPPED_ROOT
) -> Path:
    shard_name = "model-00001-of-00001.safetensors"
    prefix = kda_layer_prefix(layer_idx, model_root)
    weights = {f"{prefix}{name}": tensor.contiguous() for name, tensor in random_weights(config).items()}
    save_file(weights, checkpoint_dir / shard_name)
    index = {"weight_map": {name: shard_name for name in weights}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return checkpoint_dir / shard_name


def test_loads_one_indexed_full_rank_kda_layer(tmp_path: Path) -> None:
    config = make_small_kda_test_config(use_full_rank_gate=True)
    shard = _write_indexed_layer(tmp_path, layer_idx=1, config=config)

    assert resolve_kda_layer_shards(tmp_path, 1, config) == (shard,)
    actual = load_kda_layer_state_dict(tmp_path, 1, config)

    assert set(actual) == set(required_kda_weight_names(config))
    assert "g_proj.weight" in actual
    assert "g_a_proj.weight" not in actual
    assert actual["A_log"].shape == (1, 1, config.num_heads, 1)


def test_rejects_incomplete_checkpoint_shard_set(tmp_path: Path, expect_error) -> None:
    config = make_small_kda_test_config(use_full_rank_gate=True)
    _write_indexed_layer(tmp_path, layer_idx=1, config=config).unlink()

    with expect_error(FileNotFoundError, "missing complete KDA checkpoint shard"):
        resolve_kda_layer_shards(tmp_path, 1, config)


def test_rejects_index_missing_required_kda_weight(tmp_path: Path, expect_error) -> None:
    config = make_small_kda_test_config(use_full_rank_gate=True)
    _write_indexed_layer(tmp_path, layer_idx=1, config=config)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"][f"{kda_layer_prefix(1)}g_proj.weight"]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with expect_error(ValueError, "g_proj.weight"):
        resolve_kda_layer_shards(tmp_path, 1, config)


@pytest.mark.parametrize(
    "model_root, use_full_rank_gate",
    [
        pytest.param(KIMI_K3_WRAPPED_ROOT, True, id="kimi_k3_wrapped"),
        pytest.param(KIMI_K3_BARE_ROOT, True, id="kimi_k3_bare"),
        pytest.param(GLM_5_3_FLASH_ROOT, False, id="glm_5_3_flash"),
    ],
)
def test_resolves_each_known_model_root(tmp_path: Path, model_root: str, use_full_rank_gate: bool) -> None:
    config = make_small_kda_test_config(use_full_rank_gate=use_full_rank_gate)
    _write_indexed_layer(tmp_path, layer_idx=0, config=config, model_root=model_root)

    assert resolve_model_root(tmp_path) == model_root
    assert set(load_kda_layer_state_dict(tmp_path, 0, config)) == set(required_kda_weight_names(config))


def test_loads_glm_5_3_flash_layer_0(glm_5_3_flash_checkpoint_dir: Path) -> None:
    """Real GLM-5.3-Flash layer 0 (local only): low-rank KDA weights load, normalize and match the pin."""
    checkpoint_config = json.loads((glm_5_3_flash_checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    assert checkpoint_config == glm_5_3_flash_model_config(), "checkpoint config.json differs from the pinned copy"
    config = glm_5_3_flash_kda_config()
    assert resolve_model_root(glm_5_3_flash_checkpoint_dir) == GLM_5_3_FLASH_ROOT

    state_dict = load_kda_layer_state_dict(glm_5_3_flash_checkpoint_dir, GLM_5_3_FLASH_FIRST_KDA_LAYER, config)

    assert set(state_dict) == set(required_kda_weight_names(config))
    assert "g_a_proj.weight" in state_dict and "g_proj.weight" not in state_dict
    assert state_dict["A_log"].shape == (1, 1, 64, 1)
    # Checkpoint dtypes: decay parameters F32, everything else BF16 (conv and o_norm too, unlike K3's F32).
    assert {name for name, tensor in state_dict.items() if tensor.dtype == torch.float32} == {"A_log", "dt_bias"}
    assert all(t.dtype in (torch.float32, torch.bfloat16) for t in state_dict.values())
    assert kda_state_dict_sha256(state_dict) == GLM_5_3_FLASH_LAYER_0_SHA256
