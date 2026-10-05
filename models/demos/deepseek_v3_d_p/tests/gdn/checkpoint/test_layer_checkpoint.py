# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for loading one GDN layer from an indexed safetensor checkpoint (synthetic tiny checkpoints)."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import qwen_model_config
from models.demos.deepseek_v3_d_p.reference.gdn.tests.helpers import TINY, random_weights
from models.demos.deepseek_v3_d_p.tests.gdn.checkpoint_utils import gdn_layer_prefix, load_gdn_layer_state_dict

_NESTED = ("qwen38_27b", "model.language_model.")  # text_config present
_FLAT = ("qwen38_2_4t", "model.")  # 2.4T: flat text-only config


def _tiny_model_config(model: str) -> dict:
    """A pinned config.json with its GDN fields shrunk to TINY, keeping the model's nesting and model_type."""
    model_config = qwen_model_config(model)
    text = model_config.get("text_config", model_config)
    text.update(
        hidden_size=TINY.hidden_size,
        linear_num_key_heads=TINY.num_key_heads,
        linear_num_value_heads=TINY.num_value_heads,
        linear_key_head_dim=TINY.head_k_dim,
        linear_value_head_dim=TINY.head_v_dim,
    )
    assert GDNConfig.from_model_config(model_config) == TINY
    return model_config


def _write_checkpoint(directory: Path, model: str, root: str, layers: dict[int, dict[str, torch.Tensor]]) -> None:
    """One shard per layer, a full index, and the tiny config.json."""
    weight_map = {}
    for layer_idx, weights in layers.items():
        shard = f"model-{layer_idx:05d}.safetensors"
        tensors = {gdn_layer_prefix(layer_idx, root) + name: tensor.contiguous() for name, tensor in weights.items()}
        tensors[f"{root}layers.{layer_idx}.input_layernorm.weight"] = torch.ones(TINY.hidden_size)
        save_file(tensors, directory / shard)
        weight_map |= {key: shard for key in tensors}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps(_tiny_model_config(model)))


@pytest.mark.parametrize("model, root", [_NESTED, _FLAT], ids=["nested-language_model-root", "flat-model-root"])
def test_loads_one_layer_under_the_config_root(tmp_path: Path, model: str, root: str) -> None:
    """Layers 1 and 10 share the 'layers.1' prefix text; the loader must take exactly layer 1."""
    layer_1, layer_10 = random_weights(TINY, seed=1), random_weights(TINY, seed=10)
    _write_checkpoint(tmp_path, model, root, {1: layer_1, 10: layer_10})
    loaded = load_gdn_layer_state_dict(tmp_path, 1, TINY)
    assert set(loaded) == set(layer_1)
    for name, tensor in layer_1.items():
        assert torch.equal(loaded[name], tensor) and loaded[name].dtype == tensor.dtype, name


def test_root_follows_config_shape(tmp_path: Path, expect_error) -> None:
    """A flat config with nested keys (or the reverse) finds no layer keys."""
    _write_checkpoint(tmp_path, _FLAT[0], _NESTED[1], {0: random_weights(TINY)})
    with expect_error(ValueError, "missing GDN weights"):
        load_gdn_layer_state_dict(tmp_path, 0, TINY)


@pytest.mark.parametrize(
    "extra",
    ["in_proj_qkvz.weight", "in_proj_ba.weight", "in_proj_qkv.weight_scale_inv"],
    ids=["qwen3_next_qkvz", "qwen3_next_ba", "block_scale"],
)
def test_rejects_unmodeled_layer_keys(tmp_path: Path, extra: str, expect_error) -> None:
    weights = random_weights(TINY) | {extra: torch.zeros(4)}
    _write_checkpoint(tmp_path, *_NESTED, {0: weights})
    with expect_error(ValueError, f"does not model: \\['{extra}'\\]"):
        load_gdn_layer_state_dict(tmp_path, 0, TINY)


def test_rejects_missing_layer_key(tmp_path: Path, expect_error) -> None:
    weights = random_weights(TINY)
    del weights["in_proj_z.weight"]
    _write_checkpoint(tmp_path, *_NESTED, {0: weights})
    with expect_error(ValueError, "missing GDN weights: \\['in_proj_z.weight'\\]"):
        load_gdn_layer_state_dict(tmp_path, 0, TINY)


def test_rejects_missing_shard(tmp_path: Path, expect_error) -> None:
    _write_checkpoint(tmp_path, *_NESTED, {0: random_weights(TINY)})
    (tmp_path / "model-00000.safetensors").unlink()
    with expect_error(FileNotFoundError, "model-00000.safetensors"):
        load_gdn_layer_state_dict(tmp_path, 0, TINY)


def test_rejects_wrong_shape_and_other_config(tmp_path: Path, expect_error) -> None:
    weights = random_weights(TINY)
    weights["A_log"] = torch.zeros(TINY.num_value_heads + 1)
    _write_checkpoint(tmp_path, *_NESTED, {0: weights})
    with expect_error(ValueError, "A_log shape"):
        load_gdn_layer_state_dict(tmp_path, 0, TINY)
    other = GDNConfig(**(asdict(TINY) | {"output_gate_activation": "sigmoid"}))
    with expect_error(ValueError, "another GDN layer"):
        load_gdn_layer_state_dict(tmp_path, 0, other)
