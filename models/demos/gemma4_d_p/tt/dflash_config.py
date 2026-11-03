# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint configuration and selective weight loading for DFlash prefill."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DFLASH_MODEL = "z-lab/gemma-4-31B-it-DFlash"


@dataclass(frozen=True)
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    target_layer_ids: tuple[int, ...]
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    @classmethod
    def from_dict(cls, data):
        if data["model_type"] != "qwen3" or data.get("attention_bias", False) or data.get("rope_scaling"):
            raise ValueError("DFlash prefill requires Qwen3 projections without bias or RoPE scaling")
        config = cls(
            hidden_size=data["hidden_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            target_layer_ids=tuple(data["dflash_config"]["target_layer_ids"]),
            rms_norm_eps=data["rms_norm_eps"],
            rope_theta=data["rope_theta"],
            max_position_embeddings=data["max_position_embeddings"],
        )
        if min(config.hidden_size, config.num_hidden_layers, config.num_key_value_heads, config.head_dim) <= 0:
            raise ValueError("DFlash dimensions must be positive")
        ids = config.target_layer_ids
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("DFlash target layer IDs must be nonempty, unique, and nonnegative")
        if config.rms_norm_eps <= 0 or config.rope_theta <= 0 or config.max_position_embeddings <= 0:
            raise ValueError("DFlash norm epsilon, RoPE theta, and context capacity must be positive")
        return config

    def validate(self, mesh_config, max_seq_len, *, hidden_size=None, num_layers=None):
        if self.hidden_size % (mesh_config.tp_degree * 32) or self.head_dim % 32:
            raise ValueError("DFlash hidden shards and head dimensions must contain whole tiles")
        if self.num_key_value_heads % mesh_config.tp_degree:
            raise ValueError("DFlash KV heads must divide evenly across TP")
        if not 0 < max_seq_len <= self.max_position_embeddings or max_seq_len % (mesh_config.cp_degree * 32):
            raise ValueError("DFlash sequence capacity must fit its RoPE tables and contain whole CP-local tiles")
        if hidden_size is not None and hidden_size != self.hidden_size:
            raise ValueError(f"DFlash expects target hidden size {self.hidden_size}, got {hidden_size}")
        if num_layers is not None and max(self.target_layer_ids) >= num_layers:
            raise ValueError(f"DFlash requires target layer {max(self.target_layer_ids)}, got {num_layers} layers")

    def weight_shapes(self):
        h = self.hidden_size
        shapes = {"fc.weight": (h, len(self.target_layer_ids) * h), "hidden_norm.weight": (h,)}
        for layer in range(self.num_hidden_layers):
            prefix = f"layers.{layer}.self_attn"
            shapes[f"{prefix}.k_proj.weight"] = (self.num_key_value_heads * self.head_dim, h)
            shapes[f"{prefix}.v_proj.weight"] = (self.num_key_value_heads * self.head_dim, h)
            shapes[f"{prefix}.k_norm.weight"] = (self.head_dim,)
        return shapes


def _checkpoint_files(checkpoint, *, weights=False):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HOME

    path = Path(checkpoint)
    if path.is_dir():
        return path / "config.json", path / "model.safetensors"
    cache_dir = Path(os.environ.get("HF_HOME", HF_HOME)) / "hub"
    config_file = Path(hf_hub_download(checkpoint, "config.json", cache_dir=cache_dir))
    weights_file = config_file.parent / "model.safetensors"
    if weights:
        weights_file = Path(
            hf_hub_download(checkpoint, "model.safetensors", revision=config_file.parent.name, cache_dir=cache_dir)
        )
    return config_file, weights_file


def load_dflash_config(checkpoint=DEFAULT_DFLASH_MODEL):
    config_file, _ = _checkpoint_files(checkpoint)
    return DFlashConfig.from_dict(json.loads(config_file.read_text()))


def load_dflash_weights(checkpoint=DEFAULT_DFLASH_MODEL):
    from safetensors import safe_open

    config_file, weights_file = _checkpoint_files(checkpoint, weights=True)
    config = DFlashConfig.from_dict(json.loads(config_file.read_text()))
    state = {}
    with safe_open(weights_file, framework="pt", device="cpu") as reader:
        for key, shape in config.weight_shapes().items():
            if key not in reader.keys() or tuple(reader.get_slice(key).get_shape()) != shape:
                raise ValueError(f"DFlash checkpoint requires {key} with shape {shape}")
            state[key] = reader.get_tensor(key)
    return config, state


def dflash_tensor_cache_path(config, state, mesh_shape):
    """Separate cache keyed by the exact consumed weights and their configuration."""
    import torch
    from huggingface_hub.constants import HF_HOME

    digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode())
    for key in sorted(state):
        weight = state[key].contiguous()
        digest.update(f"{key}:{weight.dtype}".encode())
        digest.update(memoryview(weight.view(torch.uint8).numpy()))
    root = Path(os.environ.get("HF_HOME", HF_HOME)) / "tt_cache/gemma4_d_p/z-lab--gemma-4-31B-it-DFlash"
    path = root / digest.hexdigest()[:16] / f"bf16_mesh{mesh_shape[0]}x{mesh_shape[1]}_v1"
    path.mkdir(parents=True, exist_ok=True)
    return path
