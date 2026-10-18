# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Validation shared by the reusable NLLB API and command line."""

from collections.abc import Mapping
from numbers import Integral
import numpy as np
import torch


def _unique_checkpoint_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint index key: {key}")
        result[key] = value
    return result


def _read_checkpoint(path):
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(weights, Mapping):
        raise ValueError(f"Checkpoint must be a state dictionary: {path}")
    for name, value in weights.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint requires named tensors: {path}")
    return weights


def load_checkpoint(weights_path):
    """Load materialized PyTorch files/shards on CPU, each resolved shard once."""
    from pathlib import Path
    import json

    path = Path(weights_path)
    if path.is_file():
        return _read_checkpoint(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    index = path / "pytorch_model.bin.index.json"
    if not index.is_file():
        return _read_checkpoint(path / "pytorch_model.bin")
    document = json.loads(index.read_text(), object_pairs_hook=_unique_checkpoint_object)
    if not isinstance(document, dict):
        raise ValueError("Checkpoint index must be an object")
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index requires a nonempty weight_map")
    shards = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name or not isinstance(shard, str) or not shard:
            raise ValueError("Checkpoint index names must be nonempty strings")
        resolved = (path / shard).resolve()
        if not resolved.is_relative_to(path.resolve()):
            raise ValueError("Checkpoint shard must remain inside checkpoint directory")
        shards.setdefault(resolved, []).append(name)
    weights = {}
    for shard, names in shards.items():
        values = _read_checkpoint(shard)
        extras = set(values) - set(names)
        if extras:
            raise ValueError(f"Unindexed weights in {shard.name}: {sorted(extras)}")
        for name in names:
            if name not in values:
                raise ValueError(f"Indexed weight {name} missing from {shard.name}")
            weights[name] = values[name]
        del values
    return weights


def integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return int(value)


def validate_config(config):
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    c = dict(config)
    for key in (
        "d_model",
        "vocab_size",
        "encoder_layers",
        "decoder_layers",
        "encoder_attention_heads",
        "decoder_attention_heads",
        "encoder_ffn_dim",
        "decoder_ffn_dim",
        "max_position_embeddings",
    ):
        if key not in c:
            raise ValueError(f"Missing config field: {key}")
        c[key] = integer(c[key], key, 1, 2**31 - 1)
    if c["d_model"] < 4 or c["vocab_size"] < 3:
        raise ValueError("Model dimension must be >=4 and vocabulary >=3")
    for side in ("encoder", "decoder"):
        if c["d_model"] % c[side + "_attention_heads"]:
            raise ValueError(f"d_model must be divisible by {side}_attention_heads")
    required = {
        "model_type": "m2m_100",
        "activation_function": "relu",
        "is_encoder_decoder": True,
        "tie_word_embeddings": True,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "decoder_start_token_id": 2,
    }
    for key, expected in required.items():
        value = c.get(key, expected)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Unsupported {key}: expected {expected!r}")
        c[key] = value
    if not isinstance(c.get("scale_embedding", True), bool):
        raise ValueError("scale_embedding must be boolean")
    if c.get("position_embedding_type", "sinusoidal") != "sinusoidal":
        raise ValueError("Only sinusoidal positions are supported")
    return c


def checkpoint_shapes(config):
    c = validate_config(config)
    d, v = c["d_model"], c["vocab_size"]
    shapes = {"model.shared.weight": (v, d)}
    for side in ("encoder", "decoder"):
        root = "model." + side
        shapes[root + ".layer_norm.weight"] = (d,)
        shapes[root + ".layer_norm.bias"] = (d,)
        for i in range(c[side + "_layers"]):
            p = f"{root}.layers.{i}"
            attentions = ["self_attn"] + (["encoder_attn"] if side == "decoder" else [])
            for attention in attentions:
                for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    shapes[f"{p}.{attention}.{projection}.weight"] = (d, d)
                    shapes[f"{p}.{attention}.{projection}.bias"] = (d,)
                shapes[f"{p}.{attention}_layer_norm.weight"] = (d,)
                shapes[f"{p}.{attention}_layer_norm.bias"] = (d,)
            f = c[side + "_ffn_dim"]
            shapes[p + ".fc1.weight"] = (f, d)
            shapes[p + ".fc1.bias"] = (f,)
            shapes[p + ".fc2.weight"] = (d, f)
            shapes[p + ".fc2.bias"] = (d,)
            shapes[p + ".final_layer_norm.weight"] = (d,)
            shapes[p + ".final_layer_norm.bias"] = (d,)
    return shapes


def validate_checkpoint(weights, config):
    if not isinstance(weights, Mapping):
        raise ValueError("Checkpoint must be a state dictionary")
    expected = checkpoint_shapes(config)
    for name, shape in expected.items():
        value = weights.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Checkpoint tensor {name} must have shape {shape}")
        if not value.is_floating_point() or value.device.type != "cpu":
            raise ValueError(f"Checkpoint tensor {name} must be floating point on CPU")
    shared = weights["model.shared.weight"]
    aliases = {"model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"}
    for name in aliases & weights.keys():
        value = weights[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != shared.dtype
            or value.shape != shared.shape
            or not torch.equal(value, shared)
        ):
            raise ValueError(f"Tied checkpoint tensor {name} disagrees with shared embedding")
    # Official sinusoidal modules may persist only an empty dtype marker.
    markers = {"model.encoder.embed_positions._float_tensor", "model.decoder.embed_positions._float_tensor"}
    for name in weights.keys() - expected.keys() - aliases:
        if name not in markers or not isinstance(weights[name], torch.Tensor) or weights[name].numel() != 1:
            raise ValueError(f"Unsupported checkpoint tensor: {name}")


def token_array(value, name, vocab, max_length, batch=None):
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-two NumPy integer array")
    if value.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer token IDs")
    if not 1 <= value.shape[0] <= 4 or not 1 <= value.shape[1] <= max_length:
        raise ValueError(f"{name} requires batch 1..4 and length 1..{max_length}")
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f"{name} batch does not match input_ids")
    if np.any(value < 0) or np.any(value >= vocab):
        raise ValueError(f"{name} contains token IDs outside vocabulary")


def validate_inputs(input_ids, attention_mask, config):
    token_array(input_ids, "input_ids", config["vocab_size"], min(256, config["max_position_embeddings"]))
    if (
        not isinstance(attention_mask, np.ndarray)
        or attention_mask.shape != input_ids.shape
        or attention_mask.dtype.kind not in "biu"
    ):
        raise ValueError("attention_mask must be a matching NumPy integer/bool array")
    if not np.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must be binary")
    if not np.all(np.any(attention_mask, axis=1)):
        raise ValueError("Each source row must have an unmasked token")


def validate_checkpoint_config(weights_path, config):
    """Compare adjacent official metadata when present, including non-shape semantics."""
    import json
    from pathlib import Path

    path = Path(weights_path)
    metadata = (path if path.is_dir() else path.parent) / "config.json"
    if not metadata.is_file():
        return  # Explicit configuration remains required for a standalone .bin.
    official = validate_config(json.loads(metadata.read_text()))
    keys = (
        "d_model",
        "vocab_size",
        "encoder_layers",
        "decoder_layers",
        "encoder_attention_heads",
        "decoder_attention_heads",
        "encoder_ffn_dim",
        "decoder_ffn_dim",
        "max_position_embeddings",
        "pad_token_id",
        "eos_token_id",
        "decoder_start_token_id",
        "activation_function",
        "model_type",
        "is_encoder_decoder",
        "tie_word_embeddings",
        "scale_embedding",
    )
    for key in keys:
        if config.get(key, True) != official.get(key, True):
            raise ValueError(f"Checkpoint config mismatch: {key}")
