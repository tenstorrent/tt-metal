# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Incremental raw-HF checkpoint loading. Q/K conversion belongs to QKVProjection only."""

import json
from pathlib import Path

from safetensors import safe_open

from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import DEFAULT_MAX_SEQ_LEN, PrefillGeometry

LAYER_SHAPES = {
    "input_layernorm.weight": (4096,),
    "self_attn.q_proj.weight": (4096, 4096),
    "self_attn.k_proj.weight": (1024, 4096),
    "self_attn.v_proj.weight": (1024, 4096),
    "self_attn.o_proj.weight": (4096, 4096),
    "post_attention_layernorm.weight": (4096,),
    "mlp.gate_proj.weight": (14336, 4096),
    "mlp.up_proj.weight": (14336, 4096),
    "mlp.down_proj.weight": (4096, 14336),
}


def validate_checkpoint_config(config, *, max_seq_len=DEFAULT_MAX_SEQ_LEN):
    """Reject architectural substitutions before allocating device weights."""
    geometry = PrefillGeometry(max_seq_len)
    required = {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "tie_word_embeddings": False,
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(f"checkpoint {name}={config.get(name)!r}; expected {expected!r}")
    for name in ("attention_bias", "mlp_bias"):
        if config.get(name, False) is not False:
            raise ValueError(f"Llama prefill does not support {name}")
    if config.get("head_dim", 128) != 128 or config.get("sliding_window") is not None:
        raise ValueError("Llama prefill requires head_dim=128 and full causal attention")
    rope = config.get("rope_scaling") or {}
    for name, expected in {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }.items():
        if rope.get(name) != expected:
            raise ValueError(f"checkpoint rope_scaling.{name}={rope.get(name)!r}; expected {expected!r}")
    if config.get("max_position_embeddings", 0) < geometry.max_seq_len:
        raise ValueError(f"checkpoint context is smaller than the requested {geometry.max_seq_len}-token limit")


class CheckpointWeights:
    """Read only requested tensors and release each shard handle before returning.

    Callers should discard each host layer mapping after device construction. The returned values
    retain their original HF row order and dtype. No Q/K permutation or RoPE conversion occurs.
    """

    def __init__(self, path, *, max_seq_len=DEFAULT_MAX_SEQ_LEN):
        self.path = Path(path).resolve()
        self.config = json.loads((self.path / "config.json").read_text())
        validate_checkpoint_config(self.config, max_seq_len=max_seq_len)
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def _read(self, names_and_shapes):
        grouped = {}
        for name, shape in names_and_shapes.items():
            if name not in self.index:
                raise ValueError(f"checkpoint is missing {name}")
            filename = self.path / self.index[name]
            if filename.resolve().parent != self.path:
                raise ValueError(f"checkpoint shard must be directly inside the checkpoint: {filename}")
            grouped.setdefault(filename, []).append((name, shape))
        result = {}
        for filename, entries in grouped.items():
            with safe_open(filename, framework="pt", device="cpu") as shard:
                for name, shape in entries:
                    tensor = shard.get_tensor(name)
                    if tuple(tensor.shape) != shape or not tensor.is_floating_point():
                        raise ValueError(f"checkpoint {name} must be a floating tensor with shape {shape}")
                    result[name] = tensor
        return result

    def embedding(self):
        name = "model.embed_tokens.weight"
        return self._read({name: (128256, 4096)})[name]

    def final_norm(self):
        name = "model.norm.weight"
        return self._read({name: (4096,)})[name]

    def lm_head(self):
        name = "lm_head.weight"
        return self._read({name: (128256, 4096)})[name]

    def layer(self, layer_idx):
        if type(layer_idx) is not int or not 0 <= layer_idx < 32:
            raise ValueError("layer_idx must be an integer in [0,32)")
        prefix = f"model.layers.{layer_idx}."
        values = self._read({prefix + name: shape for name, shape in LAYER_SHAPES.items()})
        return {name.removeprefix(prefix): tensor for name, tensor in values.items()}
