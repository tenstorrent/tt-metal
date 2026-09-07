# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint key/shape contract and model loading.

expected_checkpoint_keys generates the contract from a NomicMoEConfig rather than storing a
captured list, so comparing it against the real checkpoint tests the generator itself: the MoE
placement predicate, the expert packing shapes, and the bias-free router.
"""

from __future__ import annotations

from typing import Optional

import torch

from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import NomicMoEConfig
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicBertModel


def expected_checkpoint_keys(config: NomicMoEConfig) -> dict[str, tuple[int, ...]]:
    """The full key to shape contract implied by config."""
    hidden = config.hidden_size
    ffn = config.intermediate_size
    experts = config.num_experts

    keys: dict[str, tuple[int, ...]] = {
        "embeddings.word_embeddings.weight": (config.vocab_size, hidden),
        "embeddings.token_type_embeddings.weight": (config.type_vocab_size, hidden),
        "emb_ln.weight": (hidden,),
        "emb_ln.bias": (hidden,),
    }

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"encoder.layers.{layer_idx}."
        keys[prefix + "attn.Wqkv.weight"] = (config.qkv_dim, hidden)
        keys[prefix + "attn.Wqkv.bias"] = (config.qkv_dim,)
        keys[prefix + "attn.out_proj.weight"] = (hidden, hidden)
        keys[prefix + "attn.out_proj.bias"] = (hidden,)
        keys[prefix + "norm1.weight"] = (hidden,)
        keys[prefix + "norm1.bias"] = (hidden,)
        keys[prefix + "norm2.weight"] = (hidden,)
        keys[prefix + "norm2.bias"] = (hidden,)

        if config.is_moe_layer(layer_idx):
            keys[prefix + "mlp.router.layer.weight"] = (experts, hidden)
            keys[prefix + "mlp.experts.mlp.w1"] = (experts * ffn, hidden)
            keys[prefix + "mlp.experts.mlp.w2"] = (experts * ffn, hidden)
            keys[prefix + "mlp.experts.bias"] = (hidden,)
        else:
            keys[prefix + "mlp.fc1.weight"] = (ffn, hidden)
            keys[prefix + "mlp.fc1.bias"] = (ffn,)
            keys[prefix + "mlp.fc2.weight"] = (hidden, ffn)
            keys[prefix + "mlp.fc2.bias"] = (hidden,)

    return keys


# Parameter paths that must not appear. Each is an upstream feature this checkpoint does not
# use; a hit means the reference is silently dropping a real weight.
ABSENT_KEY_SUBSTRINGS = (
    "position_embeddings",  # rotary-only
    "pooler",  # add_pooling_layer is False
    "cls.",  # no pretraining head
    "lm_head",
    "ln_f",  # post-norm ends with norm2; no final norm outside the blocks
    "inv_freq",  # non-persistent buffer
    "norm_factor",  # non-persistent buffer
    "mlp.router.layer.bias",  # router is bias-free
    "vision",
)


def load_state_dict_from_safetensors(path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


def load_reference_model(
    config: NomicMoEConfig,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> NomicBertModel:
    """Build the reference and, when a state dict is given, load it with strict=True.

    strict=True is the point: the module tree mirrors upstream's names, so a clean load is the
    structural proof that the reference holds the same parameters in the same places.
    """
    model = NomicBertModel(config)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
