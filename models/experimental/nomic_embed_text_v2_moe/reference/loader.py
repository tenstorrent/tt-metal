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

from models.experimental.nomic_embed_text_v2_moe.common import resolve_checkpoint
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import (
    NomicMoEConfig,
    load_vendored_config,
)
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicBertModel


def expected_checkpoint_keys(config: NomicMoEConfig) -> dict[str, tuple[int, ...]]:
    """Generate the full key-to-shape contract implied by the config.

    Computed from the config rather than recorded from a checkpoint, so comparing it against
    the real file tests this generator's own logic: the MoE placement predicate, the expert
    packing shapes, and the bias-free router.

    Args:
        config: NomicMoEConfig describing the architecture.

    Returns:
        dict[str, tuple[int, ...]]: 148 entries for this checkpoint, mapping each parameter
        name to its exact shape. Dense layers contribute mlp.fc1/fc2, MoE layers contribute
        mlp.router and mlp.experts.
    """
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
    """Read a safetensors file into a plain state dict.

    Args:
        path: Path to the .safetensors file, from resolve_checkpoint.

    Returns:
        dict[str, torch.Tensor]: Parameter name to tensor, 148 entries and all fp32 for this
        checkpoint.
    """
    from safetensors.torch import load_file

    return load_file(str(path))


def load_reference_model(
    config: NomicMoEConfig,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> NomicBertModel:
    """Build the reference model and optionally load weights into it.

    strict=True is the point, not a precaution: the module tree mirrors upstream's names, so a
    clean load with zero missing and zero unexpected keys is the structural proof that the
    reference holds the same parameters in the same places.

    Args:
        config: NomicMoEConfig to build from.
        state_dict: Weights to load. None leaves the randomly initialised parameters in place,
            which is what the shape-only tests want.

    Returns:
        NomicBertModel: In eval mode.

    Raises:
        RuntimeError: If state_dict is given and any key or shape does not match.
    """
    model = NomicBertModel(config)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_pretrained_reference_model(allow_download: bool = True) -> NomicBertModel:
    """Load a reference model holding the real pinned checkpoint weights.

    Composes checkpoint resolution, the safetensors read, and the strict load. Prefer this over
    repeating the sequence; load_reference_model stays available for callers that already hold
    a state dict.

    Deliberately offers no revision argument: the config comes from the vendored snapshot, so a
    revision knob here would pair one revision's weights with another's config.

    Args:
        allow_download: When False, fail instead of fetching the 1.8 GB checkpoint.

    Returns:
        NomicBertModel: In eval mode, holding the pinned checkpoint's 475,292,928 parameters.
    """
    state_dict = load_state_dict_from_safetensors(resolve_checkpoint(allow_download=allow_download))
    return load_reference_model(load_vendored_config(), state_dict)
