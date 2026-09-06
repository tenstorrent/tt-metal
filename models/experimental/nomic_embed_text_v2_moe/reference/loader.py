# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint contract and loading for the vendored reference.

`expected_checkpoint_keys` *generates* the key/shape contract from a `NomicMoEConfig`
rather than storing a captured list. That way the generator -- in particular the
`i % 2 == 1` MoE-placement predicate and the expert packing shapes -- is itself under
test when compared against the real checkpoint, instead of a frozen list being compared
to a frozen list.
"""

from __future__ import annotations

from typing import Optional

import torch

from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import NomicMoEConfig
from models.experimental.nomic_embed_text_v2_moe.reference.modeling_nomic_moe import NomicBertModel


def expected_checkpoint_keys(config: NomicMoEConfig) -> dict[str, tuple[int, ...]]:
    """The full `key -> shape` contract implied by `config`."""
    H = config.hidden_size
    F = config.intermediate_size
    E = config.num_experts

    keys: dict[str, tuple[int, ...]] = {
        "embeddings.word_embeddings.weight": (config.vocab_size, H),
        "embeddings.token_type_embeddings.weight": (config.type_vocab_size, H),
        "emb_ln.weight": (H,),
        "emb_ln.bias": (H,),
    }

    for i in range(config.num_hidden_layers):
        prefix = f"encoder.layers.{i}."
        keys[prefix + "attn.Wqkv.weight"] = (3 * H, H)
        keys[prefix + "attn.Wqkv.bias"] = (3 * H,)
        keys[prefix + "attn.out_proj.weight"] = (H, H)
        keys[prefix + "attn.out_proj.bias"] = (H,)
        keys[prefix + "norm1.weight"] = (H,)
        keys[prefix + "norm1.bias"] = (H,)
        keys[prefix + "norm2.weight"] = (H,)
        keys[prefix + "norm2.bias"] = (H,)

        if config.is_moe_layer(i):
            keys[prefix + "mlp.router.layer.weight"] = (E, H)  # bias=False: no router bias
            keys[prefix + "mlp.experts.mlp.w1"] = (E * F, H)
            keys[prefix + "mlp.experts.mlp.w2"] = (E * F, H)
            keys[prefix + "mlp.experts.bias"] = (H,)  # ONE shared bias, not per-expert
        else:
            keys[prefix + "mlp.fc1.weight"] = (F, H)
            keys[prefix + "mlp.fc1.bias"] = (F,)
            keys[prefix + "mlp.fc2.weight"] = (H, F)
            keys[prefix + "mlp.fc2.bias"] = (H,)

    return keys


# Parameter paths that must NOT appear. Each corresponds to an upstream feature this
# checkpoint does not use; if one showed up, the vendored reference would be silently
# dropping a real weight.
ABSENT_KEY_SUBSTRINGS = (
    "position_embeddings",  # rotary-only; no learned position table
    "pooler",  # add_pooling_layer=False
    "cls.",  # no pretraining head
    "lm_head",
    "ln_f",  # no final norm outside the blocks (post-norm ends with norm2)
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
    """Build the reference and, when a state dict is given, load it with `strict=True`.

    `strict=True` is the point: because the module tree mirrors upstream's names exactly,
    a clean load with zero missing and zero unexpected keys is the structural proof that
    the vendored reference has the same parameters, in the same places, as the checkpoint.
    """
    model = NomicBertModel(config)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
