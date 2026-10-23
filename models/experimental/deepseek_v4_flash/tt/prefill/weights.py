# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint -> weight dicts for the prefill model.

:func:`checkpoint_weights` returns the flat ``name -> lazy thunk`` dict
:class:`~..model.DeepSeekV4PrefillModel` takes (``embed_tokens.weight``, ``layers.{i}.<layer keys>``,
``hc_head.*``, ``norm.weight``, ``lm_head.weight``) and :func:`checkpoint_expert_provider` the per-layer
routed-expert provider. Nothing is read until a thunk is called, so a populated tile cache skips the
checkpoint entirely. The layer keys are the ones the decode model builds (``DeepSeekV4Model.
_build_layer_weights``) plus a CSA layer's lightning-indexer tensors (read only with ``lightning_indexer=True``).
"""

from typing import Callable

import torch

from ..quant import dequantize_weight
from ..weight_loader import DeepseekV4WeightLoader


def _thunk(loader: DeepseekV4WeightLoader, name: str) -> Callable[[], torch.Tensor]:
    """Zero-arg thunk dequantizing checkpoint weight ``name`` to a host tensor."""
    return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


_ATTENTION_KEYS = (
    "q_a_proj.weight",
    "q_a_norm.weight",
    "q_b_proj.weight",
    "kv_proj.weight",
    "kv_norm.weight",
    "o_a_proj.weight",
    "o_b_proj.weight",
    "sinks",
)
_COMPRESSOR_KEYS = (
    "compressor.kv_proj.weight",
    "compressor.gate_proj.weight",
    "compressor.kv_norm.weight",
    "compressor.position_bias",
)


_INDEXER_KEYS = (
    "compressor.indexer.kv_proj.weight",
    "compressor.indexer.gate_proj.weight",
    "compressor.indexer.kv_norm.weight",
    "compressor.indexer.position_bias",
    "compressor.indexer.q_b_proj.weight",
    "compressor.indexer.weights_proj.weight",
)


def layer_weights(loader: DeepseekV4WeightLoader, layer_idx: int, layer_type: str, is_hash: bool) -> dict:
    """Layer ``layer_idx``'s weights under the module-relative names the prefill layer takes."""
    keys = list(_ATTENTION_KEYS)
    if layer_type != "sliding_attention":
        keys += _COMPRESSOR_KEYS
    if layer_type == "compressed_sparse_attention":
        keys += _INDEXER_KEYS  # lazy thunks: only read when the prefill runs the lightning indexer
    weights: dict = {f"self_attn.{k}": _thunk(loader, f"layers.{layer_idx}.self_attn.{k}") for k in keys}

    weights["mlp.gate.weight"] = _thunk(loader, f"layers.{layer_idx}.mlp.gate.weight")
    if is_hash:
        # The frozen token-id -> expert-id table: small, integer, and never tile-cached.
        weights["mlp.gate.tid2eid"] = loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long()
    else:
        weights["mlp.gate.e_score_correction_bias"] = _thunk(
            loader, f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        )
    for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        weights[f"mlp.shared_experts.{k}"] = _thunk(loader, f"layers.{layer_idx}.mlp.shared_experts.{k}")
    for hc in ("attn_hc", "ffn_hc"):
        for p in ("fn", "base", "scale"):
            weights[f"{hc}.{p}"] = _thunk(loader, f"layers.{layer_idx}.{hc}.{p}")
    for k in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        weights[k] = _thunk(loader, f"layers.{layer_idx}.{k}")
    return weights


def checkpoint_weights(loader: DeepseekV4WeightLoader, config, num_layers: int) -> dict:
    """The whole prefill model's weights, as lazy thunks (see the module docstring)."""
    weights: dict = {
        # The table is stored bf16: no scale to apply.
        "embed_tokens.weight": lambda: loader.get_tensor("embed_tokens.weight"),
        "norm.weight": _thunk(loader, "norm.weight"),
        "lm_head.weight": _thunk(loader, "lm_head.weight"),
    }
    for p in ("hc_fn", "hc_base", "hc_scale"):
        weights[f"hc_head.{p}"] = _thunk(loader, f"hc_head.{p}")
    for li in range(num_layers):
        layer = layer_weights(loader, li, config.layer_types[li], config.mlp_layer_types[li] == "hash_moe")
        weights.update({f"layers.{li}.{k}": v for k, v in layer.items()})
    return weights


def checkpoint_expert_provider(loader: DeepseekV4WeightLoader) -> Callable[[int], Callable[[int], tuple]]:
    """``provider_for(layer_idx)(e) -> (gate_up [2I, D], down [D, I])`` float32 host tensors for expert ``e``."""

    def provider_for(layer_idx: int):
        def provider(e: int):
            base = f"layers.{layer_idx}.mlp.experts.{e}"
            gate = _thunk(loader, f"{base}.gate_proj.weight")()
            up = _thunk(loader, f"{base}.up_proj.weight")()
            down = _thunk(loader, f"{base}.down_proj.weight")()
            return torch.cat([gate, up], dim=0).float(), down.float()

        return provider

    return provider_for
