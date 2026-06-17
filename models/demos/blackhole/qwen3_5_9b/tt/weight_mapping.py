# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Remap a HuggingFace Qwen3.5 state dict to the internal key scheme model_config.load_state_dict feeds the TT model.

The internal scheme is deliberately MINIMAL: only the three top-level weights the
framework Embedding / final RMSNorm / LM head look up are renamed —
``embed_tokens``→``tok_embeddings``, ``model.norm``→``norm``, ``lm_head``→``output`` —
and every per-layer weight passes through RAW under ``layers.{i}.``. That is exactly
what the per-module weight loaders consume: attention/gdn/mlp/weights.py read the raw
transformers submodule names (``self_attn.q_proj``, ``linear_attn.in_proj_qkv``,
``linear_attn.conv1d``, ``mlp.gate_proj``, ...) after layer.py's substate() strips the
``layers.{i}.{self_attn|linear_attn|mlp}.`` prefix. This MUST match the test path
(tests/unit/test_model.py::_remap_hf_state_dict), which feeds the TT model these same
raw layer keys — so the production load path and the validated test path agree.

(History: an older remap renamed ``in_proj_qkv``→``qkv_proj`` and split ``conv1d`` into
per-stream q/k/v conv weights for a previous GDN loader. The current GDN loader does the
fuse/split itself from the raw ``in_proj_qkv`` / ``conv1d`` weights, so that renaming is
gone — keep linear_attn weights raw.)
"""
from typing import Dict

import torch


def remap_qwen35_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap an HF Qwen3.5 state dict to the internal scheme (see module docstring).

    Args:
        state_dict: Raw HuggingFace state dict (model. or model.language_model. prefixed).

    Returns:
        Remapped state dict: tok_embeddings / norm / output renamed, layer weights raw.
    """

    def strip_prefix(key: str) -> str:
        for prefix in ("model.language_model.", "model."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    remapped = {}
    for key, tensor in state_dict.items():
        # Drop the vision tower and multi-token-prediction heads (text-only model).
        if "visual" in key or key.startswith("model.visual") or key.startswith("mtp"):
            continue

        new_key = strip_prefix(key)

        # The three top-level weights the framework modules look up by a fixed name.
        if new_key == "embed_tokens.weight":
            remapped["tok_embeddings.weight"] = tensor
        elif key == "lm_head.weight" or new_key == "lm_head.weight":
            remapped["output.weight"] = tensor
        else:
            # norm.weight and every layers.N.* weight pass through RAW — the per-module
            # loaders consume the unmodified transformers submodule names.
            remapped[new_key] = tensor
    return remapped
