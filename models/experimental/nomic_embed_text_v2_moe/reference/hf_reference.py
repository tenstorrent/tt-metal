# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loading the upstream HF model, and containment for the native-class collision.

transformers >= 5 ships a native transformers.models.nomic_bert targeting
nomic-embed-text-v1.5: separate q/k/v/o projections, no bias, SwiGLU MLP, no MoE. It is
registered for model_type == "nomic_bert", which is what this checkpoint declares.

Measured on transformers 5.12.1 at the pinned revision:
  AutoConfig.from_pretrained  -> native config class; only use_cache is dropped.
  AutoModel.from_pretrained   -> native model class, and it does not raise. Every MoE tensor,
                                 mlp.fc1/fc2, and all q/k/v/o biases are reported UNEXPECTED
                                 and discarded; gate_proj/up_proj/down_proj are reported
                                 MISSING and randomly initialised. The result has 136
                                 parameters, no MoE, and returns finite, wrong numbers.

Containment: pass trust_remote_code=True, then assert the resolved class came from
transformers_modules. The assert is the load-bearing part; without it a future transformers
release that changes resolution order silently downgrades the golden reference.
"""

from __future__ import annotations

import torch

from models.experimental.nomic_embed_text_v2_moe.common import MODEL

REMOTE_MODULE_PREFIX = "transformers_modules"


class RemoteCodeResolutionError(RuntimeError):
    """A class resolved to a native transformers implementation instead of the remote code."""


def assert_resolved_from_remote_code(obj: object, what: str) -> None:
    """Verify an Auto* class resolved to the remote code, not the native implementation.

    This assert is the load-bearing part of the containment, not trust_remote_code itself:
    without it, a future transformers release that changes resolution order would silently
    downgrade the golden reference.

    Args:
        obj: The loaded config or model to check.
        what: Name used in the error message, "AutoConfig" or "AutoModel".

    Returns:
        None.

    Raises:
        RemoteCodeResolutionError: If the object's class came from anywhere other than
            transformers_modules.
    """
    module = type(obj).__module__
    if not module.startswith(REMOTE_MODULE_PREFIX):
        raise RemoteCodeResolutionError(
            f"{what} resolved to {module}.{type(obj).__name__}, not the remote code under "
            f"{REMOTE_MODULE_PREFIX!r}. The native transformers nomic_bert implementation targets "
            "nomic-embed-text-v1.5, has no MoE, and discards this checkpoint's expert weights "
            "without raising. Pass trust_remote_code=True."
        )


def load_hf_config():
    """Load the upstream config at the pinned revision, guaranteed to be the remote code.

    Returns:
        NomicBertConfig: Upstream's own config class, from transformers_modules.

    Raises:
        RemoteCodeResolutionError: If the native transformers class was resolved instead.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        MODEL.model_id,
        revision=MODEL.revision,
        trust_remote_code=True,
    )
    assert_resolved_from_remote_code(config, "AutoConfig")
    return config


def load_hf_model():
    """Load the upstream model at the pinned revision, guaranteed to be the remote code.

    This is the parity oracle every comparison runs against, which is why the resolution check
    matters: a bare AutoModel.from_pretrained returns a v1.5 class with no MoE and does not
    raise.

    Returns:
        NomicBertModel: Upstream's class, in eval mode, fp32, with the expert weights intact.

    Raises:
        RemoteCodeResolutionError: If the native transformers class was resolved instead.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL.model_id,
        revision=MODEL.revision,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    assert_resolved_from_remote_code(model, "AutoModel")
    model.eval()
    return model


def hf_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the upstream model and return its last hidden state.

    attention_mask is required here, unlike in the reference: upstream calls
    get_extended_attention_mask unconditionally and raises AttributeError on None.

    Args:
        model: The upstream model from load_hf_model.
        input_ids: (B, S) int64 token ids.
        attention_mask: (B, S) int64, 1 for real tokens and 0 for padding. Not optional.
        token_type_ids: (S,) int64, or None.

    Returns:
        torch.Tensor: (B, S, 768) fp32, unwrapped from upstream's output object.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    return out.last_hidden_state


def hf_layer_ladder(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture the upstream model's output at every parity ladder point.

    The counterpart to reference_ladder in the test module, so the two can be compared path by
    path.

    Args:
        model: The upstream model from load_hf_model.
        input_ids: (B, S) int64 token ids.
        attention_mask: (B, S) int64, 1 for real tokens and 0 for padding.

    Returns:
        dict[str, torch.Tensor]: 13 entries, emb_ln plus each encoder.layers.{i}, each holding
        that module's (B, S, 768) output.
    """
    from models.experimental.nomic_embed_text_v2_moe.common import capture_hidden_states, layer_ladder_paths

    paths = layer_ladder_paths(model.config.n_layer)
    captures, handles = capture_hidden_states(model, paths)
    try:
        hf_forward(model, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return captures
