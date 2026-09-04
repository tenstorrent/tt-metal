# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loading the upstream HF model safely, and the trap that makes that non-trivial.

`transformers` >= 5 ships a NATIVE `transformers.models.nomic_bert` targeting
**nomic-embed-text-v1.5**: separate q/k/v/o projections, no MoE, a SwiGLU
`gate_proj`/`up_proj`/`down_proj` MLP. It is registered for `model_type == "nomic_bert"`,
which is exactly what this checkpoint's `config.json` declares.

Measured on transformers 5.12.1 with the pinned revision:

*   `AutoConfig.from_pretrained(MODEL_ID)` -> the NATIVE config class. It is comparatively
    benign: every field of `config.json` survives except `use_cache`.
*   `AutoModel.from_pretrained(MODEL_ID)` -> the NATIVE model class, and this one is not
    benign at all. It **does not raise**. It reports every MoE tensor
    (`mlp.experts.mlp.w1/w2`, `mlp.router.layer.weight`, `mlp.experts.bias`), every
    `mlp.fc1/fc2`, and all q/k/v/o biases as UNEXPECTED -- silently discarded -- and
    reports `gate_proj`/`up_proj`/`down_proj` as MISSING, i.e. **randomly initialised**.
    The result is 136 parameters, no MoE, and a forward pass that returns finite,
    plausible, entirely wrong numbers.

So the containment is: always pass `trust_remote_code=True` *and* `code_revision`, then
assert the resolved class actually came from `transformers_modules`. The assert is the
part that matters -- without it, a future transformers release that changes resolution
order would silently downgrade the golden reference.
"""

from __future__ import annotations

import torch

from models.experimental.nomic_embed_text_v2_moe.common import CODE_REVISION, MODEL_ID, MODEL_REVISION

REMOTE_MODULE_PREFIX = "transformers_modules"


class RemoteCodeResolutionError(RuntimeError):
    """A class resolved to a native transformers implementation instead of the remote code."""


def assert_resolved_from_remote_code(obj: object, what: str) -> None:
    """Fail loudly if `obj`'s class did not come from the hub's remote code."""
    module = type(obj).__module__
    if not module.startswith(REMOTE_MODULE_PREFIX):
        raise RemoteCodeResolutionError(
            f"{what} resolved to {module}.{type(obj).__name__}, not the remote code under "
            f"{REMOTE_MODULE_PREFIX!r}. The native transformers `nomic_bert` implementation targets "
            "nomic-embed-text-v1.5 -- it has no MoE, and it discards this checkpoint's expert "
            "weights without raising. Pass trust_remote_code=True and code_revision."
        )


def load_hf_config(revision: str = MODEL_REVISION, code_revision: str = CODE_REVISION):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=True,
        code_revision=code_revision,
    )
    assert_resolved_from_remote_code(config, "AutoConfig")
    return config


def load_hf_model(revision: str = MODEL_REVISION, code_revision: str = CODE_REVISION):
    """The upstream model at the pinned revisions, in eval mode, guaranteed remote-code."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=True,
        code_revision=code_revision,
        dtype=torch.float32,
    )
    assert_resolved_from_remote_code(model, "AutoModel")
    model.eval()
    return model


def hf_last_hidden_state(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the upstream model.

    `attention_mask` is required, not optional: upstream calls
    `get_extended_attention_mask(attention_mask, ...)` unconditionally and raises
    `AttributeError` on None. The vendored reference deliberately defaults it instead.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    return out.last_hidden_state


def hf_layer_ladder(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture `emb_ln` and every `encoder.layers.{i}` output from the upstream model."""
    from models.experimental.nomic_embed_text_v2_moe.common import capture_hidden_states, layer_ladder_paths

    paths = layer_ladder_paths(model.config.n_layer)
    captures, handles = capture_hidden_states(model, paths)
    try:
        hf_last_hidden_state(model, input_ids, attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    return captures
