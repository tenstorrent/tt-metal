# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF reference for the EmbeddingGemma 2 text tower (fp32, CPU).

Needs transformers >= 5.20 (`models/embedding_gemma2`). tt-metal's python_env pins 5.12.1, so the
tests run with an overlay directory first on PYTHONPATH (see README.md):

    PYTHONPATH=/local/ttuser/gtobar/venvs/tf_overlay:$PWD

The reference is the unmodified HF `EmbeddingGemma2TextModel`, loaded from the checkpoint's
`language_model.*` tensors and upcast to fp32. `run_reference` records per-layer activations with
forward hooks; it never changes the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

MIN_TRANSFORMERS = (5, 20)


def _check_transformers():
    import transformers

    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if ver < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"transformers {transformers.__version__} lacks embedding_gemma2 (need >= 5.20). "
            "Prepend the overlay to PYTHONPATH, e.g. PYTHONPATH=/local/ttuser/gtobar/venvs/tf_overlay:$PWD"
        )


def load_reference_text_model(snapshot, state_dict: dict[str, torch.Tensor], attn_implementation: str = "sdpa"):
    """Build HF `EmbeddingGemma2TextModel` (fp32, eval) from the text state dict (no prefix)."""
    _check_transformers()
    from transformers import AutoConfig
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2TextModel

    cfg = AutoConfig.from_pretrained(str(snapshot)).text_config
    cfg._attn_implementation = attn_implementation
    model = EmbeddingGemma2TextModel(cfg)
    sd = {k: v.float() for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # only the non-persistent buffers (rope inv_freq, embed_scale) may be absent from the checkpoint
    missing = [m for m in missing if "inv_freq" not in m and not m.endswith("embed_scale")]
    if missing or unexpected:
        raise RuntimeError(f"reference load mismatch: missing={missing} unexpected={unexpected}")
    return model.float().eval(), cfg


@dataclass
class ReferenceOutputs:
    inputs_embeds: torch.Tensor  # [B, S, 512]
    per_layer_inputs: torch.Tensor  # [B, S, 24, 512]
    layer_in: dict[int, torch.Tensor] = field(default_factory=dict)  # hidden entering layer i [B,S,512]
    layer_out: dict[int, torch.Tensor] = field(default_factory=dict)  # hidden leaving layer i
    attn_in: dict[int, torch.Tensor] = field(default_factory=dict)  # input_layernorm(x) [B,S,512]
    attn_out: dict[int, torch.Tensor] = field(default_factory=dict)  # o_proj output [B,S,512]
    mlp_in: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_out: dict[int, torch.Tensor] = field(default_factory=dict)
    last_hidden: torch.Tensor | None = None  # [B, S, 768] (after final norm + projection)
    embedding: torch.Tensor | None = None  # [B, 768] masked mean + L2 normalize


def sentence_embedding(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """sentence-transformers Pooling(mean, include_prompt) + Normalize."""
    m = attention_mask.to(last_hidden.dtype).unsqueeze(-1)
    pooled = (last_hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
    return torch.nn.functional.normalize(pooled, p=2, dim=-1)


@torch.no_grad()
def run_reference(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    capture_layers: tuple[int, ...] | None = None,
    capture_sublayers: tuple[int, ...] = (),
) -> ReferenceOutputs:
    """One fp32 forward; records layer in/out for `capture_layers` (None = all) and attention/MLP
    in/out for `capture_sublayers`."""
    n = len(model.layers)
    capture_layers = tuple(range(n)) if capture_layers is None else capture_layers
    store: dict[str, dict[int, torch.Tensor]] = {k: {} for k in ("li", "lo", "ai", "ao", "mi", "mo")}
    handles = []

    def layer_hook(i):
        def hook(mod, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            store["li"][i] = x.detach().clone()
            store["lo"][i] = out.detach().clone()

        return hook

    def attn_hook(i):
        def hook(mod, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            store["ai"][i] = x.detach().clone()
            store["ao"][i] = out[0].detach().clone()

        return hook

    def mlp_hook(i):
        def hook(mod, args, out):
            store["mi"][i] = args[0].detach().clone()
            store["mo"][i] = out.detach().clone()

        return hook

    for i in capture_layers:
        handles.append(model.layers[i].register_forward_hook(layer_hook(i), with_kwargs=True))
    for i in capture_sublayers:
        handles.append(model.layers[i].self_attn.register_forward_hook(attn_hook(i), with_kwargs=True))
        handles.append(model.layers[i].mlp.register_forward_hook(mlp_hook(i)))
    try:
        inputs_embeds = model.embed_tokens(input_ids)
        per_layer_inputs = model.ple(inputs_embeds)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    last = out.last_hidden_state
    return ReferenceOutputs(
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        layer_in=store["li"],
        layer_out=store["lo"],
        attn_in=store["ai"],
        attn_out=store["ao"],
        mlp_in=store["mi"],
        mlp_out=store["mo"],
        last_hidden=last,
        embedding=sentence_embedding(last, attention_mask),
    )
