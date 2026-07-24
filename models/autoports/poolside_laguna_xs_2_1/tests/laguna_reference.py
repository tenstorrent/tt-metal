# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Layer-only HuggingFace reference for poolside/Laguna-XS-2.1.

This module builds a single ``LagunaDecoderLayer`` and reproduces
``LagunaModel.forward``'s per-layer dispatch (mask + position-embedding choice by
attention type) so that a single decoder layer can be validated against the exact
HF numerical form, for prefill and incremental decode, without instantiating the
full 40-layer / 67 GB model.

The remote Laguna modeling code is loaded with ``trust_remote_code`` and cached by
transformers; we pull the concrete classes out of the dynamic module cache.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

MODEL_ID = "poolside/Laguna-XS-2.1"


def build_config(attn_implementation: str = "eager"):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Force a concrete attention interface; eager is the exact numerical reference.
    cfg._attn_implementation = attn_implementation
    cfg._attn_implementation_internal = attn_implementation
    return cfg


def _cls(name):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(f"modeling_laguna.{name}", MODEL_ID)


def build_rotary(config):
    """Build the (full_attention, sliding_attention) rotary embeddings exactly as
    ``LagunaModel.__init__`` does."""
    RE = _cls("LagunaRotaryEmbedding")
    rp = config.rope_parameters
    full_cfg = copy.deepcopy(config)
    full_cfg.rope_parameters = dict(rp["full_attention"])
    full_re = RE(config=full_cfg)

    swa_re = None
    if getattr(config, "swa_rope_parameters", None) is not None:
        swa_cfg = copy.deepcopy(config)
        swa_cfg.rope_parameters = dict(config.swa_rope_parameters)
        swa_cfg.partial_rotary_factor = swa_cfg.rope_parameters.get("partial_rotary_factor")
        swa_re = RE(config=swa_cfg)
    return full_re, swa_re


def build_layer(config, layer_idx, *, dtype=torch.float32, seed=1234, state_dict=None):
    """Instantiate one ``LagunaDecoderLayer``. If ``state_dict`` is given (keys with
    the ``model.layers.{idx}.`` prefix stripped), load real weights; otherwise fill
    with deterministic synthetic weights derived from ``seed``."""
    DL = _cls("LagunaDecoderLayer")
    layer = DL(config, layer_idx)
    if state_dict is not None:
        missing, unexpected = layer.load_state_dict(state_dict, strict=False)
        # Fused experts: HF LagunaExperts uses gate_up_proj/down_proj 3D params, but the
        # checkpoint ships per-expert 2D weights. The caller is expected to have already
        # converted them (see load_layer_state_dict). Any genuine miss is a bug.
        assert not missing, f"missing keys: {missing[:8]}"
        assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    else:
        g = torch.Generator().manual_seed(seed)
        for p in layer.parameters():
            if p.dim() == 1:
                p.data.normal_(0.0, 0.02, generator=g)
            else:
                p.data.normal_(0.0, 0.02, generator=g)
    # Architecture attributes (is_sliding, num_heads, sliding_window, mlp kind) are baked
    # at construction from layer_idx. Remap the *cache* index to 0 so a single-layer
    # DynamicCache stores/reads at index 0 and get_seq_length() reports correctly.
    layer.self_attn.layer_idx = 0
    return layer.to(dtype).eval()


@dataclass
class RefContext:
    config: object
    layer: object
    full_re: object
    swa_re: object
    attention_type: str

    def position_embeddings(self, hidden_states, position_ids):
        if self.attention_type == "sliding_attention" and self.swa_re is not None:
            re = self.swa_re
        else:
            re = self.full_re
        return re(hidden_states, position_ids)

    def build_mask(self, inputs_embeds, attention_mask, past_key_values, position_ids):
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        if self.attention_type == "sliding_attention":
            return create_sliding_window_causal_mask(**mask_kwargs)
        return create_causal_mask(**mask_kwargs)


def make_context(config, layer_idx, **kw):
    layer = build_layer(config, layer_idx, **kw)
    full_re, swa_re = build_rotary(config)
    attn_type = config.layer_types[layer_idx] if getattr(config, "layer_types", None) else "full_attention"
    return RefContext(config, layer, full_re, swa_re, attn_type)


@torch.no_grad()
def reference_forward(ctx: RefContext, hidden_states, past_key_values=None, position_ids=None):
    """Run one decoder layer over ``hidden_states`` [batch, seq, hidden] with the
    given cache and positions, reproducing LagunaModel.forward's dispatch. Returns
    (output_hidden [batch, seq, hidden], past_key_values)."""
    from transformers.cache_utils import DynamicCache

    cfg = ctx.config
    bsz, seq, _ = hidden_states.shape
    if past_key_values is None:
        past_key_values = DynamicCache(config=cfg)
    if position_ids is None:
        past_seen = past_key_values.get_seq_length()
        position_ids = (torch.arange(seq) + past_seen).unsqueeze(0).expand(bsz, -1)

    mask = ctx.build_mask(hidden_states, None, past_key_values, position_ids)
    pos_emb = ctx.position_embeddings(hidden_states, position_ids)

    out = ctx.layer(
        hidden_states,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        position_embeddings=pos_emb,
    )
    # LagunaDecoderLayer.forward returns the hidden-state tensor directly.
    hidden = out[0] if isinstance(out, tuple) else out
    return hidden, past_key_values
