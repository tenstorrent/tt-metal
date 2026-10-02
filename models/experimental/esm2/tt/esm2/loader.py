# SPDX-License-Identifier: MIT
"""Load the pinned ESM-2 safetensors checkpoint into canonical module names.

Checkpoint (HF EsmForMaskedLM) -> canonical (tt/esm2/reference_layers) map:
  esm.embeddings.word_embeddings.weight      -> embeddings.word_embeddings.weight
  esm.encoder.layer.N.attention.self.query.* -> layers.N.attn.q.*
  ...key/value                              -> layers.N.attn.k./v.*
  esm.encoder.layer.N.attention.output.dense.* -> layers.N.attn_out.*
  esm.encoder.layer.N.attention.LayerNorm.* -> layers.N.ln_attn.*
  esm.encoder.layer.N.LayerNorm.*           -> layers.N.ln_ffn.*
  esm.encoder.layer.N.intermediate.dense.*  -> layers.N.ffn1.*
  esm.encoder.layer.N.output.dense.*        -> layers.N.ffn2.*
  esm.encoder.emb_layer_norm_after.*        -> final_ln.*
  lm_head.dense.* / lm_head.layer_norm.*    -> lm.dense.* / lm.ln.*
  lm_head.bias                              -> lm.bias
  (decoder weight is tied to word embeddings; contact_head, per-layer rotary
   inv_freq buffers and the unused absolute-position table are not in the
   pinned manifest for this config.)
"""

from __future__ import annotations

import os

import torch

from .config import Esm2TTConfig

_CHECKPOINT_TO_CANONICAL = [
    ("esm.embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"),
    ("lm_head.dense.weight", "lm.dense.weight"),
    ("lm_head.dense.bias", "lm.dense.bias"),
    ("lm_head.layer_norm.weight", "lm.ln.weight"),
    ("lm_head.layer_norm.bias", "lm.ln.bias"),
    ("lm_head.bias", "lm.bias"),
    ("esm.encoder.emb_layer_norm_after.weight", "final_ln.weight"),
    ("esm.encoder.emb_layer_norm_after.bias", "final_ln.bias"),
]


def layer_key_map(layer_idx: int) -> list[tuple[str, str]]:
    hf, canon = f"esm.encoder.layer.{layer_idx}.", f"layers.{layer_idx}."
    return [
        (hf + "attention.self.query.weight", canon + "attn.q.weight"),
        (hf + "attention.self.query.bias", canon + "attn.q.bias"),
        (hf + "attention.self.key.weight", canon + "attn.k.weight"),
        (hf + "attention.self.key.bias", canon + "attn.k.bias"),
        (hf + "attention.self.value.weight", canon + "attn.v.weight"),
        (hf + "attention.self.value.bias", canon + "attn.v.bias"),
        (hf + "attention.output.dense.weight", canon + "attn_out.weight"),
        (hf + "attention.output.dense.bias", canon + "attn_out.bias"),
        (hf + "attention.LayerNorm.weight", canon + "ln_attn.weight"),
        (hf + "attention.LayerNorm.bias", canon + "ln_attn.bias"),
        (hf + "LayerNorm.weight", canon + "ln_ffn.weight"),
        (hf + "LayerNorm.bias", canon + "ln_ffn.bias"),
        (hf + "intermediate.dense.weight", canon + "ffn1.weight"),
        (hf + "intermediate.dense.bias", canon + "ffn1.bias"),
        (hf + "output.dense.weight", canon + "ffn2.weight"),
        (hf + "output.dense.bias", canon + "ffn2.bias"),
    ]


def _safetensors_path(weights_path: str) -> str:
    if os.path.isdir(weights_path):
        return os.path.join(weights_path, "model.safetensors")
    return weights_path


def load_canonical_weights(weights_path: str, config: Esm2TTConfig) -> dict:
    """safetensors file-or-dir -> {canonical_name: torch.FloatTensor fp32}."""
    from safetensors.torch import load_file

    raw = load_file(_safetensors_path(weights_path))
    mapping = list(_CHECKPOINT_TO_CANONICAL)
    for n in range(config.num_hidden_layers):
        mapping.extend(layer_key_map(n))
    out: dict = {}
    missing = []
    for ckpt_key, canon_key in mapping:
        if ckpt_key in raw:
            out[canon_key] = raw[ckpt_key].to(torch.float32)
        else:
            missing.append(ckpt_key)
    if missing:
        raise KeyError(f"checkpoint missing expected tensors: {missing[:5]} (of {len(missing)})")
    return out
