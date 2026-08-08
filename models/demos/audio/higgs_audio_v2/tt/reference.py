# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""
HF v2 checkpoint loader + state-dict remapper for the TTNN Higgs Audio v2 model.

The on-disk weights at /data/hf_cache/higgs/ use the native transformers v2
flat-config schema. Per-layer keys mostly match HuggingFace Llama naming, so we
reuse tt_transformers.tt.load_checkpoints.map_hf_to_meta_keys for the text side
and handle the Higgs-specific audio dual paths here.

This module is intentionally a *loader only*. The PyTorch reference generation
runs via native `transformers.HiggsAudioV2ForConditionalGeneration` — see
scripts/capture_pytorch_baseline.py. We do not wrap the upstream model.

Key remap (post-call):
    HF                                                          tt-local
    ---------------------------------------------------------------------------
    model.embed_tokens.weight                                   tok_embeddings.weight
    model.embed_audio_tokens.embed_audio_tokens.weight          audio_tok_embeddings.weight
    model.norm.weight                                           norm.weight
    text_lm_head.weight                                         text_output.weight
    audio_lm_head.weight                                        audio_output.weight
    model.layers.N.input_layernorm.weight                       layers.N.attention_norm.weight
    model.layers.N.audio_input_layernorm.weight                 layers.N.audio_attention_norm.weight
    model.layers.N.post_attention_layernorm.weight              layers.N.ffn_norm.weight
    model.layers.N.audio_post_attention_layernorm.weight        layers.N.audio_ffn_norm.weight
    model.layers.N.self_attn.{q,k,v,o}_proj.weight              layers.N.attention.{wq,wk,wv,wo}.weight
    model.layers.N.mlp.{gate,up,down}_proj.weight               layers.N.feed_forward.{w1,w3,w2}.weight
    model.layers.N.audio_mlp.{gate,up,down}_proj.weight         layers.N.audio_feed_forward.{w1,w3,w2}.weight
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open

from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys


@dataclass
class HiggsAudioV2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict[str, Any] | None
    num_codebooks: int
    codebook_size: int
    audio_vocab_size: int
    audio_bos_token_id: int
    audio_delay_token_id: int
    audio_token_id: int
    audio_stream_bos_id: int
    audio_stream_eos_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    torch_dtype: str

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "HiggsAudioV2Config":
        cfg = json.loads(pathlib.Path(path).read_text())
        rp = cfg.get("rope_parameters") or {}
        rope_theta = float(rp.get("rope_theta", cfg.get("rope_theta", 500000.0)))
        rope_scaling = {k: v for k, v in rp.items() if k != "rope_theta"} if rp else None
        codebook_size = int(cfg["codebook_size"])
        num_codebooks = int(cfg["num_codebooks"])
        return cls(
            hidden_size=int(cfg["hidden_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg["head_dim"]),
            intermediate_size=int(cfg["intermediate_size"]),
            vocab_size=int(cfg["vocab_size"]),
            max_position_embeddings=int(cfg["max_position_embeddings"]),
            rms_norm_eps=float(cfg["rms_norm_eps"]),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            audio_vocab_size=num_codebooks * codebook_size,
            audio_bos_token_id=int(cfg["audio_bos_token_id"]),
            audio_delay_token_id=int(cfg["audio_delay_token_id"]),
            audio_token_id=int(cfg["audio_token_id"]),
            audio_stream_bos_id=int(cfg["audio_stream_bos_id"]),
            audio_stream_eos_id=int(cfg["audio_stream_eos_id"]),
            bos_token_id=int(cfg["bos_token_id"]),
            eos_token_id=int(cfg["eos_token_id"]),
            pad_token_id=int(cfg["pad_token_id"]),
            torch_dtype=str(cfg.get("dtype", "bfloat16")),
        )


def load_hf_state_dict(safetensors_path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(safetensors_path), framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def remap_higgs_to_tt(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap upstream HF v2 keys to tt_transformers Meta-style + Higgs-local names.

    tt_transformers' generic HF->Meta map uses word-boundary regex (``\\b``), which
    does not cross underscores. So substrings like ``audio_input_layernorm`` are
    NOT touched by its ``("input_layernorm", "attention_norm")`` rule, and
    ``audio_lm_head`` is NOT touched by ``("lm_head", "output")``. We rewrite the
    Higgs-specific dual-FFN keys here and feed the result to map_hf_to_meta_keys
    only for the text-side renames it does handle correctly.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k

        # Stacked audio token embedding lookup table.
        if nk == "model.embed_audio_tokens.embed_audio_tokens.weight":
            nk = "model.audio_tok_embeddings.weight"

        # Per-layer Higgs-specific (audio side of DualFFN). Order matters:
        # the longer ``audio_post_attention_layernorm`` must be rewritten before
        # the shorter ``audio_input_layernorm`` so the generic ``\\b`` rules can't
        # catch a partial.
        nk = nk.replace("audio_post_attention_layernorm", "audio_ffn_norm")
        nk = nk.replace("audio_input_layernorm", "audio_attention_norm")
        nk = nk.replace("audio_mlp.gate_proj", "audio_feed_forward.w1")
        nk = nk.replace("audio_mlp.up_proj", "audio_feed_forward.w3")
        nk = nk.replace("audio_mlp.down_proj", "audio_feed_forward.w2")

        # Dual LM heads — flat, no surrounding underscore boundary.
        nk = nk.replace("text_lm_head", "text_output")
        nk = nk.replace("audio_lm_head", "audio_output")

        out[nk] = v
    return map_hf_to_meta_keys(out)


def load_higgs_v2_state_dict(model_dir: str | pathlib.Path) -> tuple[HiggsAudioV2Config, dict[str, torch.Tensor]]:
    """One-shot: read config.json + model.safetensors, return (cfg, remapped_state_dict)."""
    p = pathlib.Path(model_dir)
    cfg = HiggsAudioV2Config.from_json(p / "config.json")
    sd = load_hf_state_dict(p / "model.safetensors")
    return cfg, remap_higgs_to_tt(sd)


__all__ = [
    "HiggsAudioV2Config",
    "load_hf_state_dict",
    "remap_higgs_to_tt",
    "load_higgs_v2_state_dict",
]
