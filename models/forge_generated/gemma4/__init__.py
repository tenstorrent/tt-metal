"""Gemma4 — shared module library for prefill and decode.

This package mirrors the tt-metal `tt_transformers` pattern: per-module
classes that own their weight ttnn.Tensors and emit the same TTNN op
sequence as the legacy `gemma4_{prefill,decode}/model.py` helpers.

During the migration (Phase 1), classes are bootstrapped from the
existing consteval cache via `<Class>.from_consteval(...)`. Phase 2
adds `<Class>.from_state_dict(...)` for HF-driven loading. Phase 3
deletes the codegen-derived `gemma4_{prefill,decode}/` files entirely.

See docs/superpowers/specs/2026-04-25-gemma4-metal-alignment-design.md.
"""
from gemma4.rms_norm import RMSNorm
from gemma4.feed_forward import FeedForward
from gemma4.scaled_embedding import ScaledEmbedding
from gemma4.lm_head import LMHead
from gemma4.prelude import (
    SlidingPreludeDecode,
    SlidingPreludePrefill,
    FullPreludeDecode,
    FullPreludePrefill,
)
from gemma4.attention import Attention
from gemma4.decoder_layer import SlidingDecoderLayer, FullDecoderLayer
from gemma4.rope import RoPESetup
from gemma4.runtime_inputs import (
    synthesize_prefill_inputs,
    synthesize_decode_inputs,
)
from gemma4.layer_table import LAYER_TABLE_PREFILL, LAYER_TABLE_DECODE
from gemma4.model import Gemma4Model, Gemma4ForCausalLM

__all__ = [
    "RMSNorm",
    "FeedForward",
    "ScaledEmbedding",
    "LMHead",
    "SlidingPreludeDecode",
    "SlidingPreludePrefill",
    "FullPreludeDecode",
    "FullPreludePrefill",
    "Attention",
    "SlidingDecoderLayer",
    "FullDecoderLayer",
    "RoPESetup",
    "synthesize_prefill_inputs",
    "synthesize_decode_inputs",
    "LAYER_TABLE_PREFILL",
    "LAYER_TABLE_DECODE",
    "Gemma4Model",
    "Gemma4ForCausalLM",
]
