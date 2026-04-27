"""Gemma4-31B-it forward implementation for tt-metal (1×4 mesh).

Per-module classes own their ttnn.Tensor weights and emit the
codegen-derived ttnn op sequences. Construction is HF-state_dict
driven via `Gemma4ForCausalLM.from_state_dict(hf, mesh_device,
is_decode=...)`; the runtime call path (`model(input_list)` →
logits tensor) takes only the runtime input list (KV caches,
position IDs, token IDs).
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
from gemma4.model import Gemma4ForCausalLM

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
    "Gemma4ForCausalLM",
]
