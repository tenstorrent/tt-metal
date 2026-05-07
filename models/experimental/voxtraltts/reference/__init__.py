# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference helpers for Voxtral TTS."""

from .functional import (
    VoxtralAcousticConfig,
    VoxtralTextConfig,
    acoustic_attention,
    acoustic_transformer_layer,
    compute_rope_frequencies,
    get_default_acoustic_config,
    get_default_text_config,
    rms_norm,
    swiglu_mlp,
    text_attention,
    text_decoder_layer,
)
from .voxtral_config import (
    DEFAULT_VOXTRAL_MODEL,
    VoxtralAcousticTransformerConfig,
    VoxtralAudioEncodingConfig,
    VoxtralAudioModelConfig,
    VoxtralAudioTokenizerConfig,
    VoxtralConfig,
    load_voxtral_config,
)
from .voxtral_request import compose_speech_request, load_mistral_tokenizer


def __getattr__(name: str):
    if name == "VoxtralCPUReference":
        from .cpu_reference import VoxtralCPUReference

        return VoxtralCPUReference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_VOXTRAL_MODEL",
    "VoxtralCPUReference",
    "VoxtralAcousticConfig",
    "VoxtralTextConfig",
    "VoxtralAcousticTransformerConfig",
    "VoxtralAudioEncodingConfig",
    "VoxtralAudioModelConfig",
    "VoxtralAudioTokenizerConfig",
    "VoxtralConfig",
    "compose_speech_request",
    "acoustic_attention",
    "acoustic_transformer_layer",
    "compute_rope_frequencies",
    "get_default_acoustic_config",
    "get_default_text_config",
    "load_mistral_tokenizer",
    "load_voxtral_config",
    "rms_norm",
    "swiglu_mlp",
    "text_attention",
    "text_decoder_layer",
]
