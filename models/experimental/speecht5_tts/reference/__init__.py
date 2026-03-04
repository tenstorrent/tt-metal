# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SpeechT5 TTS Reference Implementations

Op-by-op PyTorch implementations validated against HuggingFace (PCC = 1.0).
Ready for TTNN translation.
"""

from .speecht5_encoder import (
    SpeechT5Encoder,
    SpeechT5Config,
    load_from_huggingface as load_encoder_from_huggingface,
)
from .speecht5_decoder import (
    SpeechT5Decoder,
    SpeechT5DecoderConfig,
    SpeechDecoderPrenet,
    load_from_huggingface as load_decoder_from_huggingface,
)
from .speecht5_postnet import (
    SpeechT5SpeechDecoderPostnet,
    SpeechT5PostNetConfig,
    load_from_huggingface as load_postnet_from_huggingface,
)
from .speecht5_full_model import (
    SpeechT5FullReference,
    load_full_reference_from_huggingface,
)

__all__ = [
    "SpeechT5Config",
    "SpeechT5Encoder",
    "SpeechT5Decoder",
    "SpeechT5DecoderConfig",
    "SpeechDecoderPrenet",
    "SpeechT5SpeechDecoderPostnet",
    "SpeechT5PostNetConfig",
    "SpeechT5FullReference",
    "load_encoder_from_huggingface",
    "load_decoder_from_huggingface",
    "load_postnet_from_huggingface",
    "load_full_reference_from_huggingface",
]
