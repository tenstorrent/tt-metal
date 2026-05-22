# VibeVoice-1.5B reference package (vendored for TT-Metal PCC).
from vibevoice.modular import (
    VibeVoiceConfig,
    VibeVoiceForConditionalGeneration,
    VibeVoiceForConditionalGenerationInference,
)
from vibevoice.processor import VibeVoiceProcessor, VibeVoiceTokenizerProcessor

__all__ = [
    "VibeVoiceConfig",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceForConditionalGenerationInference",
    "VibeVoiceProcessor",
    "VibeVoiceTokenizerProcessor",
]
