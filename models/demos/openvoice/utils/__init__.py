# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for OpenVoice TTNN.
"""

from models.demos.openvoice.utils.weight_loader import (
    load_openvoice_checkpoint,
    remove_weight_norm_from_state_dict,
    reshape_conv1d_to_conv2d_weight,
    convert_to_ttnn_tensor,
    TTNNParameterDict,
)
from models.demos.openvoice.utils.audio import (
    load_audio,
    save_audio,
    spectrogram_torch,
    spectrogram_numpy,
    mel_spectrogram,
    audio_to_spectrogram,
    AudioProcessor,
)
from models.demos.openvoice.utils.bert_features import (
    BERTFeatureExtractor,
    get_bert_extractor,
    LANGUAGE_CODES,
    TONE_CODES,
)

__all__ = [
    # Weight loading
    "load_openvoice_checkpoint",
    "remove_weight_norm_from_state_dict",
    "reshape_conv1d_to_conv2d_weight",
    "convert_to_ttnn_tensor",
    "TTNNParameterDict",
    # Audio processing
    "load_audio",
    "save_audio",
    "spectrogram_torch",
    "spectrogram_numpy",
    "mel_spectrogram",
    "audio_to_spectrogram",
    "AudioProcessor",
    # BERT features
    "BERTFeatureExtractor",
    "get_bert_extractor",
    "LANGUAGE_CODES",
    "TONE_CODES",
]
