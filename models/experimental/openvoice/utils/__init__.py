# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for OpenVoice TTNN.
"""

from models.experimental.openvoice.utils.audio import (
    AudioProcessor,
    audio_to_spectrogram,
    load_audio,
    mel_spectrogram,
    save_audio,
    spectrogram_numpy,
    spectrogram_torch,
)
from models.experimental.openvoice.utils.bert_features import (
    LANGUAGE_CODES,
    TONE_CODES,
    BERTFeatureExtractor,
    get_bert_extractor,
)
from models.experimental.openvoice.utils.weight_loader import (
    TTNNParameterDict,
    convert_to_ttnn_tensor,
    load_openvoice_checkpoint,
    remove_weight_norm_from_state_dict,
    reshape_conv1d_to_conv2d_weight,
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
