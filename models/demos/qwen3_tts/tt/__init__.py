# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.demos.qwen3_tts.tt.model_config import (
    Qwen3TTSTalkerConfig,
    Qwen3TTSCodePredictorConfig,
    get_compute_kernel_config,
    get_compute_kernel_config_hifi4,
)
from models.demos.qwen3_tts.tt.rmsnorm import RMSNorm, QKNorm
from models.demos.qwen3_tts.tt.mlp import MLP
from models.demos.qwen3_tts.tt.attention import Attention
from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
from models.demos.qwen3_tts.tt.talker import Talker
from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS, create_qwen3_tts_model
from models.demos.qwen3_tts.tt.kv_cache import KVCache, create_kv_cache
from models.demos.qwen3_tts.tt.rope import (
    compute_rope_frequencies,
    compute_mrope_frequencies,
    get_rope_tensors,
    get_mrope_tensors,
    get_transformation_mat,
)
from models.demos.qwen3_tts.tt.generator import Qwen3TTSGenerator, create_generator
from models.demos.qwen3_tts.tt.speech_tokenizer import (
    SpeechTokenizerConfig,
    TtSpeechTokenizerDecoder,
    extract_speech_tokenizer_weights,
)

__all__ = [
    # Configs
    "Qwen3TTSTalkerConfig",
    "Qwen3TTSCodePredictorConfig",
    "get_compute_kernel_config",
    "get_compute_kernel_config_hifi4",
    # Building blocks
    "RMSNorm",
    "QKNorm",
    "MLP",
    "Attention",
    "DecoderLayer",
    # Models
    "Talker",
    "CodePredictor",
    "Qwen3TTS",
    "create_qwen3_tts_model",
    # KV Cache
    "KVCache",
    "create_kv_cache",
    # RoPE
    "compute_rope_frequencies",
    "compute_mrope_frequencies",
    "get_rope_tensors",
    "get_mrope_tensors",
    "get_transformation_mat",
    # Generator
    "Qwen3TTSGenerator",
    "create_generator",
    # Speech Tokenizer
    "SpeechTokenizerConfig",
    "TtSpeechTokenizerDecoder",
    "extract_speech_tokenizer_weights",
]
