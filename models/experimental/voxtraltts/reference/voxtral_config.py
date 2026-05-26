# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_VOXTRAL_MODEL = "mistralai/Voxtral-4B-TTS-2603"
DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN = 512


@dataclass(frozen=True)
class VoxtralAcousticTransformerConfig:
    input_dim: int = 3072
    dim: int = 3072
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    use_biases: bool = False
    rope_theta: float = 10000.0
    sigma: float = 1e-5
    sigma_max: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoxtralAcousticTransformerConfig":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__ if name in data})


@dataclass(frozen=True)
class VoxtralAudioEncodingConfig:
    codebook_pattern: str = "parallel"
    interleave_audio_tokens_per_segment: int = 8192
    interleave_text_tokens_per_segment: int = 8192
    single_trailing_segment: bool = False
    num_codebooks: int = 37
    sampling_rate: int = 24000
    frame_rate: float = 12.5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoxtralAudioEncodingConfig":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__ if name in data})


@dataclass(frozen=True)
class VoxtralAudioModelConfig:
    semantic_codebook_size: int = 8192
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    input_embedding_concat_type: str = "sum"
    p_uncond: float = 0.0
    text_feature_bugged: bool = False
    condition_dropped_token_id: int = 42
    audio_encoding_args: VoxtralAudioEncodingConfig = field(default_factory=VoxtralAudioEncodingConfig)
    acoustic_transformer_args: VoxtralAcousticTransformerConfig = field(
        default_factory=VoxtralAcousticTransformerConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoxtralAudioModelConfig":
        kwargs = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        kwargs["audio_encoding_args"] = VoxtralAudioEncodingConfig.from_dict(data.get("audio_encoding_args", {}))
        kwargs["acoustic_transformer_args"] = VoxtralAcousticTransformerConfig.from_dict(
            data.get("acoustic_transformer_args", {})
        )
        return cls(**kwargs)


@dataclass(frozen=True)
class VoxtralAudioTokenizerConfig:
    channels: int = 1
    sampling_rate: int = 24000
    pretransform_patch_size: int = 240
    patch_proj_kernel_size: int = 7
    semantic_codebook_size: int = 8192
    semantic_dim: int = 256
    acoustic_codebook_size: int = 21
    acoustic_dim: int = 36
    conv_weight_norm: bool = True
    causal: bool = True
    attn_sliding_window_size: int = 16
    half_attn_window_upon_downsampling: bool = True
    dim: int = 1024
    hidden_dim: int = 4096
    head_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 8
    qk_norm_eps: float = 1e-6
    qk_norm: bool = True
    use_biases: bool = False
    norm_eps: float = 0.01
    layer_scale: bool = True
    layer_scale_init: float = 0.01
    decoder_transformer_lengths_str: str = "2,2,2,2"
    decoder_convs_kernels_str: str = "3,4,4,4"
    decoder_convs_strides_str: str = "1,2,2,2"
    voice: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoxtralAudioTokenizerConfig":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__ if name in data})


@dataclass(frozen=True)
class VoxtralConfig:
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 131072
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5
    max_seq_len: int = 65536
    max_position_embeddings: int = 128000
    model_type: str = "voxtral_tts"
    tied_embeddings: bool = True
    use_biases: bool = False
    causal: bool = True
    audio_model_args: VoxtralAudioModelConfig = field(default_factory=VoxtralAudioModelConfig)
    audio_tokenizer_args: VoxtralAudioTokenizerConfig = field(default_factory=VoxtralAudioTokenizerConfig)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> "VoxtralConfig":
        multimodal = params.get("multimodal", {})
        kwargs = {name: params[name] for name in cls.__dataclass_fields__ if name in params}
        kwargs["audio_model_args"] = VoxtralAudioModelConfig.from_dict(multimodal.get("audio_model_args", {}))
        kwargs["audio_tokenizer_args"] = VoxtralAudioTokenizerConfig.from_dict(
            multimodal.get("audio_tokenizer_args", {})
        )
        return cls(**kwargs)


def _resolve_params_path(model_name_or_path: str) -> Path:
    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        return model_path / "params.json"

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to download Voxtral params.json from Hugging Face.") from exc

    return Path(
        hf_hub_download(
            repo_id=model_name_or_path,
            filename="params.json",
            local_files_only=os.getenv("CI") == "true",
        )
    )


def parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def audio_tokenizer_latent_dim(cfg: VoxtralAudioTokenizerConfig) -> int:
    """Continuous latent width before tokenizer decoder (``semantic_dim + acoustic_dim``)."""
    return int(cfg.semantic_dim + cfg.acoustic_dim)


def load_voxtral_config(model_name_or_path: str = DEFAULT_VOXTRAL_MODEL) -> VoxtralConfig:
    """Load Voxtral's Mistral-format params.json into a typed reference config."""

    params_path = _resolve_params_path(model_name_or_path)
    with open(params_path, "r") as f:
        params = json.load(f)
    return VoxtralConfig.from_params(params)
