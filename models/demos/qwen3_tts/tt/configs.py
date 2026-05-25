# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pure-Python configuration dataclasses for Qwen3-TTS components.
No ttnn dependency — can be imported in any environment.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CodePredictorConfig:
    """Configuration for the 5-layer Code Predictor transformer.

    Key dimensions:
        - hidden_size (1024): transformer hidden dim
        - talker_hidden_size (2048): input projection source dim
        - codec_embedding_dim (2048): codec embeddings are in Talker's space, NOT CP's
        - vocab_size (2048): codebook vocabulary per group
        - num_code_groups (16): total codebooks (CB0 from Talker, CB1-CB15 from CP)
    """

    hidden_size: int = 1024
    talker_hidden_size: int = 2048
    num_layers: int = 5
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    vocab_size: int = 2048
    num_code_groups: int = 16
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 65536

    @property
    def num_cb_predict(self):
        return self.num_code_groups - 1

    @classmethod
    def from_dict(cls, d: dict) -> "CodePredictorConfig":
        return cls(
            hidden_size=d.get("hidden_size", 1024),
            talker_hidden_size=d.get("talker_hidden_size", 2048),
            num_layers=d.get("num_hidden_layers", 5),
            num_heads=d.get("num_attention_heads", 16),
            num_kv_heads=d.get("num_key_value_heads", 8),
            head_dim=d.get("head_dim", 128),
            intermediate_size=d.get("intermediate_size", 3072),
            vocab_size=d.get("vocab_size", 2048),
            num_code_groups=d.get("num_code_groups", 16),
            rope_theta=d.get("rope_theta", 1000000.0),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            max_position_embeddings=d.get("max_position_embeddings", 65536),
        )


@dataclass
class SpeakerEncoderConfig:
    """Configuration for the ECAPA-TDNN speaker encoder.

    Default values match Qwen3-TTS-12Hz-1.7B-Base (12M params).
    Note: HF config default says enc_dim=1024, but actual model weights
    have fc output of 2048. The config.json for the 1.7B model uses enc_dim=2048.
    """

    enc_dim: int = 2048
    mel_dim: int = 128
    sample_rate: int = 24000
    enc_channels: List[int] = field(default_factory=lambda: [512, 512, 512, 512, 1536])
    enc_kernel_sizes: List[int] = field(default_factory=lambda: [5, 3, 3, 3, 1])
    enc_dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 1])
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerEncoderConfig":
        return cls(
            enc_dim=d.get("enc_dim", 2048),
            mel_dim=d.get("mel_dim", 128),
            sample_rate=d.get("sample_rate", 24000),
            enc_channels=d.get("enc_channels", [512, 512, 512, 512, 1536]),
            enc_kernel_sizes=d.get("enc_kernel_sizes", [5, 3, 3, 3, 1]),
            enc_dilations=d.get("enc_dilations", [1, 2, 3, 4, 1]),
            enc_attention_channels=d.get("enc_attention_channels", 128),
            enc_res2net_scale=d.get("enc_res2net_scale", 8),
            enc_se_channels=d.get("enc_se_channels", 128),
        )


@dataclass
class VocoderConfig:
    """Configuration for the Code2Wav vocoder (Qwen3TTSTokenizerV2Decoder).

    Default values match speech_tokenizer/config.json from 1.7B-Base.
    Architecture: RVQ dequant → CausalConv → 8L Transformer (SW=72) →
                  2x TransConv upsample → 4x DecoderBlock (SnakeBeta) → waveform
    Total upsample: 2*2*8*5*4*3 = 1920x → 12.5 Hz → 24 kHz
    """

    codebook_size: int = 2048
    codebook_dim: int = 512
    hidden_size: int = 512
    latent_dim: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    intermediate_size: int = 1024
    sliding_window: int = 72
    num_quantizers: int = 16
    decoder_dim: int = 1536
    upsample_rates: List[int] = field(default_factory=lambda: [8, 5, 4, 3])
    upsampling_ratios: List[int] = field(default_factory=lambda: [2, 2])
    layer_scale_initial_scale: float = 0.01
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8000
    output_sample_rate: int = 24000

    @classmethod
    def from_dict(cls, d: dict) -> "VocoderConfig":
        return cls(
            codebook_size=d.get("codebook_size", 2048),
            codebook_dim=d.get("codebook_dim", 512),
            hidden_size=d.get("hidden_size", 512),
            latent_dim=d.get("latent_dim", 1024),
            num_hidden_layers=d.get("num_hidden_layers", 8),
            num_attention_heads=d.get("num_attention_heads", 16),
            num_key_value_heads=d.get("num_key_value_heads", 16),
            head_dim=d.get("head_dim", 64),
            intermediate_size=d.get("intermediate_size", 1024),
            sliding_window=d.get("sliding_window", 72),
            num_quantizers=d.get("num_quantizers", 16),
            decoder_dim=d.get("decoder_dim", 1536),
            upsample_rates=list(d.get("upsample_rates", [8, 5, 4, 3])),
            upsampling_ratios=list(d.get("upsampling_ratios", [2, 2])),
            layer_scale_initial_scale=d.get("layer_scale_initial_scale", 0.01),
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=d.get("rope_theta", 10000.0),
            max_position_embeddings=d.get("max_position_embeddings", 8000),
            output_sample_rate=d.get("output_sample_rate", 24000),
        )
