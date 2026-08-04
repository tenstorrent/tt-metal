# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One place for every dtype, memory config and shape constant in the port.

Centralised for the reason DD-1 in CLAUDE.md records: memory configuration is
where bring-ups go wrong, and hunting a wrong memory config across a dozen files
is far more expensive than looking it up in one.

Every number here is read from the actual cosyvoice.yaml of FunAudioLLM/CosyVoice-300M
and confirmed against captured tensor shapes in tests/golden/manifest.json. Nothing
is assumed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import ttnn

# Weights may be stored bfloat8_b to halve bandwidth. On TTM-R1 this bought 5-7%
# at large batch with PCC >= 0.99 preserved (CLAUDE.md Stage 5), so it is offered
# but stays opt-in until measured on this model.
WEIGHT_BF8 = os.environ.get("COSYVOICE_WEIGHT_BF8", "0") == "1"
WEIGHTS_DTYPE = ttnn.bfloat8_b if WEIGHT_BF8 else ttnn.bfloat16
ACTIVATIONS_DTYPE = ttnn.bfloat16

# conv2d/conv_transpose2d allocate from the L1_SMALL bank; l1_small_size=0 fails
# with "bank size is 0 B" (CLAUDE.md DD-1).
L1_SMALL_SIZE = 32768
TRACE_REGION_SIZE = 23887872


@dataclass(frozen=True)
class LLMConfig:
    """cosyvoice.llm.llm.TransformerLM."""

    text_token_size: int = 51866  # Whisper multilingual tokenizer, 100 languages
    speech_token_size: int = 4096
    eos_token: int = 4096  # == speech_token_size
    text_encoder_input_size: int = 512
    llm_input_size: int = 1024
    llm_output_size: int = 1024
    spk_embed_dim: int = 192

    # text_encoder: ConformerEncoder
    text_encoder_blocks: int = 6
    text_encoder_heads: int = 16
    text_encoder_dim: int = 1024
    text_encoder_ffn: int = 4096

    # llm: TransformerEncoder (the autoregressive decoder)
    ar_blocks: int = 14
    ar_heads: int = 16
    ar_dim: int = 1024
    ar_ffn: int = 4096
    ar_head_dim: int = 64  # ar_dim // ar_heads

    # Both encoders use ESPnet relative-position attention, NOT RoPE. The
    # wormhole LLM demos assume RoPE, so they are a structural reference only.
    pos_enc: str = "rel_pos_espnet"
    self_attn: str = "rel_selfattn"
    use_cnn_module: bool = False  # -> cnn_cache is always [n_layers,0,0,0]
    macaron_style: bool = False

    # RAS sampling. NOTE the repetition-aware retry falls back to a plain
    # multinomial over the FULL 4097 vocabulary -- not a re-draw from the
    # truncated distribution. ttnn.sampling covers the primary path only.
    top_p: float = 0.8
    top_k: int = 25
    win_size: int = 10
    tau_r: float = 0.1

    max_token_text_ratio: float = 20.0
    min_token_text_ratio: float = 2.0

    @property
    def kv_cache_shape(self):
        """att_cache as the reference materialises it: K and V concatenated on
        the last axis, confirmed from a captured [14, 16, 209, 128] tensor."""
        return (self.ar_blocks, self.ar_heads, None, 2 * self.ar_head_dim)


@dataclass(frozen=True)
class FlowConfig:
    """cosyvoice.flow.flow.MaskedDiffWithXvec."""

    input_size: int = 512
    output_size: int = 80  # mel bins
    spk_embed_dim: int = 192
    vocab_size: int = 4096
    input_frame_rate: int = 50  # Hz; mel is 22050/256 ~= 86 Hz

    # encoder: ConformerEncoder
    encoder_blocks: int = 6
    encoder_heads: int = 8
    encoder_dim: int = 512
    encoder_ffn: int = 2048

    length_regulator_channels: int = 80
    sampling_ratios: tuple = (1, 1, 1, 1)

    # decoder: ConditionalCFM
    cfm_in_channels: int = 240
    solver: str = "euler"
    t_scheduler: str = "cosine"
    sigma_min: float = 1e-6
    # Non-zero => classifier-free guidance. The reference ALREADY batches the
    # conditional and unconditional rows into one 2-row call (x_in = zeros[2,80,T],
    # row 0 conditional, row 1 zeroed). That is a structure to preserve, not an
    # optimisation to add.
    inference_cfg_rate: float = 0.7
    n_timesteps: int = 10  # hardcoded at flow.py:inference

    # estimator: ConditionalDecoder, a UNet-1D
    est_in_channels: int = 320
    est_out_channels: int = 80
    est_channels: tuple = (256, 256)
    est_n_blocks: int = 4
    est_num_mid_blocks: int = 12  # the RTF hot spot: 10 steps x 12 mid-blocks
    est_num_heads: int = 8
    est_attention_head_dim: int = 64
    est_act_fn: str = "gelu"


@dataclass(frozen=True)
class HiFTConfig:
    """cosyvoice.hifigan.generator.HiFTGenerator (HiFTNet = NSF + iSTFTNet)."""

    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 22050
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: float = 10.0

    upsample_rates: tuple = (8, 8)
    upsample_kernel_sizes: tuple = (16, 16)
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    source_resblock_kernel_sizes: tuple = (7, 11)
    source_resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5))

    n_fft: int = 16  # the crux: small enough that the DFT is a matmul
    hop_len: int = 4
    lrelu_slope: float = 0.1
    audio_limit: float = 0.99

    @property
    def total_upsample(self) -> int:
        """8 * 8 * 4 = 256, which must equal the mel hop_size. It does."""
        u = 1
        for r in self.upsample_rates:
            u *= r
        return u * self.hop_len

    @property
    def sinegen_variant(self) -> str:
        """22050 Hz selects SineGen (type 1), the implementation that integrates
        phase with cumsum over the AUDIO-RATE signal -- 72 192 samples for 3.3 s.
        In bfloat16 an accumulator reaching ~1e3 loses the ~1e-2 increments
        entirely, so this path needs ttnn.cumsum(dtype=float32)."""
        return "1" if self.sampling_rate == 22050 else "2"


@dataclass(frozen=True)
class MelConfig:
    """Front-end for zero-shot reference audio."""

    n_fft: int = 1024
    num_mels: int = 80
    hop_size: int = 256
    win_size: int = 1024
    fmin: int = 0
    fmax: int = 8000
    center: bool = False


@dataclass(frozen=True)
class CosyVoiceConfig:
    sample_rate: int = 22050
    llm: LLMConfig = field(default_factory=LLMConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    hift: HiFTConfig = field(default_factory=HiFTConfig)
    mel: MelConfig = field(default_factory=MelConfig)

    weights_dtype: object = WEIGHTS_DTYPE
    activations_dtype: object = ACTIVATIONS_DTYPE

    # Mode -> checkpoint. CosyVoice-300M ships no spk2info.pt, so it has zero
    # named speakers and physically cannot do SFT or instruct.
    checkpoint_for_mode: dict = field(
        default_factory=lambda: {
            "sft": "CosyVoice-300M-SFT",
            "zero_shot": "CosyVoice-300M",
            "cross_lingual": "CosyVoice-300M",
            "instruct": "CosyVoice-300M-Instruct",
        }
    )


DEFAULT = CosyVoiceConfig()
