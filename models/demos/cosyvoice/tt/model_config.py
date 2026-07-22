# SPDX-FileCopyrightText: © 2026 Tenstorrent AI, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Frozen CosyVoice2-0.5B architecture config (Stage-1 bring-up).

Every constant below was verified by Phase 0.6 against the downloaded
`model_data/cosyvoice2-0.5B/cosyvoice2.yaml` (HF rev `eec1ae6c`) AND against the
state dicts `llm.pt` / `flow.pt` / `hift.pt` (tensor shapes). This closes §11.6
U1. `scripts/extract_config.py` is the regression harness that re-asserts these
values against the yaml on disk — run it whenever the checkpoint is refreshed.

CV2 flow estimator is a **UNet1D** (`CausalConditionalDecoder`, Matcha lineage),
NOT a DiT. The DiT variant (`CausalMaskedDiffWithDiT`) is CosyVoice 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# ---------------------------------------------------------------------------
# Global / shared
# ---------------------------------------------------------------------------
SEED = 1986
"""Reproducibility seed. The yaml sets `random.seed`, `numpy.random.seed`,
`torch.manual_seed`, `torch.cuda.manual_seed_all` all to 1986 at load time
(`cosyvoice2.yaml` lines 1-5). `gen_golden.py` must replicate these 4 calls."""

SAMPLE_RATE = 24000
TOKEN_FRAME_RATE = 25
TOKEN_MEL_RATIO = 2
SPEECH_TOKEN_SIZE = 6561  # FSQ codebook
SPEECH_TOKEN_VOCAB = SPEECH_TOKEN_SIZE + 3  # 6564 — incl. eos/fill/extra logits
LLM_INPUT_SIZE = 896
LLM_OUTPUT_SIZE = 896
SPK_EMBED_DIM = 192

# Streaming (OUT OF SCOPE for Stage 1 — non-streaming path only). Recorded so
# the config is complete; do NOT wire flow_cache / pre_lookahead machinery.
CHUNK_SIZE = 25  # streaming inference chunk size (tokens)
NUM_DECODING_LEFT_CHUNKS = -1


# ---------------------------------------------------------------------------
# Mel extraction (host-side, inference path = `feat_extractor`)
# ---------------------------------------------------------------------------
MEL_NUM_BINS = 80
MEL_N_FFT = 1920
MEL_HOP_SIZE = 480
MEL_WIN_SIZE = 1920
MEL_FMIN = 0
MEL_FMAX = 8000  # inference feat_extractor fmax (NOT the GAN
# training mel_spec_transform1, which is null)
MEL_CENTER = False


# ---------------------------------------------------------------------------
# LLM — Qwen2LM (Qwen2.5-0.5B backbone + CosyVoice speech heads)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    # Backbone = transformers.Qwen2ForCausalLM (Qwen2.5-0.5B).
    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14  # q_proj (896,896) / 64 = 14
    num_key_value_heads: int = 2  # k/v_proj (128,896) / 64 = 2 (GQA)
    head_dim: int = 64
    vocab_size: int = 151936  # Qwen text vocab (embed_tokens)
    # CosyVoice-specific heads (verified in llm.pt):
    speech_embedding_num: int = SPEECH_TOKEN_VOCAB  # 6564
    llm_embedding_num: int = 2  # sos=0, task_id=1
    llm_decoder_out: int = SPEECH_TOKEN_VOCAB  # 6564
    llm_input_size: int = 896
    llm_output_size: int = 896
    spk_embed_dim: int = 192
    mix_ratio: Tuple[int, int] = (5, 15)  # bistream text:speech — Stage-2
    top_p: float = 0.8
    top_k: int = 25
    win_size: int = 10
    tau_r: float = 0.1


LLM = LLMConfig()


# ---------------------------------------------------------------------------
# Flow — CausalMaskedDiffWithXvec (encoder + CFM decoder)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlowEncoderConfig:
    """UpsampleConformerEncoder (ESPnet rel-pos self-attention, NOT RoPE)."""

    output_size: int = 512
    attention_heads: int = 8
    linear_units: int = 2048
    num_blocks: int = 6
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    normalize_before: bool = True
    input_layer: str = "linear"
    pos_enc_layer_type: str = "rel_pos_espnet"  # §9 risk — needs fresh impl
    selfattention_layer_type: str = "rel_selfattn"
    input_size: int = 512
    use_cnn_module: bool = False
    macaron_style: bool = False
    static_chunk_size: int = 25  # = chunk_size


@dataclass(frozen=True)
class FlowCFMConfig:
    """CausalConditionalCFM decoder params."""

    in_channels: int = 240
    n_spks: int = 1
    spk_emb_dim: int = 80
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "cosine"
    training_cfg_rate: float = 0.2
    inference_cfg_rate: float = 0.7
    reg_loss_type: str = "l1"
    n_timesteps: int = 10  # hardcoded in flow.inference (NOT in yaml)


@dataclass(frozen=True)
class FlowEstimatorConfig:
    """CausalConditionalDecoder — UNet1D (Matcha lineage), NOT a DiT."""

    in_channels: int = 320  # [x(80), mu(80), spks(80), cond(80)]
    out_channels: int = 80
    channels: Tuple[int, ...] = (256,)
    dropout: float = 0.0
    attention_head_dim: int = 64
    n_blocks: int = 4
    num_mid_blocks: int = 12
    num_heads: int = 8
    act_fn: str = "gelu"
    static_chunk_size: int = 50  # = chunk_size * token_mel_ratio = 25*2
    num_decoding_left_chunks: int = -1


@dataclass(frozen=True)
class FlowConfig:
    input_size: int = 512
    output_size: int = 80
    spk_embed_dim: int = 192
    output_type: str = "mel"
    vocab_size: int = 6561
    input_frame_rate: int = 25
    only_mask_loss: bool = True
    token_mel_ratio: int = 2
    pre_lookahead_len: int = 3  # Stage-2 streaming — unused Stage 1
    encoder: FlowEncoderConfig = field(default_factory=FlowEncoderConfig)
    decoder: FlowCFMConfig = field(default_factory=FlowCFMConfig)
    estimator: FlowEstimatorConfig = field(default_factory=FlowEstimatorConfig)
    # Derived: input_embedding = Embedding(6561, 512);
    #          spk_embed_affine_layer = Linear(192, 80);
    #          encoder_proj = Linear(512, 80).


FLOW = FlowConfig()


# ---------------------------------------------------------------------------
# Vocoder — HiFTGenerator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class F0PredictorConfig:
    """ConvRNNF0Predictor — read cosyvoice/hifigan/f0_predictor.py (U17)."""

    num_class: int = 1
    in_channels: int = 80
    cond_channels: int = 512


@dataclass(frozen=True)
class VocoderConfig:
    in_channels: int = 80
    base_channels: int = 512
    nb_harmonics: int = 8
    sampling_rate: int = 24000
    nsf_alpha: float = 0.1
    nsf_sigma: float = 0.003
    nsf_voiced_threshold: int = 10
    upsample_rates: Tuple[int, ...] = (8, 5, 3)  # prod 120
    upsample_kernel_sizes: Tuple[int, ...] = (16, 11, 7)
    istft_n_fft: int = 16
    istft_hop_len: int = 4
    # total upsample = 120 * 4 = 480 = 24000 / 50 Hz (mel frame rate)
    resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    source_resblock_kernel_sizes: Tuple[int, ...] = (7, 7, 11)
    source_resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    lrelu_slope: float = 0.1
    audio_limit: float = 0.99
    sinegen_type: str = "2"  # SineGen2 — phase-interpolation variant
    f0_predictor: F0PredictorConfig = field(default_factory=F0PredictorConfig)


VOCODER = VocoderConfig()


# ---------------------------------------------------------------------------
# Stage-1 placement & dtype (per TTNN model-bringup guide)
# ---------------------------------------------------------------------------
DTYPE = "bfloat16"  # bf16 end-to-end in Stage 1
# math_fidelity resolved in Phase 2 (per TTNN guide §2) — placeholder.
ON_DEVICE_COMPONENTS = ("llm", "flow", "hift")
HOST_COMPONENTS = ("text_frontend", "speech_tokenizer", "speaker_encoder", "mel")
