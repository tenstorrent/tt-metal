# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B architecture configuration parsed from config.json."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Matmul K-reduction damping — restores a numeric property the long-form loop depends on.
#
# Before upstream #50250 ("Enable UnpackToDestFp32 in matmul when appropriate"), a matmul with
# fp32_dest_acc_en=True / packer_l1_acc=False wrote fp32 partial sums to an intermediate CB and
# unpacked them back as bf16 on every cross-K-block reload.  That truncation was toward zero, so
# each reload shaved a little magnitude off the running sum: measured across all 13 matmul configs
# this model runs, the pre-#50250 build came out systematically small by ~0.0169% per K-block
# reload (down_proj K=8960/ibw=2 = 140 blocks -> -2.354%; K=1536/ibw=2 = 24 blocks -> -0.407%;
# K=1536/ibw=4 = 12 blocks -> -0.194%).  The post-fix build is unbiased (+0.005% mean).
#
# That bias was an accidental per-matmul damping term, and the ~15k-step autoregressive audio loop
# was relying on it to keep its gain below 1.  Without it the loop runs away: rms 0.10 -> 0.32 with
# clipping from min 7 and unintelligible output from min 13, reproduced on two independent seeds
# (the measured per-frame excess gain is only ~1.0002, far smaller than the damping removed).
#
# Fold the same factor into the weights at load time — linear, so scaling the weight is equivalent
# to scaling the matmul output, and it costs nothing at runtime.  Set VV_MM_DAMP=0 to disable and
# get the raw post-rebase behaviour.
#
# IMPORTANT — the factor below is nominal, not what actually lands.  Callers apply it by scaling in
# fp32 and casting to bf16, and these factors (0.02%-2.4%) are near or below bf16's ~0.39% ULP, so
# rounding does not preserve the mean: measured on layer 0, q/o_proj asks for -0.203% and realizes
# -0.024% (94% of weights unchanged), while gate/up/k_proj ask for -0.406% and realize -0.561%.
# Only the large-K tensors (down_proj, -2.366% -> -2.356%) come out as written.
#
# That uneven realized pattern is DELIBERATELY KEPT.  Solving for the pre-rounding scale so every
# tensor realizes its nominal target was implemented and measured (`vv_damp_fix`), and it is much
# worse: dynamic range collapses at min 7 and pins at crest factor 3.1 — a constant drone, with
# whisper returning nothing from min 11 — versus the as-is build holding speech-like dynamics
# (crest 7.5-8.9) through 48 min with a single 2.25-min gap.  So this is an empirical stabilizer
# whose validated setting is the one bf16 rounding produces, NOT a faithful replay of the old
# build's numerics.  Do not "fix" the rounding without re-running a >=17 min render and checking
# crest factor + whisper; rms alone calls the broken version an improvement.
_DAMP_PER_K_BLOCK = 1.69e-4


def mm_damp(k_dim: int, in0_block_w: int = 2) -> float:
    """Nominal weight scale for a K-reduction (see above: the realized value differs).

    k_dim: the matmul's K (contraction) dimension.
    in0_block_w: K-block width in tiles from the program config (2 for most of this model's
                 pinned decode configs, 4 for the wq/wo 1536x1536 projections).
    """
    if os.environ.get("VV_MM_DAMP", "1") != "1":
        return 1.0
    return 1.0 - _DAMP_PER_K_BLOCK * (k_dim / 32.0 / in0_block_w)


def damp_weight(t: "torch.Tensor", k_dim: int, in0_block_w: int = 2) -> "torch.Tensor":
    """bf16 weight carrying the validated damping.  Scaling in fp32 then rounding once is the
    whole operation — the uneven realized pattern that produces (see above) is the point.

    Weakening this damping has been tried twice and fails the same way both times: replacing the
    pattern with the uniform nominal law, and scaling the validated pattern by 0.5, each collapse
    the dynamic range to a crest-factor ~3 drone by min 7-8 (normal rms, no intelligible speech)
    and then die.  The strength sits at a narrow optimum; do not tune it without a >=17 min render
    checked on crest factor and transcription, because rms alone reads the drone as healthy.
    """
    import torch

    return (t.to(torch.float32) * mm_damp(k_dim, in0_block_w)).to(torch.bfloat16)


@dataclass
class DecoderConfig:
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 8960
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-06
    max_position_embeddings: int = 32768
    head_dim: int = 128  # hidden_size // num_attention_heads = 1536 // 12 = 128


@dataclass
class DiffusionHeadConfig:
    hidden_size: int = 1536
    head_layers: int = 4
    head_ffn_ratio: float = 3.0
    rms_norm_eps: float = 1e-05
    latent_size: int = 64


@dataclass
class TokenizerConfig:
    vae_dim: int = 64
    fix_std: float = 0.5
    causal: bool = True
    encoder_n_filters: int = 32
    encoder_ratios: List[int] = field(default_factory=lambda: [8, 5, 5, 4, 2, 2])
    encoder_depths: str = "3-3-3-3-3-3-8"
    decoder_ratios: Optional[List[int]] = None
    decoder_n_filters: int = 32
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-05
    conv_bias: bool = True
    mixer_layer: str = "depthwise_conv"


@dataclass
class SemanticTokenizerConfig:
    vae_dim: int = 128
    causal: bool = True
    encoder_n_filters: int = 32
    encoder_ratios: List[int] = field(default_factory=lambda: [8, 5, 5, 4, 2, 2])
    encoder_depths: str = "3-3-3-3-3-3-8"
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-05
    conv_bias: bool = True
    mixer_layer: str = "depthwise_conv"


@dataclass
class VibeVoiceModelConfig:
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    diffusion_head: DiffusionHeadConfig = field(default_factory=DiffusionHeadConfig)
    acoustic_tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    semantic_tokenizer: SemanticTokenizerConfig = field(default_factory=SemanticTokenizerConfig)
    # Connector dims derived from tokenizer vae_dims → LM hidden_size
    acoustic_connector_input_dim: int = 64
    semantic_connector_input_dim: int = 128
    connector_output_dim: int = 1536  # = decoder.hidden_size


def load_vibevoice_model_config(model_path: str) -> VibeVoiceModelConfig:
    """Parse VibeVoice config.json into structured dataclasses."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        # Return defaults for the 1.5B variant when no config.json is present
        return VibeVoiceModelConfig()

    with open(cfg_path) as f:
        raw = json.load(f)

    dec_raw = raw.get("decoder_config", {})
    dec = DecoderConfig(
        hidden_size=dec_raw.get("hidden_size", 1536),
        num_hidden_layers=dec_raw.get("num_hidden_layers", 28),
        num_attention_heads=dec_raw.get("num_attention_heads", 12),
        num_key_value_heads=dec_raw.get("num_key_value_heads", 2),
        intermediate_size=dec_raw.get("intermediate_size", 8960),
        vocab_size=dec_raw.get("vocab_size", 151936),
        rope_theta=float(dec_raw.get("rope_theta", 1_000_000.0)),
        rms_norm_eps=dec_raw.get("rms_norm_eps", 1e-6),
        max_position_embeddings=dec_raw.get("max_position_embeddings", 32768),
    )
    dec.head_dim = dec.hidden_size // dec.num_attention_heads

    diff_raw = raw.get("diffusion_head_config", {})
    diff = DiffusionHeadConfig(
        hidden_size=diff_raw.get("hidden_size", dec.hidden_size),
        head_layers=diff_raw.get("head_layers", 4),
        head_ffn_ratio=diff_raw.get("head_ffn_ratio", 3.0),
        rms_norm_eps=diff_raw.get("rms_norm_eps", 1e-5),
        latent_size=diff_raw.get("latent_size", 64),
    )

    def _tok(raw_key: str, default_vae: int) -> TokenizerConfig:
        t = raw.get(raw_key, {})
        return TokenizerConfig(
            vae_dim=t.get("vae_dim", default_vae),
            fix_std=t.get("fix_std", 0.5),
            causal=t.get("causal", True),
            encoder_n_filters=t.get("encoder_n_filters", 32),
            encoder_ratios=t.get("encoder_ratios", [8, 5, 5, 4, 2, 2]),
            encoder_depths=t.get("encoder_depths", "3-3-3-3-3-3-8"),
            decoder_ratios=t.get("decoder_ratios", None),
            decoder_n_filters=t.get("decoder_n_filters", 32),
            layernorm=t.get("layernorm", "RMSNorm"),
            layernorm_eps=t.get("layernorm_eps", 1e-5),
            conv_bias=t.get("conv_bias", True),
            mixer_layer=t.get("mixer_layer", "depthwise_conv"),
        )

    def _sem_tok(raw_key: str) -> SemanticTokenizerConfig:
        t = raw.get(raw_key, {})
        return SemanticTokenizerConfig(
            vae_dim=t.get("vae_dim", 128),
            causal=t.get("causal", True),
            encoder_n_filters=t.get("encoder_n_filters", 32),
            encoder_ratios=t.get("encoder_ratios", [8, 5, 5, 4, 2, 2]),
            encoder_depths=t.get("encoder_depths", "3-3-3-3-3-3-8"),
            layernorm=t.get("layernorm", "RMSNorm"),
            layernorm_eps=t.get("layernorm_eps", 1e-5),
            conv_bias=t.get("conv_bias", True),
            mixer_layer=t.get("mixer_layer", "depthwise_conv"),
        )

    acoustic_cfg = _tok("acoustic_tokenizer_config", default_vae=64)
    semantic_cfg = _sem_tok("semantic_tokenizer_config")

    return VibeVoiceModelConfig(
        decoder=dec,
        diffusion_head=diff,
        acoustic_tokenizer=acoustic_cfg,
        semantic_tokenizer=semantic_cfg,
        acoustic_connector_input_dim=acoustic_cfg.vae_dim,
        semantic_connector_input_dim=semantic_cfg.vae_dim,
        connector_output_dim=dec.hidden_size,
    )
