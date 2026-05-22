# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice Acoustic Tokenizer (encoder + decoder) — TTNN port.

Reference: VibeVoiceAcousticTokenizerModel in modular_vibevoice_tokenizer.py

Architecture:
  Encoder: identical to semantic tokenizer (vae_dim=64 instead of 128)
  Decoder (reverse of encoder):
    upsample_layers[0]: SConv1d(vae_dim → n_filters*2^(n_stages-1), K=7)
    For i in 0..n_stages-1:
      upsample_layers[i] applied first (SConvTranspose1d for i>0)
      stages[i]: depth[i] Block1D blocks (same structure as encoder)
    Final head: SConv1d(n_filters → 1, K=7)

SConvTranspose1d non-streaming: apply nn.ConvTranspose1d(no padding),
then trim (kernel_size - stride) samples from the right (causal, trim_right_ratio=1).

Note: All convolutions run host-side via torch.nn.functional (Blackhole conv bug).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.vibevoice.tt.ttnn_semantic_tokenizer import (
    ConvWeightsHost,
    Block1DWeightsHost,
    SemanticTokenizerWeights,
    _parse_depths,
    _get_conv_weights,
    _apply_conv1d,
    _block1d_forward,
    TTSemanticTokenizer,
    preprocess_semantic_tokenizer_weights,
)
from models.experimental.vibevoice.tt.vibevoice_config import SemanticTokenizerConfig, TokenizerConfig


@dataclass
class ConvTransposeWeightsHost:
    """Weights for SConvTranspose1d (causal, trim_right_ratio=1)."""

    weight: torch.Tensor  # [in_ch, out_ch, K] — PyTorch ConvTranspose1d format
    bias: Optional[torch.Tensor]
    stride: int
    trim_right: int  # = kernel_size - stride, trimmed from right after transposed conv


@dataclass
class AcousticDecoderWeights:
    input_conv: ConvWeightsHost  # upsample_layers[0]: SConv1d
    upsample_convs: List[ConvTransposeWeightsHost]  # upsample_layers[1..n_ratios]
    stages: List[List[Block1DWeightsHost]]  # decoder Block1D stages
    head_conv: ConvWeightsHost  # final SConv1d → 1 channel
    eps: float


@dataclass
class AcousticTokenizerWeights:
    encoder: SemanticTokenizerWeights
    decoder: AcousticDecoderWeights
    config: TokenizerConfig


def _get_block_weights(hf_state: dict, prefix: str, dim: int, eps: float, causal: bool = True) -> Block1DWeightsHost:
    """Load Block1DWeightsHost for any stage prefix (encoder or decoder)."""
    ffn_dim_default = 4 * dim

    dw_prefix = f"{prefix}.mixer.conv.conv.conv"
    dw = _get_conv_weights(
        hf_state, dw_prefix, in_ch=dim, out_ch=dim, kernel_size=7, stride=1, groups=dim, causal=causal
    )

    norm_w = hf_state.get(f"{prefix}.norm.weight", torch.ones(dim)).float()
    ffn_norm_w = hf_state.get(f"{prefix}.ffn_norm.weight", torch.ones(dim)).float()

    l1_w = hf_state.get(f"{prefix}.ffn.linear1.weight", torch.zeros(ffn_dim_default, dim)).float()
    l1_b = hf_state.get(f"{prefix}.ffn.linear1.bias", None)
    l2_w = hf_state.get(f"{prefix}.ffn.linear2.weight", torch.zeros(dim, ffn_dim_default)).float()
    l2_b = hf_state.get(f"{prefix}.ffn.linear2.bias", None)

    gamma = hf_state.get(f"{prefix}.gamma", None)
    ffn_gamma = hf_state.get(f"{prefix}.ffn_gamma", None)

    ffn_dim = l1_w.shape[0]
    return Block1DWeightsHost(
        dw_conv=dw,
        norm_w=norm_w.contiguous(),
        ffn_norm_w=ffn_norm_w.contiguous(),
        linear1_w=l1_w.contiguous(),
        linear1_b=l1_b.float().contiguous() if l1_b is not None else None,
        linear2_w=l2_w.contiguous(),
        linear2_b=l2_b.float().contiguous() if l2_b is not None else None,
        gamma=gamma.float().contiguous() if gamma is not None else None,
        ffn_gamma=ffn_gamma.float().contiguous() if ffn_gamma is not None else None,
        dim=dim,
        ffn_dim=ffn_dim,
        eps=eps,
    )


def preprocess_acoustic_tokenizer_weights(
    hf_state: Dict[str, torch.Tensor],
    device,
    config: TokenizerConfig,
) -> AcousticTokenizerWeights:
    """Build AcousticTokenizerWeights from the hf_state dict."""

    # ── Encoder ─────────────────────────────────────────────────────────────
    # Reuse semantic tokenizer preprocessing (encoder.* keys in hf_state)
    sem_cfg = SemanticTokenizerConfig(
        vae_dim=config.vae_dim,
        causal=config.causal,
        encoder_n_filters=config.encoder_n_filters,
        encoder_ratios=config.encoder_ratios,
        encoder_depths=config.encoder_depths,
        layernorm=config.layernorm,
        layernorm_eps=config.layernorm_eps,
        conv_bias=config.conv_bias,
        mixer_layer=getattr(config, "mixer_layer", "depthwise_conv"),
    )
    encoder_weights = preprocess_semantic_tokenizer_weights(hf_state, device, sem_cfg)

    # ── Decoder ─────────────────────────────────────────────────────────────
    eps = config.layernorm_eps
    causal = config.causal

    enc_depths = _parse_depths(config.encoder_depths)
    dec_depths = list(reversed(enc_depths))  # default: reverse encoder depths

    dec_ratios = config.decoder_ratios if config.decoder_ratios else config.encoder_ratios
    n_filters = config.decoder_n_filters
    n_stages = len(dec_depths)  # = 7
    vae_dim = config.vae_dim

    # Input conv: vae_dim → n_filters * 2^(n_stages-1)
    last_dim = n_filters * (2 ** (n_stages - 1))
    input_conv = _get_conv_weights(
        hf_state,
        "decoder.upsample_layers.0.0.conv.conv",
        in_ch=vae_dim,
        out_ch=last_dim,
        kernel_size=7,
        stride=1,
        causal=causal,
    )

    # Transposed convs: upsample_layers[1..n_ratios]
    upsample_convs: List[ConvTransposeWeightsHost] = []
    for i, ratio in enumerate(dec_ratios):
        in_ch = n_filters * (2 ** (n_stages - 1 - i))
        out_ch_exp = n_stages - 1 - i - 1
        out_ch = n_filters * (2**out_ch_exp) if out_ch_exp >= 0 else n_filters
        kernel_size = ratio * 2
        pref = f"decoder.upsample_layers.{i + 1}.0.convtr.convtr"
        w = hf_state.get(f"{pref}.weight", torch.zeros(in_ch, out_ch, kernel_size)).float().contiguous()
        b_raw = hf_state.get(f"{pref}.bias", None)
        b = b_raw.float().contiguous() if b_raw is not None else None
        upsample_convs.append(ConvTransposeWeightsHost(weight=w, bias=b, stride=ratio, trim_right=kernel_size - ratio))

    # Decoder stages
    dec_stages: List[List[Block1DWeightsHost]] = []
    for s_idx, depth in enumerate(dec_depths):
        dim = n_filters * (2 ** (n_stages - 1 - s_idx))
        blocks = [
            _get_block_weights(hf_state, f"decoder.stages.{s_idx}.{b_idx}", dim, eps, causal) for b_idx in range(depth)
        ]
        dec_stages.append(blocks)

    # Head conv: n_filters → 1 channel
    head_conv = _get_conv_weights(
        hf_state, "decoder.head.conv.conv", in_ch=n_filters, out_ch=1, kernel_size=7, stride=1, causal=causal
    )

    decoder = AcousticDecoderWeights(
        input_conv=input_conv,
        upsample_convs=upsample_convs,
        stages=dec_stages,
        head_conv=head_conv,
        eps=eps,
    )

    return AcousticTokenizerWeights(
        encoder=encoder_weights,
        decoder=decoder,
        config=config,
    )


def _apply_conv_transpose1d(x: torch.Tensor, ctw: ConvTransposeWeightsHost) -> torch.Tensor:
    """SConvTranspose1d (causal, trim_right_ratio=1) on [B, C, T] → [B, C_out, T*stride]."""
    y = F.conv_transpose1d(x, ctw.weight, ctw.bias, stride=ctw.stride)
    if ctw.trim_right > 0:
        y = y[..., : -ctw.trim_right]
    return y


class TTAcousticTokenizer:
    """TTNN port of VibeVoiceAcousticTokenizerModel.

    encode: [B, 1, 1, T] → [B, 1, T_enc, vae_dim=64]
    decode: [B, 1, T_enc, vae_dim=64] → [B, 1, 1, T_audio]

    All convolutions run host-side (Blackhole TTNN conv kernel bug).
    """

    def __init__(self, weights: AcousticTokenizerWeights, device):
        self.w = weights
        self.device = device
        self._encoder_tt = TTSemanticTokenizer(weights.encoder, device)

    def encode(self, audio: ttnn.Tensor) -> ttnn.Tensor:
        """audio: [B, 1, 1, T] → [B, 1, T_enc, vae_dim]"""
        return self._encoder_tt.forward(audio)

    def decode(self, latents: ttnn.Tensor) -> ttnn.Tensor:
        """latents: [B, 1, T_enc, vae_dim] → [B, 1, 1, T_audio]"""
        dec = self.w.decoder

        # To host, reshape to [B, vae_dim, T_enc] (BCT)
        x_host = ttnn.to_torch(latents).to(torch.float32)  # [B, 1, T_enc, vae_dim]
        B, _, T_enc, C = x_host.shape
        x = x_host.squeeze(1).permute(0, 2, 1)  # [B, vae_dim, T_enc]

        # Stage 0: input conv then block1d
        x = _apply_conv1d(x, dec.input_conv)
        for blk in dec.stages[0]:
            x = _block1d_forward(x, blk)

        # Upsample stages 1..n_ratios
        for ctw, blocks in zip(dec.upsample_convs, dec.stages[1:]):
            x = _apply_conv_transpose1d(x, ctw)
            for blk in blocks:
                x = _block1d_forward(x, blk)

        # Head conv → [B, 1, T_audio]
        x = _apply_conv1d(x, dec.head_conv)

        # [B, 1, 1, T_audio]
        x = x.to(torch.bfloat16).unsqueeze(1)
        return ttnn.as_tensor(
            x,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, audio: ttnn.Tensor) -> ttnn.Tensor:
        return self.encode(audio)
