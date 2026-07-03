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

All convolutions run on device via TTConv1d / TTConvTranspose1d / TTBlock1DDevice.
Requires device opened with l1_small_size=32768 for conv support on Blackhole.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import ttnn

from models.experimental.vibevoice.tt.ttnn_semantic_tokenizer import (
    ConvWeightsHost,
    Block1DWeightsHost,
    SemanticTokenizerWeights,
    _parse_depths,
    _get_conv_weights,
    TTConv1d,
    TTBlock1DDevice,
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


# ──────────────────────────────────────────────────────────────
# TTConvTranspose1d — SConvTranspose1d (causal) on device
# ──────────────────────────────────────────────────────────────


class TTConvTranspose1d:
    """SConvTranspose1d (causal, trim_right_ratio=1) via polyphase conv2d.

    A stride-S transposed conv with kernel K = 2*S is decomposed into S regular
    causal conv1d's of kernel size 2 (one per output phase), whose outputs are
    interleaved.  This avoids ttnn.conv_transpose2d (which mis-streams at small
    non-tile-aligned widths and is sharding-fragile) and reuses TTConv1d's working
    streaming cache.

    With W = ConvTranspose1d weight [in, out, K]:
        output[t*S + s] = x[t] * W[:, :, s] + x[t-1] * W[:, :, s+S]
    which is exactly the causal, right-trimmed SConvTranspose1d output (length T*S).
    Each phase is therefore a kernel-2 causal conv (taps x[t-1], x[t]); no extra trim.
    """

    def __init__(self, ctw: ConvTransposeWeightsHost, device, compute_dtype=ttnn.bfloat16):
        self.device = device
        self.stride = ctw.stride

        in_ch, out_ch, K = ctw.weight.shape
        S = self.stride
        assert K == 2 * S, f"polyphase assumes kernel == 2*stride (got K={K}, stride={S})"
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.K = K

        # One kernel-2 causal conv per phase; bias added in every phase (each output
        # position is produced by exactly one phase, so bias lands once per position).
        W = ctw.weight  # [in, out, K]
        self._phases = []
        for s in range(S):
            k_tm1 = W[:, :, s + S].transpose(0, 1).contiguous()  # [out, in] coeff of x[t-1]
            k_t = W[:, :, s].transpose(0, 1).contiguous()  # [out, in] coeff of x[t]
            phase_w = torch.stack([k_tm1, k_t], dim=2).contiguous()  # [out, in, 2]
            cw = ConvWeightsHost(weight=phase_w, bias=ctw.bias, stride=1, groups=1, causal_pad=1)
            self._phases.append(TTConv1d(cw, device, compute_dtype=compute_dtype))

    def reset_cache(self) -> None:
        for p in self._phases:
            p.reset_cache()

    def reset_cache_inplace(self) -> None:
        for p in self._phases:
            p.reset_cache_inplace()

    def __call__(self, x: ttnn.Tensor, use_cache: bool = False, is_final_chunk: bool = False) -> ttnn.Tensor:
        """x: [B, 1, T, in_ch] -> [B, 1, T*stride, out_ch]."""
        B, _, T, _ = x.shape
        S = self.stride

        phase_outs = [p(x, use_cache=use_cache, is_final_chunk=is_final_chunk) for p in self._phases]
        if S == 1:
            return phase_outs[0]

        # Interleave phases: concat along channel -> [B,1,T,S*out_ch], reshape to
        # [B,1,T*S,out_ch] so position t*S+s carries phase s (row-major flat order).
        cat = ttnn.concat(phase_outs, dim=3)
        if cat.layout != ttnn.ROW_MAJOR_LAYOUT:
            cat = ttnn.to_layout(cat, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.reshape(cat, [B, 1, T * S, self.out_ch])
        # Downstream ops (e.g. ttnn.rms_norm in the next block) require TILE layout.
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)


# ──────────────────────────────────────────────────────────────
# TTAcousticTokenizer
# ──────────────────────────────────────────────────────────────


class TTAcousticTokenizer:
    """TTNN port of VibeVoiceAcousticTokenizerModel.

    encode: [B, 1, 1, T] → [B, 1, T_enc, vae_dim=64]
    decode: [B, 1, T_enc, vae_dim=64] → [B, 1, 1, T_audio]

    All convolutions run on device via TTConv1d / TTConvTranspose1d.
    Device must be opened with l1_small_size=32768 for conv support on Blackhole.
    """

    def __init__(self, weights: AcousticTokenizerWeights, device):
        self.device = device
        self._encoder_tt = TTSemanticTokenizer(weights.encoder, device)

        dec = weights.decoder
        self._dec_input_conv = TTConv1d(dec.input_conv, device)
        self._dec_stage_blocks = [[TTBlock1DDevice(bw, device) for bw in stage_blocks] for stage_blocks in dec.stages]
        self._dec_upsample_convs = [TTConvTranspose1d(ctw, device) for ctw in dec.upsample_convs]
        self._dec_head_conv = TTConv1d(dec.head_conv, device)

    def reset_decode_cache(self) -> None:
        """Clear all decoder streaming caches (call before a new speech segment)."""
        self._dec_input_conv.reset_cache()
        for stage in self._dec_stage_blocks:
            for blk in stage:
                blk.reset_cache()
        for up_conv in self._dec_upsample_convs:
            up_conv.reset_cache()
        self._dec_head_conv.reset_cache()

    def reset_decode_cache_inplace(self) -> None:
        """Zero all decoder streaming caches IN PLACE (llama-pattern trace; stable addresses)."""
        self._dec_input_conv.reset_cache_inplace()
        for stage in self._dec_stage_blocks:
            for blk in stage:
                blk.reset_cache_inplace()
        for up_conv in self._dec_upsample_convs:
            up_conv.reset_cache_inplace()
        self._dec_head_conv.reset_cache_inplace()

    def encode(self, audio: ttnn.Tensor, use_cache: bool = False, is_final_chunk: bool = False) -> ttnn.Tensor:
        """audio: [B, 1, 1, T] → [B, 1, T_enc, vae_dim]"""
        return self._encoder_tt.forward(audio, use_cache=use_cache, is_final_chunk=is_final_chunk)

    def decode(self, latents: ttnn.Tensor, use_cache: bool = False, is_final_chunk: bool = False) -> ttnn.Tensor:
        """latents: [B, 1, T_enc, vae_dim] → [B, 1, 1, T_audio] (all ops on device)"""
        B = latents.shape[0]

        x = latents if latents.dtype == ttnn.bfloat16 else ttnn.typecast(latents, ttnn.bfloat16)

        x = self._dec_input_conv(x, use_cache=use_cache, is_final_chunk=is_final_chunk)
        for blk in self._dec_stage_blocks[0]:
            x = blk(x, use_cache=use_cache, is_final_chunk=is_final_chunk)

        for up_conv, blocks in zip(self._dec_upsample_convs, self._dec_stage_blocks[1:]):
            x = up_conv(x, use_cache=use_cache, is_final_chunk=is_final_chunk)
            for blk in blocks:
                x = blk(x, use_cache=use_cache, is_final_chunk=is_final_chunk)

        x = self._dec_head_conv(x, use_cache=use_cache, is_final_chunk=is_final_chunk)  # [B, 1, T_audio, 1]
        T_audio = x.shape[2]
        return ttnn.reshape(x, [B, 1, 1, T_audio])

    def __call__(self, audio: ttnn.Tensor) -> ttnn.Tensor:
        return self.encode(audio)
