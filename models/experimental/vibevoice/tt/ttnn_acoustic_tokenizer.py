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
import torch.nn.functional as F
import ttnn

from models.experimental.vibevoice.tt.ttnn_semantic_tokenizer import (
    ConvWeightsHost,
    Block1DWeightsHost,
    SemanticTokenizerWeights,
    _parse_depths,
    _get_conv_weights,
    _HIFI4,
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


def _apply_conv_transpose1d(x: torch.Tensor, ctw: ConvTransposeWeightsHost) -> torch.Tensor:
    """SConvTranspose1d (causal, trim_right_ratio=1) on [B, C, T] → [B, C_out, T*stride]."""
    y = F.conv_transpose1d(x, ctw.weight, ctw.bias, stride=ctw.stride)
    if ctw.trim_right > 0:
        y = y[..., : -ctw.trim_right]
    return y


# ──────────────────────────────────────────────────────────────
# TTConvTranspose1d — SConvTranspose1d (causal) on device
# ──────────────────────────────────────────────────────────────


class TTConvTranspose1d:
    """SConvTranspose1d (causal, trim_right_ratio=1) on device via ttnn.conv_transpose2d.

    Applies standard ConvTranspose1d (no padding), then trims kernel_size - stride
    samples from the right. Input/output: [B, 1, T, C] NHWC.
    """

    def __init__(self, ctw: ConvTransposeWeightsHost, device):
        self.device = device
        self.stride = ctw.stride
        self.trim_right = ctw.trim_right

        in_ch, out_ch, K = ctw.weight.shape
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.K = K

        # Store raw host tensors in float32 so the kernel mirror/transpose in
        # prepare_conv_transpose2d_weights is done at full precision before
        # quantising to bfloat16 for the actual conv op.
        self._raw_weight = ctw.weight.to(torch.float32).unsqueeze(2).contiguous()  # [in, out, 1, K]
        self._raw_bias = ctw.bias.to(torch.float32).view(1, 1, 1, -1).contiguous() if ctw.bias is not None else None
        self.weight = None  # set on first call
        self.bias = None
        self._prepared_for_T = None

    def _prepare(self, B: int, T: int) -> None:
        """Preprocess weights/bias for conv_transpose2d using the prepare APIs."""
        w_tt = ttnn.as_tensor(
            self._raw_weight,
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
        self.weight = ttnn.prepare_conv_transpose2d_weights(
            weight_tensor=w_tt,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            batch_size=B,
            input_height=1,
            input_width=T,
            kernel_size=(1, self.K),
            stride=(1, self.stride),
            padding=(0, 0),
            dilation=(1, 1),
            has_bias=self._raw_bias is not None,
            groups=1,
            device=self.device,
            input_dtype=ttnn.bfloat16,
            conv_config=_cfg,
            compute_config=_HIFI4,
        )
        if self._raw_bias is not None:
            # prepare_conv_transpose2d_bias requires a HOST tensor
            b_tt = ttnn.from_torch(self._raw_bias, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
            self.bias = ttnn.prepare_conv_transpose2d_bias(
                bias_tensor=b_tt,
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                batch_size=B,
                input_height=1,
                input_width=T,
                kernel_size=(1, self.K),
                stride=(1, self.stride),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=self.device,
                input_dtype=ttnn.bfloat16,
                conv_config=_cfg,
                compute_config=_HIFI4,
            )
        self._prepared_for_T = T

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [B, 1, T, in_ch] → [B, 1, T*stride, out_ch]"""
        B, _, T, _ = x.shape
        T_full = (T - 1) * self.stride + self.K
        T_out = T_full - self.trim_right  # = T * stride

        if self._prepared_for_T != T:
            self._prepare(B, T)

        x_out, [self.weight, self.bias] = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            batch_size=B,
            input_height=1,
            input_width=T,
            kernel_size=(1, self.K),
            stride=(1, self.stride),
            padding=(0, 0),
            return_output_dim=False,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
            compute_config=_HIFI4,
        )
        # x_out: [1, 1, B*T_full, out_ch] — reshape and trim right
        x_out = ttnn.reshape(x_out, [B, 1, T_full, self.out_ch])
        if self.trim_right > 0:
            x_out = ttnn.slice(
                x_out,
                [0, 0, 0, 0],
                [B, 1, T_out, self.out_ch],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return x_out


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

    def encode(self, audio: ttnn.Tensor) -> ttnn.Tensor:
        """audio: [B, 1, 1, T] → [B, 1, T_enc, vae_dim]"""
        return self._encoder_tt.forward(audio)

    def decode(self, latents: ttnn.Tensor) -> ttnn.Tensor:
        """latents: [B, 1, T_enc, vae_dim] → [B, 1, 1, T_audio] (all ops on device)"""
        B = latents.shape[0]

        x = latents if latents.dtype == ttnn.bfloat16 else ttnn.typecast(latents, ttnn.bfloat16)

        x = self._dec_input_conv(x)
        for blk in self._dec_stage_blocks[0]:
            x = blk(x)

        for up_conv, blocks in zip(self._dec_upsample_convs, self._dec_stage_blocks[1:]):
            x = up_conv(x)
            for blk in blocks:
                x = blk(x)

        x = self._dec_head_conv(x)  # [B, 1, T_audio, 1]
        T_audio = x.shape[2]
        return ttnn.reshape(x, [B, 1, 1, T_audio])

    def __call__(self, audio: ttnn.Tensor) -> ttnn.Tensor:
        return self.encode(audio)
