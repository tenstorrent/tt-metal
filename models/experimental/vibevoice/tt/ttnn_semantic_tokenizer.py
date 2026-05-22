# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice Semantic Tokenizer (encoder-only) — TTNN port.

Reference: TokenizerEncoder in modular_vibevoice_tokenizer.py

Architecture (channels-first [B, C, T] throughout):
  downsample_layers[0]: SConv1d(1 → n_filters, K=7, stride=1)  — input conv
  For i in 0..n_stages-1:
    downsample_layers[i]  applied first
    stages[i]: depth[i] Block1D blocks
  Final norm (ConvRMSNorm)
  head: SConv1d(in_ch → vae_dim=128, K=7, stride=1)

Block1D (channels-first):
  residual = x
  x = ConvRMSNorm(x)          # norm.weight [C]
  x = depthwise_conv(x)       # mixer.conv.conv.conv.weight [C, 1, K]
  x = x * gamma               # gamma [C, 1] layer scale
  x = residual + x
  residual = x
  x = ConvRMSNorm(x)          # ffn_norm.weight [C]
  x = x.permute(0,2,1)        # [B,C,T] → [B,T,C]
  x = linear1(x) → gelu → linear2(x)
  x = x.permute(0,2,1)        # [B,T,C] → [B,C,T]
  x = x * ffn_gamma
  x = residual + x

Note: ttnn.conv1d/conv2d has a Blackhole kernel bug (compile-time args OOB).
All convolutions are run host-side via torch.nn.functional.conv1d.
Linear and RMSNorm ops run on device via TTNN where applicable.
The TTNN tensor interface (input/output) is preserved.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.vibevoice.tt.vibevoice_config import SemanticTokenizerConfig


_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ──────────────────────────────────────────────────────────────
# Host-side weight containers (torch tensors, not TTNN)
# ──────────────────────────────────────────────────────────────


@dataclass
class ConvWeightsHost:
    weight: torch.Tensor  # [out_ch, in_ch//groups, K]
    bias: Optional[torch.Tensor]
    stride: int
    groups: int
    causal_pad: int  # left-pad applied before conv


@dataclass
class Block1DWeightsHost:
    dw_conv: ConvWeightsHost  # depthwise conv weight
    norm_w: torch.Tensor  # [C] ConvRMSNorm weight (pre-mixer)
    ffn_norm_w: torch.Tensor  # [C] ConvRMSNorm weight (pre-FFN)
    linear1_w: torch.Tensor  # [ffn_dim, C]
    linear1_b: Optional[torch.Tensor]
    linear2_w: torch.Tensor  # [C, ffn_dim]
    linear2_b: Optional[torch.Tensor]
    gamma: Optional[torch.Tensor]  # [C] mixer layer scale
    ffn_gamma: Optional[torch.Tensor]  # [C] FFN layer scale
    dim: int
    ffn_dim: int
    eps: float


@dataclass
class SemanticTokenizerWeights:
    """All weights for TTSemanticTokenizer — stored as host torch tensors."""

    # downsample_layers[0] = input conv (1 → n_filters)
    # downsample_layers[1..n_stages] = stride-S downsamplers
    downsample_convs: List[ConvWeightsHost]
    # stages[i] = list of Block1DWeightsHost
    stages: List[List[Block1DWeightsHost]]
    # final norm weight [C] and head conv
    final_norm_w: Optional[torch.Tensor]
    head_conv: ConvWeightsHost
    eps: float
    config: SemanticTokenizerConfig


# ──────────────────────────────────────────────────────────────
# Weight preprocessing (torch allowed, host only)
# ──────────────────────────────────────────────────────────────


def _parse_depths(depths_str: str) -> List[int]:
    return [int(d) for d in depths_str.split("-")]


def _get_conv_weights(
    sd: dict, prefix: str, in_ch: int, out_ch: int, kernel_size: int, stride: int, groups: int = 1, causal: bool = False
) -> ConvWeightsHost:
    w = sd.get(f"{prefix}.weight", torch.zeros(out_ch, in_ch // groups, kernel_size, dtype=torch.float32))
    b = sd.get(f"{prefix}.bias", None)
    # Reference: padding_total = (kernel_size - 1) * dilation - (stride - 1)
    # All convolutions here have dilation=1
    causal_pad = (kernel_size - 1 - (stride - 1)) if causal else 0
    return ConvWeightsHost(
        weight=w.float().contiguous(),
        bias=b.float().contiguous() if b is not None else None,
        stride=stride,
        groups=groups,
        causal_pad=causal_pad,
    )


def preprocess_semantic_tokenizer_weights(
    hf_state: Dict[str, torch.Tensor],
    device,
    config: SemanticTokenizerConfig,
) -> "SemanticTokenizerWeights":
    """Build SemanticTokenizerWeights from the hf_state dict.

    Expected key structure (prefix-stripped by split_submodule_weights):
      encoder.downsample_layers.N.0.conv.conv.{weight,bias}
      encoder.stages.N.B.mixer.conv.conv.conv.{weight,bias}
      encoder.stages.N.B.norm.weight
      encoder.stages.N.B.ffn_norm.weight
      encoder.stages.N.B.ffn.linear1.{weight,bias}
      encoder.stages.N.B.ffn.linear2.{weight,bias}
      encoder.stages.N.B.gamma
      encoder.stages.N.B.ffn_gamma
      encoder.head.conv.conv.{weight,bias}
    """
    depths = _parse_depths(config.encoder_depths)
    ratios = list(reversed(config.encoder_ratios))  # reference reverses ratios
    n_filters = config.encoder_n_filters
    causal = config.causal
    eps = config.layernorm_eps

    # ── downsample_layers ───────────────────────────────────────────────
    # layer 0: input conv (channels=1 → n_filters, K=7, stride=1)
    downsample_convs: List[ConvWeightsHost] = []
    dl0_prefix = "encoder.downsample_layers.0.0.conv.conv"
    dl0 = _get_conv_weights(
        hf_state, dl0_prefix, in_ch=1, out_ch=n_filters, kernel_size=7, stride=1, groups=1, causal=causal
    )
    downsample_convs.append(dl0)

    for i, ratio in enumerate(ratios):
        in_ch = n_filters * (2**i)
        out_ch = n_filters * (2 ** (i + 1))
        kernel_size = ratio * 2
        prefix = f"encoder.downsample_layers.{i + 1}.0.conv.conv"
        cw = _get_conv_weights(
            hf_state, prefix, in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=ratio, groups=1, causal=causal
        )
        downsample_convs.append(cw)

    # ── stages ──────────────────────────────────────────────────────────
    stages: List[List[Block1DWeightsHost]] = []
    for stage_idx, depth in enumerate(depths):
        dim = n_filters * (2**stage_idx)
        ffn_dim_default = 4 * dim
        blocks: List[Block1DWeightsHost] = []
        for b_idx in range(depth):
            bp = f"encoder.stages.{stage_idx}.{b_idx}"

            # Depthwise conv (groups=dim, kernel=7)
            dw_prefix = f"{bp}.mixer.conv.conv.conv"
            dw = _get_conv_weights(
                hf_state, dw_prefix, in_ch=dim, out_ch=dim, kernel_size=7, stride=1, groups=dim, causal=causal
            )

            norm_w = hf_state.get(f"{bp}.norm.weight", torch.ones(dim)).float()
            ffn_norm_w = hf_state.get(f"{bp}.ffn_norm.weight", torch.ones(dim)).float()

            l1_w = hf_state.get(f"{bp}.ffn.linear1.weight", torch.zeros(ffn_dim_default, dim)).float()
            l1_b = hf_state.get(f"{bp}.ffn.linear1.bias", None)
            l2_w = hf_state.get(f"{bp}.ffn.linear2.weight", torch.zeros(dim, ffn_dim_default)).float()
            l2_b = hf_state.get(f"{bp}.ffn.linear2.bias", None)

            gamma = hf_state.get(f"{bp}.gamma", None)
            ffn_gamma = hf_state.get(f"{bp}.ffn_gamma", None)

            ffn_dim = l1_w.shape[0]
            blk = Block1DWeightsHost(
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
            blocks.append(blk)
        stages.append(blocks)

    # ── final norm ───────────────────────────────────────────────────────
    # The final norm key is encoder.norm.weight in some models
    last_dim = n_filters * (2 ** (len(depths) - 1))
    final_norm_w = hf_state.get("encoder.norm.weight", hf_state.get("norm.weight", None))
    if final_norm_w is not None:
        final_norm_w = final_norm_w.float().contiguous()

    # ── head conv ────────────────────────────────────────────────────────
    head_prefix = "encoder.head.conv.conv"
    head_conv = _get_conv_weights(
        hf_state, head_prefix, in_ch=last_dim, out_ch=config.vae_dim, kernel_size=7, stride=1, groups=1, causal=causal
    )

    return SemanticTokenizerWeights(
        downsample_convs=downsample_convs,
        stages=stages,
        final_norm_w=final_norm_w,
        head_conv=head_conv,
        eps=eps,
        config=config,
    )


# ──────────────────────────────────────────────────────────────
# Host-side forward helpers
# ──────────────────────────────────────────────────────────────


def _apply_conv1d(x: torch.Tensor, cw: ConvWeightsHost) -> torch.Tensor:
    """Apply Conv1d on host. x: [B, C, T] → [B, C_out, T_out].

    Matches SConv1d._forward_non_streaming: left-pads by padding_total and
    right-pads by extra_padding so output length equals ceil(T / stride).
    """
    T = x.shape[-1]
    kernel_size = cw.weight.shape[-1]
    padding_total = cw.causal_pad

    # Compute right-side extra padding (matches get_extra_padding_for_conv1d)
    if cw.stride > 1:
        n_frames = (T - kernel_size + padding_total) / cw.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * cw.stride + (kernel_size - padding_total)
        extra_pad = max(0, ideal_length - T)
    else:
        extra_pad = 0

    if padding_total > 0 or extra_pad > 0:
        x = F.pad(x, (padding_total, extra_pad))
    return F.conv1d(x, cw.weight, cw.bias, stride=cw.stride, groups=cw.groups)


def _conv_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """ConvRMSNorm on [B, C, T]: normalize last dim of transposed input."""
    x = x.transpose(1, 2)  # [B, C, T] → [B, T, C]
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    x = (x.float() / rms).to(x.dtype) * weight.unsqueeze(0).unsqueeze(0)
    return x.transpose(1, 2)  # [B, T, C] → [B, C, T]


def _block1d_forward(x: torch.Tensor, bw: Block1DWeightsHost) -> torch.Tensor:
    """Block1D forward on [B, C, T]."""
    # Mixer (depthwise conv) path
    residual = x
    x = _conv_rms_norm(x, bw.norm_w, bw.eps)
    x = _apply_conv1d(x, bw.dw_conv)
    if bw.gamma is not None:
        x = x * bw.gamma.unsqueeze(-1)
    x = residual + x

    # FFN path
    residual = x
    x = _conv_rms_norm(x, bw.ffn_norm_w, bw.eps)
    x = x.permute(0, 2, 1)  # [B, C, T] → [B, T, C]
    x = F.linear(x, bw.linear1_w, bw.linear1_b)
    x = F.gelu(x)
    x = F.linear(x, bw.linear2_w, bw.linear2_b)
    x = x.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
    if bw.ffn_gamma is not None:
        x = x * bw.ffn_gamma.unsqueeze(-1)
    x = residual + x
    return x


# ──────────────────────────────────────────────────────────────
# TTSemanticTokenizer
# ──────────────────────────────────────────────────────────────


class TTSemanticTokenizer:
    """TTNN port of VibeVoiceSemanticTokenizerModel encoder.

    Input:  TTNN tensor [B, 1, 1, T] or [B, 1, T, 1] (raw audio)
    Output: TTNN tensor [B, 1, T_enc, vae_dim]

    Note: Convolutions run host-side (Blackhole TTNN conv kernel bug).
    """

    def __init__(self, weights: SemanticTokenizerWeights, device):
        self.w = weights
        self.device = device
        self.eps = weights.eps

    def forward(self, audio: ttnn.Tensor) -> ttnn.Tensor:
        """Encode audio to semantic latents.

        audio: [B, 1, 1, T] TTNN tensor (raw waveform)
        Returns: [B, 1, T_enc, vae_dim]
        """
        # Bring to host and reshape to [B, 1, T] (BCT format for conv1d)
        x_host = ttnn.to_torch(audio).to(torch.float32)  # [B, 1, 1, T]
        # audio is [B, dim1=1, H=1, T] → squeeze to [B, 1, T]
        B = x_host.shape[0]
        T = x_host.shape[-1]
        x = x_host.view(B, 1, T)  # [B, C=1, T]

        w = self.w

        # Process: downsample_layers[i] → stages[i] blocks
        for i, blocks in enumerate(w.stages):
            x = _apply_conv1d(x, w.downsample_convs[i])
            for blk in blocks:
                x = _block1d_forward(x, blk)

        # Final norm
        if w.final_norm_w is not None:
            x = _conv_rms_norm(x, w.final_norm_w, self.eps)

        # Head conv
        x = _apply_conv1d(x, w.head_conv)

        # x: [B, vae_dim, T_enc] → [B, 1, T_enc, vae_dim] for TTNN
        x = x.to(torch.bfloat16).permute(0, 2, 1).unsqueeze(1)  # [B, 1, T_enc, vae_dim]

        return ttnn.as_tensor(
            x,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, audio: ttnn.Tensor) -> ttnn.Tensor:
        return self.forward(audio)
