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

All convolutions, norms, and linear ops run on device via TTConv1d / TTBlock1DDevice.
Requires device opened with l1_small_size=32768 for conv support on Blackhole.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
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
# Device-side TT helpers: weight converters
# ──────────────────────────────────────────────────────────────


def _tile_linear(t: torch.Tensor, device) -> ttnn.Tensor:
    """[out, in] → [1, 1, in, out] TILE for ttnn.linear (x @ w semantics)."""
    return ttnn.as_tensor(
        t.to(torch.bfloat16).t().unsqueeze(0).unsqueeze(0).contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _norm_w_tt(w: torch.Tensor, device) -> ttnn.Tensor:
    """[C] norm weight → [1, 1, C//32, 32] ROW_MAJOR for ttnn.rms_norm."""
    C = w.shape[0]
    return ttnn.as_tensor(
        w.to(torch.bfloat16).view(1, 1, C // 32, 32).contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ──────────────────────────────────────────────────────────────
# TTConv1d — SConv1d on device via ttnn.conv2d(H=1 NHWC)
# ──────────────────────────────────────────────────────────────


class TTConv1d:
    """1D convolution on device via ttnn.conv2d with H=1 NHWC layout.

    Replicates SConv1d._forward_non_streaming: left causal pad + extra right pad,
    then conv with stride, then output in [B, 1, T_out, out_ch] NHWC.
    """

    def __init__(self, cw: ConvWeightsHost, device):
        self.device = device
        self.stride = cw.stride
        self.groups = cw.groups
        self.causal_pad = cw.causal_pad

        out_ch, in_per_group, K = cw.weight.shape
        self.out_ch = out_ch
        self.in_ch = in_per_group * cw.groups
        self.K = K

        # OIHW: [out_ch, in_ch//groups, H=1, K_W=K] for ttnn.conv2d
        w4d = cw.weight.to(torch.bfloat16).unsqueeze(2).contiguous()
        self.weight = ttnn.as_tensor(
            w4d,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if cw.bias is not None:
            # conv2d requires bias as [1, 1, 1, out_ch]
            self.bias = ttnn.as_tensor(
                cw.bias.to(torch.bfloat16).view(1, 1, 1, -1).contiguous(),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.bias = None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [B, 1, T, in_ch] NHWC → [B, 1, T_out, out_ch]"""
        B, _, T, _ = x.shape

        # Compute extra right-pad (matches get_extra_padding_for_conv1d)
        if self.stride > 1:
            n_frames = (T - self.K + self.causal_pad) / self.stride + 1
            ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.K - self.causal_pad)
            extra_pad = max(0, ideal_length - T)
        else:
            extra_pad = 0

        T_padded = T + self.causal_pad + extra_pad
        if self.causal_pad > 0 or extra_pad > 0:
            # ttnn.pad front padding requires ROW_MAJOR layout
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.pad(x, [(0, 0), (0, 0), (self.causal_pad, extra_pad), (0, 0)], value=0.0)

        x_out, [_, w_out], [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            batch_size=B,
            input_height=1,
            input_width=T_padded,
            kernel_size=(1, self.K),
            stride=(1, self.stride),
            padding=(0, 0),
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
            compute_config=_HIFI4,
        )
        # Output from conv2d is [1, 1, B*w_out, out_ch]; reshape to [B, 1, T_out, out_ch]
        return ttnn.reshape(x_out, [B, 1, w_out, self.out_ch])


# ──────────────────────────────────────────────────────────────
# TTBlock1DDevice — Block1D fully on device in NHWC [B, 1, T, C]
# ──────────────────────────────────────────────────────────────


class TTBlock1DDevice:
    """Block1D with all ops on device.

    Input/output format: [B, 1, T, C] NHWC (TTNN native for conv2d).
    ConvRMSNorm = ttnn.rms_norm over last dim (C) — matches reference semantics.
    FFN permute-to-TC is implicit in NHWC (already channels-last).
    """

    def __init__(self, bw: Block1DWeightsHost, device):
        self.device = device
        self.eps = bw.eps
        self.dim = bw.dim

        self.dw_conv = TTConv1d(bw.dw_conv, device)
        self.norm_w = _norm_w_tt(bw.norm_w, device)
        self.ffn_norm_w = _norm_w_tt(bw.ffn_norm_w, device)
        # linear1_w is [ffn_dim, C] in PyTorch → _tile_linear transposes to [C, ffn_dim]
        self.linear1_w = _tile_linear(bw.linear1_w, device)
        self.linear2_w = _tile_linear(bw.linear2_w, device)

        def _bias(b: Optional[torch.Tensor]) -> Optional[ttnn.Tensor]:
            if b is None:
                return None
            return ttnn.as_tensor(
                b.to(torch.bfloat16).view(1, 1, 1, -1).contiguous(),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def _scale(s: Optional[torch.Tensor]) -> Optional[ttnn.Tensor]:
            if s is None:
                return None
            C = s.shape[0]
            return ttnn.as_tensor(
                s.to(torch.bfloat16).view(1, 1, 1, C).contiguous(),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.linear1_b = _bias(bw.linear1_b)
        self.linear2_b = _bias(bw.linear2_b)
        self.gamma = _scale(bw.gamma)
        self.ffn_gamma = _scale(bw.ffn_gamma)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [B, 1, T, C] → [B, 1, T, C]"""
        # Mixer (depthwise conv) path
        residual = x
        x = ttnn.rms_norm(
            x, weight=self.norm_w, epsilon=self.eps, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x = self.dw_conv(x)
        if self.gamma is not None:
            x = ttnn.mul(x, self.gamma, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.add(residual, x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # FFN path — linear ops on last dim (C), no explicit permute needed in NHWC
        residual = x
        x = ttnn.rms_norm(
            x,
            weight=self.ffn_norm_w,
            epsilon=self.eps,
            compute_kernel_config=_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.linear(
            x, self.linear1_w, bias=self.linear1_b, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.gelu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.linear(
            x, self.linear2_w, bias=self.linear2_b, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if self.ffn_gamma is not None:
            x = ttnn.mul(x, self.ffn_gamma, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.add(residual, x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x


# ──────────────────────────────────────────────────────────────
# Device-side TT functional helpers (TTNN equivalents of torch fallbacks)
# ──────────────────────────────────────────────────────────────


def comp_pcc(golden: torch.Tensor, calculated: torch.Tensor, pcc_threshold: float = 0.99) -> "tuple[bool, float]":
    """Pearson Correlation Coefficient between two tensors (flattened).

    Returns (passes_threshold, pcc_value).
    """
    g = golden.float().flatten()
    c = calculated.float().flatten()
    pcc_val = torch.corrcoef(torch.stack([g, c]))[0, 1].item()
    return pcc_val >= pcc_threshold, pcc_val


def _tt_conv_rms_norm(x: ttnn.Tensor, weight: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """ConvRMSNorm on [B, 1, T, C] NHWC: RMS-normalise over C, then scale.

    TTNN equivalent of _conv_rms_norm.  ttnn.rms_norm normalises the last dim
    so no transpose is needed — NHWC already has C last.
    weight must be in [1, 1, C//32, 32] ROW_MAJOR as produced by _norm_w_tt.
    """
    return ttnn.rms_norm(
        x,
        weight=weight,
        epsilon=eps,
        compute_kernel_config=_HIFI4,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tt_apply_conv1d(x: ttnn.Tensor, conv: "TTConv1d") -> ttnn.Tensor:
    """Causal-padded Conv1d on [B, 1, T, C] NHWC → [B, 1, T_out, out_ch].

    TTNN equivalent of _apply_conv1d.  Delegates to TTConv1d which computes
    the extra right-pad and dispatches to ttnn.conv2d.
    """
    return conv(x)


def _tt_block1d_forward(x: ttnn.Tensor, blk: "TTBlock1DDevice") -> ttnn.Tensor:
    """Block1D forward on [B, 1, T, C] NHWC → [B, 1, T, C].

    TTNN equivalent of _block1d_forward.  Runs mixer (depthwise conv +
    layer-scale + residual) and FFN (linear → gelu → linear + layer-scale +
    residual) entirely on device via TTBlock1DDevice.
    """
    return blk(x)


# ──────────────────────────────────────────────────────────────
# TTSemanticTokenizer
# ──────────────────────────────────────────────────────────────


class TTSemanticTokenizer:
    """TTNN port of VibeVoiceSemanticTokenizerModel encoder.

    All convolutions, norms, and linear ops run on device via TTConv1d / TTBlock1DDevice.
    Device must be opened with l1_small_size=32768 for conv support on Blackhole.

    Input:  [B, 1, 1, T] raw audio
    Output: [B, 1, T_enc, vae_dim]
    """

    def __init__(self, weights: SemanticTokenizerWeights, device):
        self.device = device
        self.eps = weights.eps

        self._downsample_convs = [TTConv1d(cw, device) for cw in weights.downsample_convs]
        self._stages = [[TTBlock1DDevice(bw, device) for bw in stage_blocks] for stage_blocks in weights.stages]

        if weights.final_norm_w is not None:
            C = weights.final_norm_w.shape[0]
            self._final_norm_w = _norm_w_tt(weights.final_norm_w, device)
        else:
            self._final_norm_w = None

        self._head_conv = TTConv1d(weights.head_conv, device)

    def forward(self, audio: ttnn.Tensor, golden: Optional[torch.Tensor] = None) -> ttnn.Tensor:
        """Encode audio to semantic latents (all ops on device).

        Args:
            audio:  [B, 1, 1, T] raw audio tensor on device.
            golden: optional [B, vae_dim, T_enc] torch reference tensor.
                    If provided, PCC between TTNN output and golden is printed.
        """
        B = audio.shape[0]
        T = audio.shape[-1]

        # [B, 1, 1, T] → [B, 1, T, 1] NHWC for TTConv1d
        x = ttnn.reshape(audio, [B, 1, T, 1])
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)

        for i, stage_blocks in enumerate(self._stages):
            x = self._downsample_convs[i](x)
            for blk in stage_blocks:
                x = blk(x)

        if self._final_norm_w is not None:
            x = ttnn.rms_norm(
                x,
                weight=self._final_norm_w,
                epsilon=self.eps,
                compute_kernel_config=_HIFI4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        x = self._head_conv(x)  # [B, 1, T_enc, vae_dim]

        if golden is not None:
            # [B, 1, T_enc, vae_dim] NHWC → [B, vae_dim, T_enc] channels-first
            out_torch = ttnn.to_torch(x).squeeze(1).permute(0, 2, 1)
            passed, pcc_val = comp_pcc(golden, out_torch)
            print(f"[TTSemanticTokenizer] PCC = {pcc_val:.6f} ({'PASS' if passed else 'FAIL'})")

        return x

    def __call__(self, audio: ttnn.Tensor, golden: Optional[torch.Tensor] = None) -> ttnn.Tensor:
        return self.forward(audio, golden)
