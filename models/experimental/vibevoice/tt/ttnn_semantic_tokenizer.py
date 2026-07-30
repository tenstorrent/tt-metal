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
import os
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


# ── FFN down-proj (linear2) decode program configs (VV_POST_L2_PROGCFG=1) ─────
# The deepest tokenizer stages (dim 2048 / 1024) run one latent frame => T<=32 rows
# (M_tiles=1) and their down-proj (linear2, K=4*dim) is the biggest post-phase matmul
# on the auto config.  These swept 1D mcast_in0 configs (per_core_M=1) recovered
# ~1.8x/1.5x when originally measured.  Keyed by dim, gated on T<=32; the up-proj is
# already DRAM-BW-bound at auto so it stays auto.
def _mm1d_post(cx, cy, in0_block_w, per_core_n):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


_FFN_DOWN_PROGCFG = {
    2048: _mm1d_post(8, 4, 8, 2),  # 8192x2048  145 -> 81 us (1.8x)
    1024: _mm1d_post(8, 2, 8, 2),  # 4096x1024   60 -> 40 us (1.5x)
}


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


def _tile_linear(t: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """[out, in] → [1, 1, in, out] TILE for ttnn.linear (x @ w semantics)."""
    return ttnn.as_tensor(
        t.t().unsqueeze(0).unsqueeze(0).contiguous(),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _norm_w_tt(w: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """[C] norm weight → [1, 1, C//32, 32] ROW_MAJOR for ttnn.rms_norm."""
    C = w.shape[0]
    tdtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.as_tensor(
        w.to(tdtype).view(1, 1, C // 32, 32).contiguous(),
        device=device,
        dtype=dtype,
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

    def __init__(self, cw: ConvWeightsHost, device, compute_dtype=ttnn.bfloat16):
        self.device = device
        self.compute_dtype = compute_dtype
        self.stride = cw.stride
        self.groups = cw.groups
        self.causal_pad = cw.causal_pad

        out_ch, in_per_group, K = cw.weight.shape
        self.out_ch = out_ch
        self.in_ch = in_per_group * cw.groups
        self.K = K

        tdtype = torch.bfloat16 if compute_dtype == ttnn.bfloat16 else torch.float32
        # OIHW: [out_ch, in_ch//groups, H=1, K_W=K] for ttnn.conv2d.
        # Kept on HOST (unprepared conv-weight layout): the first ttnn.conv2d call takes the
        # clean host-preprocess path (a log_trace) and returns the device-prepared weight/bias
        # via return_weights_and_bias, which we cache below. Putting an unprepared tensor on
        # device instead makes conv2d log a "weights not properly prepared" warning and pull it
        # back to host anyway. Warm-up runs every conv once before trace capture, so the prep
        # happens at load time, outside the trace region.
        w4d = cw.weight.to(tdtype).unsqueeze(2).contiguous()
        self.weight = ttnn.as_tensor(
            w4d,
            dtype=compute_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        if cw.bias is not None:
            # conv2d requires bias as [1, 1, 1, out_ch]
            self.bias = ttnn.as_tensor(
                cw.bias.to(tdtype).view(1, 1, 1, -1).contiguous(),
                dtype=compute_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        else:
            self.bias = None

        # Streaming context cache: last `causal_pad` input columns [B, 1, causal_pad, in_ch].
        # context_size == causal_pad == (K-1)*dilation - (stride-1) (reference SConv1d).
        self._cache = None
        self._cache_zeros_host = None  # cached host zeros for in-place reset (llama-pattern trace)

        # Prepared-weight cache keyed by input geometry. ttnn.conv2d prepares the weight (and
        # sometimes shards weight_matrix_height across a core grid chosen from the input geometry)
        # on first use and hands it back via return_weights_and_bias; reusing a weight prepared for
        # one geometry with a different one trips the width-sharded
        # `act_matrix_width == weight_matrix_height` assertion.  A single conv instance can legitimately
        # see multiple geometries (e.g. a single-shot encode of a short prompt vs. a streaming encode
        # of a long one), so cache the prepared weight/bias per geometry signature instead of a single
        # rebind.  Steady-state (trace) callers hit one key repeatedly → same tensor address, so trace
        # capture/replay stays stable.
        self._prepared: Dict[tuple, tuple] = {}

    def reset_cache(self) -> None:
        self._cache = None

    def reset_cache_inplace(self) -> None:
        """Zero the streaming cache IN PLACE (keep the buffer address).  Used by the llama-pattern
        fused-frame trace: the caches are reset at a segment start while the trace is live, and
        reallocating (reset_cache -> None -> realloc) would corrupt the trace's captured addresses.
        No-op if the cache hasn't been allocated yet (then the next call allocates it fresh)."""
        if self._cache is None:
            return
        if self._cache_zeros_host is None:
            tdtype = torch.bfloat16 if self.compute_dtype == ttnn.bfloat16 else torch.float32
            self._cache_zeros_host = ttnn.from_torch(
                torch.zeros(list(self._cache.shape), dtype=tdtype),
                dtype=self.compute_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        ttnn.copy_host_to_device_tensor(self._cache_zeros_host, self._cache)

    def _extra_right_pad(self, T: int) -> int:
        """get_extra_padding_for_conv1d (only meaningful for stride > 1)."""
        if self.stride <= 1:
            return 0
        n_frames = (T - self.K + self.causal_pad) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.K - self.causal_pad)
        return max(0, ideal_length - T)

    def __call__(self, x: ttnn.Tensor, use_cache: bool = False, is_final_chunk: bool = False) -> ttnn.Tensor:
        """x: [B, 1, T, in_ch] NHWC → [B, 1, T_out, out_ch].

        Streaming (use_cache=True): the left causal pad is replaced by the cached
        tail of previous inputs; no extra right-pad except on the final chunk.
        Mirrors SConv1d._forward_streaming.
        """
        B, _, T, _ = x.shape
        cp = self.causal_pad

        if use_cache:
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            if cp > 0:
                if self._cache is None:
                    # Fixed, pre-allocated streaming cache: allocated once and updated in
                    # place below, so its buffer address stays constant across calls.  It
                    # must not be reassigned to a fresh ttnn.slice per call — ttnn trace
                    # capture/replay requires stable buffer addresses.
                    self._cache = ttnn.zeros(
                        [B, 1, cp, self.in_ch],
                        dtype=self.compute_dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                cache = self._cache
                x_ctx = ttnn.concat([cache, x], dim=2)  # [B, 1, cp + T, in_ch]
                # Update the cache IN PLACE with the last `cp` input columns (pre
                # right-pad): write into the fixed buffer rather than rebinding it.
                new_tail = ttnn.slice(
                    x_ctx, [0, 0, T, 0], [B, 1, cp + T, self.in_ch], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.copy(input_a=new_tail, input_b=self._cache)
            else:
                x_ctx = x
            extra_pad = self._extra_right_pad(T) if is_final_chunk else 0
            if extra_pad > 0:
                x_ctx = ttnn.pad(x_ctx, [(0, 0), (0, 0), (0, extra_pad), (0, 0)], value=0.0)
            x = x_ctx
            T_padded = cp + T + extra_pad
            # Cache/concat already materialised the pad; conv sees the full width.
            input_width = T_padded
            conv_padding: tuple = (0, 0)
        else:
            # Non-streaming: keep TILE and fold causal/extra pad into conv2d's native
            # [pad_top, pad_bottom, pad_left, pad_right] — kills Untilize→Pad and cheapens I2S.
            extra_pad = self._extra_right_pad(T)
            T_padded = T + cp + extra_pad
            if x.layout != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            input_width = T
            conv_padding = (0, 0, cp, extra_pad)

        # VV_CONV_SINGLE_BLOCK=1 (default): force the depthwise conv (groups>1) onto the
        # single-height-block path by setting act_block_h >= the full output height.  The
        # multi-block path in compute_depthwise_conv1d.cpp uses a separate-scratch scheme whose
        # numerics differ, and this conv is the only value-changing op in the streaming feedback
        # loop, so a small delta here compounds over a long-form render.  Pinning the single-block
        # path keeps the depthwise numerics stable.
        _conv_cfg = None
        if os.environ.get("VV_CONV_SINGLE_BLOCK", "1") == "1" and self.groups > 1 and use_cache:
            out_w = (T_padded - self.K) // self.stride + 1
            abh = ((out_w + 31) // 32) * 32
            # A full-height single block > ~1600 rows overflows L1 (the g=32/outw=3200 last decoder
            # stage needs ~2.9 MB > 1.5 MB), so cap it — that conv stays on auto (multi-block).
            _sb_cap = int(os.environ.get("VV_CONV_SB_CAP", "1600"))
            # HEIGHT_SHARDED pin (default on; VV_CONV_HS=0 to force the act_block_h_override pin):
            # act_block_h stays at the full per-core height (single block) while the work spreads
            # over many cores — no oversized act_block_h_override and none of its "not a valid
            # override" spam.  It only fits the wide-output stages: high-channel/tiny-width stages
            # (out_w < VV_CONV_HS_MIN_OUTW) overflow L1 under HEIGHT_SHARDED, so they stay on auto
            # (see the sub-threshold note below).  VV_CONV_HS_MAX_OUTW>0 caps the HS set, so HS coverage can be
            # made identical to the override pin's — isolating the layout mechanism from the pinned-set
            # change.  Default flipped on after the 4p_climate_100min traced render (0/81 anomalous
            # minutes, 0% clipping) validated it.
            _hs_max = int(os.environ.get("VV_CONV_HS_MAX_OUTW", "0"))
            _hs_min = int(os.environ.get("VV_CONV_HS_MIN_OUTW", "128"))
            _hs = os.environ.get("VV_CONV_HS", "1") == "1" and out_w >= _hs_min and (_hs_max == 0 or out_w <= _hs_max)
            # Below the HS threshold the override pin is inert, so it is never passed: those outputs
            # are a few tiles at most, conv2d spreads them one tile per core, and act_block_h
            # collapses to that per-core height — a single act block either way, the same config auto
            # picks.  An override wider than the per-core height only costs a "not a valid override"
            # info line per shard-layout candidate per call (24/frame from the two outw=40 depthwise
            # stages: acoustic-decoder stage 2 and semantic-encoder stage 4).  Verified bit-exact
            # against the pinned path (streaming decode+encode SHA-256 equal over 8 frames, traced
            # post-diffusion PCC 1.0, byte-identical demo wav), so there is no A/B switch.
            if _hs:
                _conv_cfg = ttnn.Conv2dConfig(shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
            elif _hs_min <= out_w <= _sb_cap:
                _conv_cfg = ttnn.Conv2dConfig(act_block_h_override=abh)
            if os.environ.get("VV_CONV_DBG") == "1":
                print(
                    f"[conv sb] g={self.groups} in={self.in_ch} out={self.out_ch} "
                    f"Tpad={T_padded} outw={out_w} abh={abh} pinned={_conv_cfg is not None}",
                    flush=True,
                )
        # Reuse the weight prepared for this exact geometry if we've seen it; otherwise prepare from
        # the host original.  The prepared layout depends on T_padded and on whether the single-block
        # override is applied (act_block_h), so both go in the key.
        geo_key = (T_padded, _conv_cfg is not None, use_cache)
        w_in, b_in = self._prepared.get(geo_key, (self.weight, self.bias))
        x_out, [_, w_out], [w_prep, b_prep] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=w_in,
            bias_tensor=b_in,
            device=self.device,
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            batch_size=B,
            input_height=1,
            input_width=input_width,
            kernel_size=(1, self.K),
            stride=(1, self.stride),
            padding=conv_padding,
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.compute_dtype,
            compute_config=_HIFI4,
            conv_config=_conv_cfg,
        )
        self._prepared[geo_key] = (w_prep, b_prep)
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

    def __init__(self, bw: Block1DWeightsHost, device, compute_dtype=ttnn.bfloat16):
        self.device = device
        self.eps = bw.eps
        self.dim = bw.dim

        tdtype = torch.bfloat16 if compute_dtype == ttnn.bfloat16 else torch.float32

        # VV_POST_SCALE_FOLD=1: fold the per-channel layer scales into the weights that
        # produce the scaled tensors — gamma into the depthwise conv weight/bias, ffn_gamma
        # into linear2's — removing two eltwise muls per block.  Both are exact
        # output-channel scales, but folding pre-scales the weights (rounded to bf16 once)
        # instead of scaling the bf16 product, so it is NOT bit-identical.
        _fold = os.environ.get("VV_POST_SCALE_FOLD") == "1"
        dw = bw.dw_conv
        l2_w_host, l2_b_host = bw.linear2_w, bw.linear2_b
        self._fold_gamma = _fold and bw.gamma is not None
        self._fold_ffn_gamma = _fold and bw.ffn_gamma is not None
        if self._fold_gamma:
            g = bw.gamma.float()
            dw = ConvWeightsHost(
                weight=dw.weight.float() * g.view(-1, 1, 1),
                bias=(dw.bias.float() * g) if dw.bias is not None else None,
                stride=dw.stride,
                groups=dw.groups,
                causal_pad=dw.causal_pad,
            )
        if self._fold_ffn_gamma:
            fg = bw.ffn_gamma.float()
            l2_w_host = bw.linear2_w.float() * fg.view(-1, 1)
            l2_b_host = (bw.linear2_b.float() * fg) if bw.linear2_b is not None else None

        self.dw_conv = TTConv1d(dw, device, compute_dtype=compute_dtype)
        self.norm_w = _norm_w_tt(bw.norm_w, device, dtype=compute_dtype)
        self.ffn_norm_w = _norm_w_tt(bw.ffn_norm_w, device, dtype=compute_dtype)
        # linear1_w is [ffn_dim, C] in PyTorch → _tile_linear transposes to [C, ffn_dim]
        self.linear1_w = _tile_linear(bw.linear1_w, device, dtype=compute_dtype)
        self.linear2_w = _tile_linear(l2_w_host.contiguous(), device, dtype=compute_dtype)
        # Tuned down-proj config for the deep (dim 2048/1024) stages; auto otherwise.
        self._l2_progcfg = _FFN_DOWN_PROGCFG.get(self.dim) if os.environ.get("VV_POST_L2_PROGCFG", "1") == "1" else None

        def _bias(b: Optional[torch.Tensor]) -> Optional[ttnn.Tensor]:
            if b is None:
                return None
            return ttnn.as_tensor(
                b.to(tdtype).view(1, 1, 1, -1).contiguous(),
                device=device,
                dtype=compute_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def _scale(s: Optional[torch.Tensor]) -> Optional[ttnn.Tensor]:
            if s is None:
                return None
            C = s.shape[0]
            return ttnn.as_tensor(
                s.to(tdtype).view(1, 1, 1, C).contiguous(),
                device=device,
                dtype=compute_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.linear1_b = _bias(bw.linear1_b)
        self.linear2_b = _bias(l2_b_host)
        self.gamma = None if self._fold_gamma else _scale(bw.gamma)
        self.ffn_gamma = None if self._fold_ffn_gamma else _scale(bw.ffn_gamma)

    def reset_cache(self) -> None:
        self.dw_conv.reset_cache()

    def reset_cache_inplace(self) -> None:
        self.dw_conv.reset_cache_inplace()

    def __call__(self, x: ttnn.Tensor, use_cache: bool = False, is_final_chunk: bool = False) -> ttnn.Tensor:
        """x: [B, 1, T, C] → [B, 1, T, C]"""
        # Mixer (depthwise conv) path
        residual = x
        x = ttnn.rms_norm(
            x, weight=self.norm_w, epsilon=self.eps, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x = self.dw_conv(x, use_cache=use_cache, is_final_chunk=is_final_chunk)
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
        # per_core_M=1 config is valid only when T<=32 rows (M_tiles=1); else auto.
        l2_pc = self._l2_progcfg if (self._l2_progcfg is not None and x.shape[2] <= 32) else None
        x = ttnn.linear(
            x,
            self.linear2_w,
            bias=self.linear2_b,
            compute_kernel_config=_HIFI4,
            program_config=l2_pc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    Input:  [B, 1, T, 1] NHWC (preferred) or [B, 1, 1, T] (reshaped once on device)
    Output: [B, 1, T_enc, vae_dim]
    """

    def __init__(self, weights: SemanticTokenizerWeights, device):
        self.device = device
        self.eps = weights.eps

        self._downsample_convs = [TTConv1d(cw, device) for cw in weights.downsample_convs]
        self._stages = [[TTBlock1DDevice(bw, device) for bw in stage_blocks] for stage_blocks in weights.stages]

        if weights.final_norm_w is not None:
            self._final_norm_w = _norm_w_tt(weights.final_norm_w, device)
        else:
            self._final_norm_w = None

        self._head_conv = TTConv1d(weights.head_conv, device)

    def reset_cache(self) -> None:
        """Clear all streaming caches (call before encoding a new segment)."""
        for c in self._downsample_convs:
            c.reset_cache()
        for stage in self._stages:
            for blk in stage:
                blk.reset_cache()
        self._head_conv.reset_cache()

    def reset_cache_inplace(self) -> None:
        """Zero all streaming caches IN PLACE (llama-pattern trace; stable addresses)."""
        for c in self._downsample_convs:
            c.reset_cache_inplace()
        for stage in self._stages:
            for blk in stage:
                blk.reset_cache_inplace()
        self._head_conv.reset_cache_inplace()

    def forward(
        self,
        audio: ttnn.Tensor,
        golden: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> ttnn.Tensor:
        """Encode audio to semantic latents (all ops on device).

        Args:
            audio:  [B, 1, T, 1] NHWC preferred (avoids a device ReshapeView), or
                    [B, 1, 1, T] which is reshaped once. TILE layout preferred.
            golden: optional [B, vae_dim, T_enc] torch reference tensor.
                    If provided, PCC between TTNN output and golden is printed.
            use_cache: stream this chunk using the per-conv causal caches.
            is_final_chunk: add ceil-alignment right-pad on the final chunk only.
        """
        B = audio.shape[0]

        # Prefer host/upload as [B, 1, T, 1]; only reshape the legacy [B, 1, 1, T] layout.
        if audio.shape[2] == 1 and audio.shape[3] != 1:
            T = int(audio.shape[3])
            x = ttnn.reshape(audio, [B, 1, T, 1])
        else:
            x = audio
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        for i, stage_blocks in enumerate(self._stages):
            x = self._downsample_convs[i](x, use_cache=use_cache, is_final_chunk=is_final_chunk)
            for blk in stage_blocks:
                x = blk(x, use_cache=use_cache, is_final_chunk=is_final_chunk)

        if self._final_norm_w is not None:
            x = ttnn.rms_norm(
                x,
                weight=self._final_norm_w,
                epsilon=self.eps,
                compute_kernel_config=_HIFI4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        x = self._head_conv(x, use_cache=use_cache, is_final_chunk=is_final_chunk)  # [B, 1, T_enc, vae_dim]

        if golden is not None:
            # [B, 1, T_enc, vae_dim] NHWC → [B, vae_dim, T_enc] channels-first
            out_torch = ttnn.to_torch(x).squeeze(1).permute(0, 2, 1)
            passed, pcc_val = comp_pcc(golden, out_torch)
            print(f"[TTSemanticTokenizer] PCC = {pcc_val:.6f} ({'PASS' if passed else 'FAIL'})")

        return x

    def __call__(
        self,
        audio: ttnn.Tensor,
        golden: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_final_chunk: bool = False,
    ) -> ttnn.Tensor:
        return self.forward(audio, golden, use_cache=use_cache, is_final_chunk=is_final_chunk)
