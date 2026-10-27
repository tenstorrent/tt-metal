# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 VAE decoder on one Blackhole p150 with TTNN ops.

`AutoencoderKLQwenImage21` is a 3D VAE, but a still image is a single frame and every temporal
mechanism degenerates (see `reference/torch_vae.py` for the derivation): the causal 3D convolutions
are plain 2D convolutions, the `upsample3d` `time_conv` is skipped on the first chunk, and
`QwenImage21DupUp3D` collapses to a channel/pixel gather. So this is a 2D decoder:

    latent [1, 64, 64, 64] -> post_quant_conv -> conv_in -> mid -> 5 up blocks -> head
                           -> [1, 4, 1024, 1024] clamped to [-1, 1]

Activations are channels-last bf16 in DRAM, carried as `[1, 1, H*W, C]` (the shape `ttnn.conv2d`
returns and `ttnn.rms_norm` wants) with `(h, w)` tracked alongside. Channel counts are rounded up to
a multiple of 32 by zero-padding the convolution weights, which keeps every tensor tile-aligned; the
RMS norms compensate for the padded width by folding `sqrt(c / c_padded)` into gamma (see
`TTRmsNorm`), so the 144-channel stage needs no unaligned-reduction workaround.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

import ttnn

from ..common.config import VAE, VAEConfig
from ..common.weights import LazyCheckpoint

# Torch-only pixel helper, re-exported so callers can do `from ..tt.vae import to_pil`.
from ..reference.torch_vae import to_pil  # noqa: F401

TILE = 32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def align32(c: int) -> int:
    return ((c + TILE - 1) // TILE) * TILE


@dataclass
class VAEPrecision:
    """Dtype / math-fidelity policy. bf16 activations and weights, fp32 accumulation everywhere."""

    act_dtype: ttnn.DataType = ttnn.bfloat16
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    conv_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    norm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    # head_dim 1152 makes the flash-attention circular buffers exceed L1 unless k_chunk_size
    # is tiny, so the explicit matmul + softmax path (4096x4096 bf16 scores = 32 MB) is the default.
    attn_impl: str = "matmul"  # "matmul" or "sdpa"
    act_block_h: int = 0  # 0 = let conv2d choose
    # (h, w, in_channels, out_channels) -> num DRAM width slices, for convs that do not fit in L1.
    slice_overrides: Dict[Tuple[int, int, int, int], int] = field(default_factory=dict)

    def conv_compute(self, dev):
        return ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=self.conv_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def norm_compute(self, dev):
        return ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=self.norm_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def mm_compute(self, dev):
        return ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=self.mm_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )


# ------------------------------------------------------------------------------------------- convs


class TTConv2d:
    """`ttnn.conv2d` wrapper: `[1, 1, H*W, C_in] -> ([1, 1, H'*W', C_out], H', W')`.

    Channels are zero-padded to a multiple of 32, so the surrounding activations are always
    tile-aligned; `pad_out_channels=False` keeps the output width as is, for the 4-channel head
    convolution.

    `ttnn.conv2d` returns its weights already laid out for the parallelization it chose, which
    depends on the input height and width, so the device copies are cached per `(h, w)` and the host
    copy is kept as the source for a geometry that has not been seen yet. Having the device weights
    resident after one warm-up call is what makes the layer trace-capturable. Retain the four most
    recently used geometries, enough for all condition images in one request, and release older
    device copies before preparing another geometry. The serving pipeline runs the VAE eagerly.
    """

    def __init__(
        self,
        dev,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        padding: int,
        prec: VAEPrecision,
        pad_out_channels: bool = True,
        output_layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ) -> None:
        out_c, in_c, kh, kw = weight.shape
        self.in_c = align32(in_c)
        self.out_c = align32(out_c) if pad_out_channels else out_c
        if (self.in_c, self.out_c) != (in_c, out_c):
            w = torch.zeros(self.out_c, self.in_c, kh, kw, dtype=torch.float32)
            w[:out_c, :in_c] = weight.float()
            b = torch.zeros(self.out_c, dtype=torch.float32)
            b[:out_c] = bias.float()
        else:
            w, b = weight.float(), bias.float()

        self.dev = dev
        self.kernel = (kh, kw)
        self.padding = (padding, padding)
        self.prec = prec
        self.host_w = ttnn.from_torch(w.to(torch.bfloat16), dtype=prec.weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.host_b = ttnn.from_torch(
            b.reshape(1, 1, 1, self.out_c).to(torch.bfloat16), dtype=prec.weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.prepared: OrderedDict[Tuple[int, int], Tuple[ttnn.Tensor, ttnn.Tensor]] = OrderedDict()
        self.conv_config = ttnn.Conv2dConfig(
            # Cached halo tables accumulate across aspect ratios; keep them out of the fixed small-L1 pool.
            config_tensors_in_dram=True,
            weights_dtype=prec.weight_dtype,
            act_block_h_override=prec.act_block_h,
            output_layout=output_layout,
        )
        self.compute_config = prec.conv_compute(dev)

    def __call__(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        x = ttnn.reshape(x, (1, h, w, self.in_c))
        n_slices = self.prec.slice_overrides.get((h, w, self.in_c, self.out_c))
        slice_config = (
            ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=n_slices) if n_slices else None
        )
        key = (h, w)
        if key not in self.prepared and len(self.prepared) == 4:
            _, old = self.prepared.popitem(last=False)
            for tensor in old:
                ttnn.deallocate(tensor)
        wt, bt = self.prepared.get(key, (self.host_w, self.host_b))
        out, (oh, ow), (wt, bt) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=wt,
            bias_tensor=bt,
            device=self.dev,
            in_channels=self.in_c,
            out_channels=self.out_c,
            batch_size=1,
            input_height=h,
            input_width=w,
            kernel_size=self.kernel,
            stride=(1, 1),
            padding=self.padding,
            dilation=(1, 1),
            groups=1,
            dtype=self.prec.act_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            slice_config=slice_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self.prepared[key] = (wt, bt)
        self.prepared.move_to_end(key)
        return ttnn.reshape(out, (1, 1, oh * ow, self.out_c)), oh, ow


# -------------------------------------------------------------------------------------------- norm


class TTRmsNorm:
    """`QwenImage21RMS_norm` over the channel dim of `[1, 1, H*W, C]`.

    The reference is `F.normalize(x, dim=C) * sqrt(C) * gamma`, which is `x / sqrt(mean_C(x^2))
    * gamma`, i.e. plain RMS norm with a negligible epsilon. When the activation carries `C_pad > C` channels (all zero, because the producing
    convolution has zero weights there) `ttnn.rms_norm` divides by `C_pad`, so gamma absorbs
    `sqrt(C / C_pad)` and is zero on the padded channels.
    """

    EPS = 1e-12

    def __init__(self, dev, gamma: torch.Tensor, prec: VAEPrecision) -> None:
        c = gamma.numel()
        c_pad = align32(c)
        g = torch.zeros(c_pad, dtype=torch.float32)
        g[:c] = gamma.reshape(-1).float() * math.sqrt(c / c_pad)
        self.c, self.c_pad = c, c_pad
        self.gamma = ttnn.from_torch(
            g.reshape(1, 1, 1, c_pad).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=DRAM,
        )
        self.compute_config = prec.norm_compute(dev)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, epsilon=self.EPS, weight=self.gamma, compute_kernel_config=self.compute_config)


# ------------------------------------------------------------------------------------------ blocks


class TTResBlock:
    """norm -> silu -> 3x3 conv -> norm -> silu -> 3x3 conv, plus a 1x1-conv or identity shortcut."""

    def __init__(self, dev, ckpt: LazyCheckpoint, prefix: str, prec: VAEPrecision) -> None:
        g = lambda k: ckpt.get(prefix + k, torch.float32)
        self.norm1 = TTRmsNorm(dev, g("norm1.gamma"), prec)
        self.conv1 = TTConv2d(dev, g("conv1.weight"), g("conv1.bias"), padding=1, prec=prec)
        self.norm2 = TTRmsNorm(dev, g("norm2.gamma"), prec)
        self.conv2 = TTConv2d(dev, g("conv2.weight"), g("conv2.bias"), padding=1, prec=prec)
        self.shortcut = (
            TTConv2d(dev, g("conv_shortcut.weight"), g("conv_shortcut.bias"), padding=0, prec=prec)
            if (prefix + "conv_shortcut.weight") in ckpt
            else None
        )

    def __call__(self, x: ttnn.Tensor, h: int, w: int, *, keep_input: bool = False) -> Tuple[ttnn.Tensor, int, int]:
        """`keep_input` leaves `x` allocated; the up blocks need it for their DupUp3D shortcut."""
        own_skip = self.shortcut is not None
        skip = self.shortcut(x, h, w)[0] if own_skip else x
        y = self.norm1(x)
        if own_skip and not keep_input:
            ttnn.deallocate(x)
        y2 = ttnn.silu(y)
        ttnn.deallocate(y)
        y, h, w = self.conv1(y2, h, w)
        ttnn.deallocate(y2)
        y2 = self.norm2(y)
        ttnn.deallocate(y)
        y = ttnn.silu(y2)
        ttnn.deallocate(y2)
        y2, h, w = self.conv2(y, h, w)
        ttnn.deallocate(y)
        out = ttnn.add(y2, skip)
        ttnn.deallocate(y2)
        if own_skip or not keep_input:
            ttnn.deallocate(skip)
        return out, h, w


class TTAttention:
    """Single-head self-attention over the H*W spatial positions (the 1x1 convs are linears)."""

    def __init__(self, dev, ckpt: LazyCheckpoint, prefix: str, dim: int, prec: VAEPrecision) -> None:
        g = lambda k: ckpt.get(prefix + k, torch.float32)
        self.dev = dev
        self.dim = dim
        self.prec = prec
        self.scale = 1.0 / math.sqrt(dim)
        self.norm = TTRmsNorm(dev, g("norm.gamma"), prec)

        # LazyCheckpoint.get returns owned tensors, so scale folding cannot alter another reader's weights.
        wqkv = g("to_qkv.weight").reshape(3 * dim, dim)  # 1x1 convolution == linear
        bqkv = g("to_qkv.bias")
        if prec.attn_impl == "matmul":
            # Fold the softmax scale into Q so the score matmul needs no extra pass.
            wqkv[:dim] *= self.scale
            bqkv[:dim] *= self.scale
        self.wqkv = ttnn.from_torch(
            wqkv.t().contiguous().to(torch.bfloat16),
            dtype=prec.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=DRAM,
        )
        self.bqkv = ttnn.from_torch(
            bqkv.reshape(1, 1, 1, 3 * dim).to(torch.bfloat16),
            dtype=prec.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=DRAM,
        )
        self.wproj = ttnn.from_torch(
            g("proj.weight").reshape(dim, dim).t().contiguous().to(torch.bfloat16),
            dtype=prec.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=DRAM,
        )
        self.bproj = ttnn.from_torch(
            g("proj.bias").reshape(1, 1, 1, dim).to(torch.bfloat16),
            dtype=prec.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=DRAM,
        )
        self.mm_config = prec.mm_compute(dev)
        self.sdpa_config = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
        )
        grid = dev.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=32, exp_approx_mode=True
        )

    def __call__(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        d = self.dim
        n = self.norm(x)
        qkv = ttnn.linear(n, self.wqkv, bias=self.bqkv, compute_kernel_config=self.mm_config, dtype=self.prec.act_dtype)
        ttnn.deallocate(n)
        s = h * w
        q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, s, d])
        k = ttnn.slice(qkv, [0, 0, 0, d], [1, 1, s, 2 * d])
        v = ttnn.slice(qkv, [0, 0, 0, 2 * d], [1, 1, s, 3 * d])
        ttnn.deallocate(qkv)

        if self.prec.attn_impl == "sdpa":
            out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                scale=self.scale,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_config,
            )
        else:
            scores = ttnn.matmul(
                q, k, transpose_b=True, compute_kernel_config=self.mm_config, dtype=self.prec.act_dtype
            )
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.mm_config)
            ttnn.deallocate(scores)
            out = ttnn.matmul(probs, v, compute_kernel_config=self.mm_config, dtype=self.prec.act_dtype)
            ttnn.deallocate(probs)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        proj = ttnn.linear(
            out, self.wproj, bias=self.bproj, compute_kernel_config=self.mm_config, dtype=self.prec.act_dtype
        )
        ttnn.deallocate(out)
        res = ttnn.add(proj, x)
        ttnn.deallocate(proj)
        ttnn.deallocate(x)
        return res, h, w


class TTMidBlock:
    def __init__(self, dev, ckpt: LazyCheckpoint, prefix: str, dim: int, prec: VAEPrecision) -> None:
        self.resnets = [TTResBlock(dev, ckpt, f"{prefix}resnets.{i}.", prec) for i in range(2)]
        self.attn = TTAttention(dev, ckpt, f"{prefix}attentions.0.", dim, prec)

    def __call__(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        x, h, w = self.resnets[0](x, h, w)
        x, h, w = self.attn(x, h, w)
        return self.resnets[1](x, h, w)


def dup_up_tt(x: ttnn.Tensor, h: int, w: int, in_c: int, out_c: int, factor_t: int) -> Tuple[ttnn.Tensor, int, int]:
    """`QwenImage21DupUp3D` for one frame and `factor_s == 2`, on `[1, 1, H*W, C]` bf16.

    Mirrors `reference.torch_vae.dup_up_fast`:
      * `factor_t == 2`, `in == out`     -> nearest 2x upsample
      * `factor_t == 2`, `in == 2 * out` -> odd channels, then nearest 2x upsample
      * `factor_t == 1`, `in == 2 * out` -> even channels become rows 2h, odd channels rows 2h+1,
                                            then each column is duplicated

    The input tensor is left allocated for the caller to free.
    """
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT) if x.layout != ttnn.ROW_MAJOR_LAYOUT else x
    src4 = ttnn.reshape(rm, (1, h, w, in_c))

    if factor_t == 2:
        if in_c == out_c:
            sel = src4
        else:
            assert in_c == 2 * out_c, (in_c, out_c)
            sel = ttnn.slice(src4, [0, 0, 0, 1], [1, h, w, in_c], [1, 1, 1, 2])
        out = ttnn.upsample(sel, scale_factor=2)
        if sel is not src4:
            ttnn.deallocate(sel)
        if rm is not x:
            ttnn.deallocate(rm)
        return ttnn.reshape(out, (1, 1, 4 * h * w, out_c)), 2 * h, 2 * w

    assert factor_t == 1 and in_c == 2 * out_c, (in_c, out_c, factor_t)
    even = ttnn.slice(src4, [0, 0, 0, 0], [1, h, w, in_c], [1, 1, 1, 2])
    odd = ttnn.slice(src4, [0, 0, 0, 1], [1, h, w, in_c], [1, 1, 1, 2])
    if rm is not x:
        ttnn.deallocate(rm)
    # Keep row width bounded by channels: flattening w*out_c makes wide-image reshapes exceed L1.
    pairs = ttnn.concat([even, odd], dim=3)
    ttnn.deallocate(even)
    ttnn.deallocate(odd)
    rows = ttnn.permute(ttnn.reshape(pairs, (h, w, 2, out_c)), (0, 2, 1, 3))
    ttnn.deallocate(pairs)
    rows = ttnn.reshape(rows, (1, 2 * h, w, out_c))
    out = ttnn.upsample(rows, scale_factor=[1, 2])
    ttnn.deallocate(rows)
    return ttnn.reshape(out, (1, 1, 4 * h * w, out_c)), 2 * h, 2 * w


class TTResidualUpBlock:
    """`num_res_blocks + 1` residual blocks, then (unless this is the last block) a nearest-2x
    upsample + 3x3 conv, with the DupUp3D gather of the block input added on."""

    def __init__(
        self,
        dev,
        ckpt: LazyCheckpoint,
        prefix: str,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal: bool,
        up_flag: bool,
        prec: VAEPrecision,
    ) -> None:
        self.in_dim, self.out_dim, self.up_flag = in_dim, out_dim, up_flag
        self.factor_t = 2 if (up_flag and temporal) else 1
        self.resnets = [TTResBlock(dev, ckpt, f"{prefix}resnets.{i}.", prec) for i in range(num_res_blocks + 1)]
        if up_flag:
            self.up_conv = TTConv2d(
                dev,
                ckpt.get(f"{prefix}upsampler.resample.1.weight", torch.float32),
                ckpt.get(f"{prefix}upsampler.resample.1.bias", torch.float32),
                padding=1,
                prec=prec,
            )

    def __call__(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        y, yh, yw = x, h, w
        for i, resnet in enumerate(self.resnets):
            # The first resnet must not free the block input: it is the DupUp3D shortcut source.
            y, yh, yw = resnet(y, yh, yw, keep_input=(i == 0 and self.up_flag))
        if not self.up_flag:
            return y, yh, yw

        rm = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(y)
        rm = ttnn.reshape(rm, (1, yh, yw, self.out_dim))
        up = ttnn.upsample(rm, scale_factor=2)
        ttnn.deallocate(rm)
        up = ttnn.reshape(up, (1, 1, 4 * yh * yw, self.out_dim))
        conv, ch, cw = self.up_conv(up, 2 * yh, 2 * yw)
        ttnn.deallocate(up)

        short, sh, sw = dup_up_tt(x, h, w, self.in_dim, self.out_dim, self.factor_t)
        ttnn.deallocate(x)
        assert (sh, sw) == (ch, cw), ((sh, sw), (ch, cw))
        short_t = ttnn.to_layout(short, ttnn.TILE_LAYOUT)
        ttnn.deallocate(short)
        out = ttnn.add(conv, short_t)
        ttnn.deallocate(conv)
        ttnn.deallocate(short_t)
        return out, ch, cw


# ----------------------------------------------------------------------------------------- decoder


class QwenImageVAEDecoder:
    """The `post_quant_conv` + decoder + clamp half of `AutoencoderKLQwenImage21.decode`.

    `decode` is the host-in, host-out entry point (see its docstring for the accepted latent
    shapes), `forward` the device-only one, and `capture_trace` / `decode_traced` replay the whole
    decode from a captured metal trace.
    """

    def __init__(
        self, device, ckpt: Optional[LazyCheckpoint] = None, prec: Optional[VAEPrecision] = None, cfg: VAEConfig = VAE
    ) -> None:
        if ckpt is None:
            from ..common.weights import vae_ckpt

            ckpt = vae_ckpt()
        self.dev = device
        self.cfg = cfg
        self.prec = prec or VAEPrecision()
        g = lambda k: ckpt.get(k, torch.float32)

        mult = list(cfg.dim_mult)
        dims = [cfg.decoder_base_dim * u for u in [mult[-1]] + mult[::-1]]  # [1152, 1152, 1152, 576, 288, 144]
        self.dims = dims

        self.post_quant_conv = TTConv2d(
            device, g("post_quant_conv.weight"), g("post_quant_conv.bias"), padding=0, prec=self.prec
        )
        self.conv_in = TTConv2d(
            device, g("decoder.conv_in.weight"), g("decoder.conv_in.bias"), padding=1, prec=self.prec
        )
        self.mid_block = TTMidBlock(device, ckpt, "decoder.mid_block.", dims[0], self.prec)
        self.up_blocks: List[TTResidualUpBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(mult) - 1
            self.up_blocks.append(
                TTResidualUpBlock(
                    device,
                    ckpt,
                    f"decoder.up_blocks.{i}.",
                    in_dim,
                    out_dim,
                    cfg.num_res_blocks,
                    temporal=cfg.temporal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    prec=self.prec,
                )
            )
        self.norm_out = TTRmsNorm(device, g("decoder.norm_out.gamma"), self.prec)
        self.conv_out = TTConv2d(
            device,
            g("decoder.conv_out.weight"),
            g("decoder.conv_out.bias"),
            padding=1,
            prec=self.prec,
            pad_out_channels=False,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self._trace_id = None
        self._trace_in: Optional[ttnn.Tensor] = None
        self._trace_out: Optional[ttnn.Tensor] = None

    # ----------------------------------------------------------------------------- host <-> device
    @staticmethod
    def _to_nhwc(latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() == 5:
            assert latents.shape[2] == 1, "single-frame decoder"
            latents = latents[:, :, 0]
        assert latents.dim() == 4, latents.shape
        b, c, h, w = latents.shape
        assert b == 1, "batch 1 only"
        return latents.permute(0, 2, 3, 1).reshape(1, 1, h * w, c).contiguous()

    def upload(self, latents: torch.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        h, w = int(latents.shape[-2]), int(latents.shape[-1])
        x = self._to_nhwc(latents.float())
        dev_x = ttnn.from_torch(
            x.to(torch.bfloat16),
            dtype=self.prec.act_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=DRAM,
        )
        return dev_x, h, w

    def _device_input(self, x: ttnn.Tensor) -> Tuple[ttnn.Tensor, int, int]:
        """Accept a channels-last device latent as `[1, H, W, C]` or square `[1, 1, H*W, C]`."""
        shape = list(x.shape)
        if len(shape) == 4 and shape[1] != 1:
            h, w = shape[1], shape[2]
            return ttnn.reshape(x, (1, 1, h * w, shape[3])), h, w
        s = shape[-2]
        h = int(math.isqrt(s))
        if h * h != s:
            raise ValueError(f"cannot infer H, W from a flattened latent of {s} positions; pass [1, H, W, C]")
        return x, h, h

    # -------------------------------------------------------------------------------------- device
    def forward(self, x: ttnn.Tensor, h: int, w: int) -> Tuple[ttnn.Tensor, int, int]:
        """Device-only decode: `[1, 1, H*W, 64]` TILE bf16 -> `[1, 1, 16H*16W, 4]` ROW_MAJOR.

        `x` is left allocated so a traced call can keep reusing the same persistent input buffer.
        """
        z, h, w = self.post_quant_conv(x, h, w)
        y, h, w = self.conv_in(z, h, w)
        ttnn.deallocate(z)
        y, h, w = self.mid_block(y, h, w)
        for up_block in self.up_blocks:
            y, h, w = up_block(y, h, w)
        n = self.norm_out(y)
        ttnn.deallocate(y)
        s = ttnn.silu(n)
        ttnn.deallocate(n)
        out, h, w = self.conv_out(s, h, w)
        ttnn.deallocate(s)
        clamped = ttnn.clamp(out, min=-1.0, max=1.0)
        ttnn.deallocate(out)
        return clamped, h, w

    def _download(self, out: ttnn.Tensor, h: int, w: int) -> torch.Tensor:
        t = ttnn.to_torch(out).float().reshape(1, h, w, -1)[..., : self.cfg.out_channels]
        return t.permute(0, 3, 1, 2).contiguous()

    def decode(self, latents, *, traced: bool = False) -> torch.Tensor:
        """Decode one latent to a torch `[1, 4, 16H, 16W]` tensor in [-1, 1].

        `latents` is a torch `[1, 64, H, W]` / `[1, 64, 1, H, W]` tensor, or an already-uploaded
        channels-last device tensor shaped `[1, H, W, 64]` (or `[1, 1, H*W, 64]` for a square
        latent, where H can be recovered).
        """
        if traced:
            return self.decode_traced(latents)
        if isinstance(latents, ttnn.Tensor):
            x, h, w = self._device_input(latents)
        else:
            x, h, w = self.upload(latents)
        out, oh, ow = self.forward(x, h, w)
        result = self._download(out, oh, ow)
        ttnn.deallocate(out)
        if not isinstance(latents, ttnn.Tensor):
            ttnn.deallocate(x)  # ours to free; a device latent passed in belongs to the caller
        return result

    # --------------------------------------------------------------------------------------- trace
    def capture_trace(self, latents: torch.Tensor) -> None:
        """Warm up (prepares conv weights, fills caches) then capture the whole decode in a trace."""
        x, h, w = self.upload(latents)
        self._trace_in = x
        warm, _, _ = self.forward(x, h, w)
        ttnn.deallocate(warm)
        ttnn.synchronize_device(self.dev)

        self._trace_id = ttnn.begin_trace_capture(self.dev, cq_id=0)
        self._trace_out, self._trace_oh, self._trace_ow = self.forward(self._trace_in, h, w)
        ttnn.end_trace_capture(self.dev, self._trace_id, cq_id=0)
        ttnn.synchronize_device(self.dev)

    def decode_traced(self, latents: torch.Tensor) -> torch.Tensor:
        assert self._trace_id is not None, "call capture_trace() first"
        host = ttnn.from_torch(
            self._to_nhwc(latents.float()).to(torch.bfloat16), dtype=self.prec.act_dtype, layout=ttnn.TILE_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host, self._trace_in)
        ttnn.execute_trace(self.dev, self._trace_id, cq_id=0, blocking=True)
        return self._download(self._trace_out, self._trace_oh, self._trace_ow)

    def release_trace(self) -> None:
        """Free the trace and its persistent input / output buffers."""
        if self._trace_id is not None:
            ttnn.release_trace(self.dev, self._trace_id)
            self._trace_id = None
        self._trace_in = None
        self._trace_out = None

    # ---------------------------------------------------------------------------------- timing aid
    def time_decode(self, latents: torch.Tensor, iters: int = 1, traced: bool = False) -> Tuple[torch.Tensor, float]:
        """Return (last output, best wall-clock seconds) for `iters` decodes."""
        best = float("inf")
        out = None
        for _ in range(iters):
            ttnn.synchronize_device(self.dev)
            t0 = time.perf_counter()
            out = self.decode(latents, traced=traced)
            ttnn.synchronize_device(self.dev)
            best = min(best, time.perf_counter() - t0)
        return out, best
