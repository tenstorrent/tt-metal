# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN port (incremental) of Kokoro ISTFTNet `Generator`.

This ports the neural conv/resblock stack and returns the pre-iSTFT tensor `x`
from the reference implementation right before:
  spec = exp(x[:freq_bins])
  phase = sin(x[freq_bins:])

The harmonic source generation + STFT and final iSTFT inversion will be ported next.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import ttnn
from models.demos.kokoro.tt.ttnn_kokoro_conv import Conv1dParams, ConvTranspose1dParams, conv1d_nlc, weight_norm_weight
from models.demos.kokoro.tt.ttnn_kokoro_norm import AdaIN1dParams, InstanceNorm1dParams, adain_1d_nlc


@dataclass(frozen=True)
class AdaINResBlock1LayerParams:
    conv1: Conv1dParams
    conv2: Conv1dParams
    adain1: AdaIN1dParams
    adain2: AdaIN1dParams
    alpha1: ttnn.Tensor
    alpha2: ttnn.Tensor


@dataclass(frozen=True)
class AdaINResBlock1Params:
    layers: list[AdaINResBlock1LayerParams]


def _preprocess_adain_istft(adain: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16) -> AdaIN1dParams:
    in_w = getattr(adain.norm, "weight", None)
    in_b = getattr(adain.norm, "bias", None)
    inst = InstanceNorm1dParams(
        weight=ttnn.from_torch(in_w.detach().cpu(), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if in_w is not None
        else None,
        bias=ttnn.from_torch(in_b.detach().cpu(), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if in_b is not None
        else None,
        eps=adain.norm.eps,
    )
    fc_w = ttnn.from_torch(adain.fc.weight.detach().cpu(), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    fc_b = ttnn.from_torch(
        adain.fc.bias.detach().cpu().reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    return AdaIN1dParams(fc_weight=fc_w, fc_bias=fc_b, instancenorm=inst)


def _preprocess_weight_norm_conv1d(conv_wn: nn.Module, *, weights_dtype=ttnn.bfloat16) -> Conv1dParams:
    w = weight_norm_weight(conv_wn.weight_v.detach().cpu(), conv_wn.weight_g.detach().cpu())
    b = conv_wn.bias.detach().cpu() if conv_wn.bias is not None else None
    w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = (
        ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        if b is not None
        else None
    )
    return Conv1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=conv_wn.in_channels,
        out_channels=conv_wn.out_channels,
        kernel_size=conv_wn.kernel_size[0],
        stride=conv_wn.stride[0],
        padding=conv_wn.padding[0],
        groups=conv_wn.groups,
    )


def _preprocess_weight_norm_convtranspose1d(
    conv_wn: nn.Module, *, weights_dtype=ttnn.bfloat16
) -> ConvTranspose1dParams:
    # weight_norm wraps ConvTranspose1d; effective weight is [in, out/groups, k] for ConvTranspose1d
    w = weight_norm_weight(conv_wn.weight_v.detach().cpu(), conv_wn.weight_g.detach().cpu())
    w2 = w.unsqueeze(2)  # [in, out/groups, 1, k]
    b = conv_wn.bias.detach().cpu() if conv_wn.bias is not None else None
    w_tt = ttnn.from_torch(w2, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = (
        ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        if b is not None
        else None
    )
    return ConvTranspose1dParams(
        weight=w_tt,
        bias=b_tt,
        in_channels=conv_wn.in_channels,
        out_channels=conv_wn.out_channels,
        kernel_size=conv_wn.kernel_size[0],
        stride=conv_wn.stride[0],
        padding=conv_wn.padding[0],
        output_padding=conv_wn.output_padding[0],
        groups=conv_wn.groups,
    )


def preprocess_adain_resblock1(
    torch_block: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> AdaINResBlock1Params:
    layers: list[AdaINResBlock1LayerParams] = []
    for c1, c2, n1, n2, a1, a2 in zip(
        torch_block.convs1,
        torch_block.convs2,
        torch_block.adain1,
        torch_block.adain2,
        torch_block.alpha1,
        torch_block.alpha2,
    ):
        conv1 = _preprocess_weight_norm_conv1d(c1, weights_dtype=weights_dtype)
        conv2 = _preprocess_weight_norm_conv1d(c2, weights_dtype=weights_dtype)
        adain1 = _preprocess_adain_istft(n1, device, weights_dtype=weights_dtype)
        adain2 = _preprocess_adain_istft(n2, device, weights_dtype=weights_dtype)
        # store as [1, 1, C] (ROW_MAJOR) for broadcast without L1 temp allocations
        alpha1 = ttnn.from_torch(
            a1.detach().cpu().permute(0, 2, 1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        alpha2 = ttnn.from_torch(
            a2.detach().cpu().permute(0, 2, 1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        layers.append(
            AdaINResBlock1LayerParams(
                conv1=conv1, conv2=conv2, adain1=adain1, adain2=adain2, alpha1=alpha1, alpha2=alpha2
            )
        )
    return AdaINResBlock1Params(layers=layers)


def _snake_like_nlc(x_nlc: ttnn.Tensor, alpha_11c: ttnn.Tensor) -> ttnn.Tensor:
    # x + (1/alpha) * sin(alpha*x)^2
    # Avoid `repeat` (can allocate L1_SMALL temps); do broadcast in ROW_MAJOR.
    x = x_nlc if x_nlc.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(x_nlc, ttnn.ROW_MAJOR_LAYOUT)
    a = alpha_11c if alpha_11c.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(alpha_11c, ttnn.ROW_MAJOR_LAYOUT)
    ax = ttnn.multiply(x, a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s = ttnn.sin(ax)
    s2 = ttnn.pow(s, 2.0)
    inv_a = ttnn.reciprocal(alpha_11c)  # [1,1,C]
    inv_a = inv_a if inv_a.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(inv_a, ttnn.ROW_MAJOR_LAYOUT)
    y = ttnn.multiply(s2, inv_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.add(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out if x_nlc.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(out, ttnn.TILE_LAYOUT)


def _maybe_to_interleaved(x: ttnn.Tensor) -> ttnn.Tensor:
    if ttnn.is_sharded(x):
        return ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return x


def _crop_len_nlc(x: ttnn.Tensor, out_len: int) -> ttnn.Tensor:
    if int(x.shape[1]) == int(out_len):
        return x
    return ttnn.slice(x, (0, 0, 0), (x.shape[0], out_len, x.shape[2]), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def adain_resblock1_forward_nlc(
    *,
    x_nlc: ttnn.Tensor,
    style_bs: ttnn.Tensor,
    params: AdaINResBlock1Params,
    device: ttnn.Device,
    compute_kernel_config,
) -> ttnn.Tensor:
    x = x_nlc
    for layer in params.layers:
        xt = adain_1d_nlc(
            x_nlc=x,
            s_bc=style_bs,
            params=layer.adain1,
            compute_kernel_config=compute_kernel_config,
        )
        xt = _snake_like_nlc(xt, layer.alpha1)
        xt = conv1d_nlc(
            x_nlc=xt,
            params=layer.conv1,
            device=device,
            compute_config=compute_kernel_config,
        )
        xt = _maybe_to_interleaved(xt)

        xt = adain_1d_nlc(
            x_nlc=xt,
            s_bc=style_bs,
            params=layer.adain2,
            compute_kernel_config=compute_kernel_config,
        )
        xt = _snake_like_nlc(xt, layer.alpha2)
        xt = conv1d_nlc(
            x_nlc=xt,
            params=layer.conv2,
            device=device,
            compute_config=compute_kernel_config,
        )
        xt = _maybe_to_interleaved(xt)
        x = _maybe_to_interleaved(x)
        out_len = min(int(xt.shape[1]), int(x.shape[1]))
        xt = _crop_len_nlc(xt, out_len)
        x = _crop_len_nlc(x, out_len)
        x = ttnn.add(x, xt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return x


@dataclass(frozen=True)
class GeneratorCoreParams:
    ups_conv: list[Conv1dParams]
    ups_stride: list[int]
    noise_convs: list[Conv1dParams]
    noise_res: list[AdaINResBlock1Params]
    resblocks: list[AdaINResBlock1Params]
    conv_post: Conv1dParams
    post_n_fft: int
    num_kernels: int
    num_upsamples: int


def preprocess_generator_core(
    torch_gen: nn.Module, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> GeneratorCoreParams:
    # Implement ConvTranspose1d (stride>1) as: zero-insert upsample + Conv1d with flipped kernel.
    ups_conv: list[Conv1dParams] = []
    ups_stride: list[int] = []
    for u in torch_gen.ups:
        assert u.output_padding[0] == 0, "Generator ups assumes output_padding=0"
        s = int(u.stride[0])
        k = int(u.kernel_size[0])
        p = int(u.padding[0])
        # effective transposed-conv weight is [in, out, k] for ConvTranspose1d
        w = weight_norm_weight(u.weight_v.detach().cpu(), u.weight_g.detach().cpu())
        w = w.permute(1, 0, 2).flip(-1).contiguous()  # [out, in, k] for Conv1d
        b = u.bias.detach().cpu() if u.bias is not None else None
        w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_tt = (
            ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if b is not None
            else None
        )
        ups_conv.append(
            Conv1dParams(
                weight=w_tt,
                bias=b_tt,
                in_channels=u.in_channels,
                out_channels=u.out_channels,
                kernel_size=k,
                stride=1,
                padding=k - 1 - p,
                groups=u.groups,
            )
        )
        ups_stride.append(s)
    # noise_convs are plain nn.Conv1d (not weight_norm). Preprocess as regular conv weights.
    nc: list[Conv1dParams] = []
    for c in torch_gen.noise_convs:
        w = c.weight.detach().cpu()
        b = c.bias.detach().cpu() if c.bias is not None else None
        w_tt = ttnn.from_torch(w, dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_tt = (
            ttnn.from_torch(b.reshape(1, 1, 1, -1), dtype=weights_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if b is not None
            else None
        )
        nc.append(
            Conv1dParams(
                weight=w_tt,
                bias=b_tt,
                in_channels=c.in_channels,
                out_channels=c.out_channels,
                kernel_size=c.kernel_size[0],
                stride=c.stride[0],
                padding=c.padding[0],
                groups=c.groups,
            )
        )
    noise_convs = nc

    noise_res = [preprocess_adain_resblock1(b, device, weights_dtype=weights_dtype) for b in torch_gen.noise_res]
    resblocks = [preprocess_adain_resblock1(b, device, weights_dtype=weights_dtype) for b in torch_gen.resblocks]
    conv_post = _preprocess_weight_norm_conv1d(torch_gen.conv_post, weights_dtype=weights_dtype)
    return GeneratorCoreParams(
        ups_conv=ups_conv,
        ups_stride=ups_stride,
        noise_convs=noise_convs,
        noise_res=noise_res,
        resblocks=resblocks,
        conv_post=conv_post,
        post_n_fft=torch_gen.post_n_fft,
        num_kernels=torch_gen.num_kernels,
        num_upsamples=torch_gen.num_upsamples,
    )


def _zero_insert_upsample_nlc(x_nlc: ttnn.Tensor, *, stride: int, device: ttnn.Device) -> ttnn.Tensor:
    """
    Emulate ConvTranspose1d stride via zero-insertion.
    Output length: (L-1)*stride + 1
    """
    assert stride >= 1
    if stride == 1:
        return x_nlc
    x_rep = ttnn.repeat_interleave(x_nlc, repeats=stride, dim=1)
    out_len = (int(x_nlc.shape[1]) - 1) * stride + 1
    x_rep = ttnn.slice(
        x_rep, (0, 0, 0), (x_rep.shape[0], out_len, x_rep.shape[2]), memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # mask: 1 at every stride-th position, else 0
    mask = torch.zeros((1, out_len, 1), dtype=torch.float32)
    mask[0, ::stride, 0] = 1.0
    mask_tt = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    x_rm = x_rep if x_rep.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(x_rep, ttnn.ROW_MAJOR_LAYOUT)
    y = ttnn.multiply(x_rm, mask_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return y if x_nlc.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(y, ttnn.TILE_LAYOUT)


class TtKokoroGeneratorCore:
    def __init__(self, device: ttnn.Device, params: GeneratorCoreParams):
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
        )

    def __call__(self, *, x_bct: ttnn.Tensor, style_s: torch.Tensor, har_per_stage: list[torch.Tensor]) -> ttnn.Tensor:
        """
        x_bct: [B, C, T] TTNN
        style_s: [B, style_dim] torch
        har_per_stage: list of torch tensors, each [B, gen_n_fft+2, Th_i], used for noise injection at each stage.

        Returns: x_post_bct torch-like logits on device [B, post_n_fft+2, Tout]
        """
        style = ttnn.from_torch(
            style_s.detach().cpu(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Work in NLC ([B, L, C]) to avoid permute/transpose edge cases.
        x = ttnn.permute(x_bct, (0, 2, 1))
        for i in range(self.params.num_upsamples):
            x = ttnn.leaky_relu(x, negative_slope=0.1)

            # x_source = noise_convs[i](har) then noise_res[i]
            har = ttnn.from_torch(
                har_per_stage[i].detach().cpu(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            har_nlc = ttnn.permute(har, (0, 2, 1))
            x_source = conv1d_nlc(
                x_nlc=har_nlc,
                params=self.params.noise_convs[i],
                device=self.device,
                compute_config=self.compute_kernel_config,
            )
            x_source = _maybe_to_interleaved(x_source)
            x_source = adain_resblock1_forward_nlc(
                x_nlc=x_source,
                style_bs=style,
                params=self.params.noise_res[i],
                device=self.device,
                compute_kernel_config=self.compute_kernel_config,
            )

            # upsample (ConvTranspose1d) implemented as zero-insert + conv1d
            x = _zero_insert_upsample_nlc(x, stride=self.params.ups_stride[i], device=self.device)
            x = conv1d_nlc(
                x_nlc=x, params=self.params.ups_conv[i], device=self.device, compute_config=self.compute_kernel_config
            )
            x = _maybe_to_interleaved(x)

            # reflection_pad for last stage is skipped here; handled later (or ported with slice/concat)
            out_len = min(int(x.shape[1]), int(x_source.shape[1]))
            x = _crop_len_nlc(x, out_len)
            x_source = _crop_len_nlc(x_source, out_len)
            x = ttnn.add(x, x_source, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            xs = None
            for j in range(self.params.num_kernels):
                blk = self.params.resblocks[i * self.params.num_kernels + j]
                y = adain_resblock1_forward_nlc(
                    x_nlc=x,
                    style_bs=style,
                    params=blk,
                    device=self.device,
                    compute_kernel_config=self.compute_kernel_config,
                )
                xs = y if xs is None else ttnn.add(xs, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.multiply(xs, 1.0 / float(self.params.num_kernels))

        x = ttnn.leaky_relu(x, negative_slope=0.01)
        x = conv1d_nlc(
            x_nlc=x, params=self.params.conv_post, device=self.device, compute_config=self.compute_kernel_config
        )
        x = _maybe_to_interleaved(x)
        return ttnn.permute(x, (0, 2, 1))
