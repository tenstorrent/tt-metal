# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host preprocessing for TTNN Kokoro ``Generator`` (weights from PyTorch reference)."""

from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import ttnn

from .kokoro_istftnet import AdaINResBlock1 as TorchAdaINResBlock1
from .kokoro_source_module_preprocess import preprocess_source_module_hn_nsf_parameters
from .kokoro_stft_preprocess import preprocess_kokoro_conv_stft_parameters


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _conv_rm(conv: nn.Conv1d) -> dict[str, Any]:
    w = conv.weight.data.unsqueeze(-1).contiguous()
    oc = w.shape[0]
    b = conv.bias.data if conv.bias is not None else None
    bias_t = torch.reshape(b, (1, 1, 1, oc)) if b is not None else None
    return {
        "weight": ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
        "bias": None if bias_t is None else ttnn.from_torch(bias_t, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
    }


def _lin_dram(linear: nn.Linear, device) -> dict[str, Any]:
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    return {
        "fc_weight": ttnn.from_torch(
            linear.weight.data.T.contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM,
        ),
        "fc_bias": ttnn.from_torch(
            linear.bias.data,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM,
        ),
    }


def _inst_dram(norm: nn.InstanceNorm1d, device) -> dict[str, Any]:
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    c = norm.num_features
    w = norm.weight.data.reshape(1, c, 1)
    b = norm.bias.data.reshape(1, c, 1)
    return {
        "inst_weight": ttnn.from_torch(
            w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
        ),
        "inst_bias": ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM),
    }


def preprocess_adain_resblock1_parameters(block: TorchAdaINResBlock1, device) -> dict[str, Any]:
    for c1 in block.convs1:
        nn_utils.remove_weight_norm(c1)
    for c2 in block.convs2:
        nn_utils.remove_weight_norm(c2)
    channels = int(block.convs1[0].weight.shape[0])
    style_dim = int(block.adain1[0].fc.weight.shape[1])
    layers: List[dict[str, Any]] = []
    for i in range(len(block.convs1)):
        c1 = block.convs1[i]
        c2 = block.convs2[i]
        d1 = int(c1.dilation[0])
        d2 = int(c2.dilation[0])
        k1 = int(c1.weight.shape[2])
        k2 = int(c2.weight.shape[2])
        p1 = _get_padding(k1, d1)
        p2 = _get_padding(k2, d2)
        a1 = block.alpha1[i].data.clone().float().clamp(min=1e-4)
        a2 = block.alpha2[i].data.clone().float().clamp(min=1e-4)
        inv1 = (1.0 / a1).reshape(1, channels, 1)
        inv2 = (1.0 / a2).reshape(1, channels, 1)
        layers.append(
            {
                "norm1": {**_lin_dram(block.adain1[i].fc, device), **_inst_dram(block.adain1[i].norm, device)},
                "norm2": {**_lin_dram(block.adain2[i].fc, device), **_inst_dram(block.adain2[i].norm, device)},
                "alpha1": ttnn.from_torch(
                    a1.reshape(1, channels, 1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                "inv_alpha1": ttnn.from_torch(
                    inv1,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                "alpha2": ttnn.from_torch(
                    a2.reshape(1, channels, 1),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                "inv_alpha2": ttnn.from_torch(
                    inv2,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                "conv1": {**_conv_rm(c1), "kernel_size": k1, "padding_h": p1, "dilation_h": d1},
                "conv2": {**_conv_rm(c2), "kernel_size": k2, "padding_h": p2, "dilation_h": d2},
            }
        )
    return {
        "channels": channels,
        "style_dim": style_dim,
        "eps": float(block.adain1[0].norm.eps),
        "layers": layers,
    }


def _safe_remove_weight_norm(m: nn.Module) -> None:
    try:
        nn_utils.remove_weight_norm(m)
    except ValueError:
        pass


def preprocess_kokoro_generator_parameters(
    gen: nn.Module,
    device,
    *,
    f0_upsampled_time: int,
    disable_complex: bool = False,
) -> dict[str, Any]:
    """
    Args:
        gen: ``Decoder.generator`` (``kokoro_istftnet.Generator``).
        f0_upsampled_time: Time length after ``f0_upsamp`` (matches ``SourceModuleHnNSF`` preprocess).
        disable_complex: Same as reference ``Decoder`` (``CustomSTFT`` vs ``TorchSTFT``).
    """
    for m in list(gen.ups) + [gen.conv_post]:
        _safe_remove_weight_norm(m)
    for n in gen.noise_convs:
        _safe_remove_weight_norm(n)

    stft_p = preprocess_kokoro_conv_stft_parameters(
        device,
        filter_length=int(gen.post_n_fft),
        hop_length=int(gen.stft.hop_length) if hasattr(gen.stft, "hop_length") else 200,
        win_length=int(gen.stft.win_length) if hasattr(gen.stft, "win_length") else int(gen.post_n_fft),
        center=True,
        pad_mode="replicate",
    )

    m_src = preprocess_source_module_hn_nsf_parameters(gen.m_source, device, time_len=f0_upsampled_time)

    noise_w: List[dict[str, Any]] = []
    for conv in gen.noise_convs:
        w = conv.weight.data.unsqueeze(-1).contiguous()
        noise_w.append(
            {
                "weight": ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
                "bias": None
                if conv.bias is None
                else ttnn.from_torch(
                    torch.reshape(conv.bias.data, (1, 1, 1, conv.out_channels)),
                    dtype=ttnn.float32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "stride": int(conv.stride[0]),
                "kernel_size": int(conv.kernel_size[0]),
                "padding": int(conv.padding[0]),
                "in_channels": int(conv.in_channels),
                "out_channels": int(conv.out_channels),
            }
        )

    ups_w: List[dict[str, Any]] = []
    for m in gen.ups:
        w = m.weight.data.unsqueeze(-1).contiguous()
        op = int(m.output_padding[0]) if isinstance(m.output_padding, tuple) else int(m.output_padding)
        ups_w.append(
            {
                "weight": ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
                "bias": None
                if m.bias is None
                else ttnn.from_torch(
                    torch.reshape(m.bias.data, (1, 1, 1, m.out_channels)),
                    dtype=ttnn.float32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "stride": int(m.stride[0]),
                "kernel_size": int(m.kernel_size[0]),
                "padding": int(m.padding[0]),
                "output_padding": op,
                "in_channels": int(m.in_channels),
                "out_channels": int(m.out_channels),
            }
        )

    noise_res_p = [preprocess_adain_resblock1_parameters(b, device) for b in gen.noise_res]
    resblocks_p = [preprocess_adain_resblock1_parameters(b, device) for b in gen.resblocks]

    conv_post = _conv_rm(gen.conv_post)

    f0_scale = (
        float(gen.f0_upsamp.scale_factor)
        if isinstance(gen.f0_upsamp.scale_factor, (int, float))
        else float(gen.f0_upsamp.scale_factor[0])
    )

    return {
        "style_dim": int(gen.noise_res[0].adain1[0].fc.weight.shape[1]),
        "num_upsamples": int(gen.num_upsamples),
        "num_kernels": int(gen.num_kernels),
        "post_n_fft": int(gen.post_n_fft),
        "f0_up_scale": f0_scale,
        "m_source": m_src,
        "stft": stft_p,
        "noise_convs": noise_w,
        "noise_res": noise_res_p,
        "ups": ups_w,
        "resblocks": resblocks_p,
        "conv_post": conv_post,
        "inv_num_kernels": float(1.0 / float(gen.num_kernels)),
        "disable_complex": bool(disable_complex),
    }
