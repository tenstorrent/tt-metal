"""
Tests for causal vocoder components:
    1. TtCausalConv1d vs PyTorch CausalConv1d
    2. TtCausalConv1dUpsample vs PyTorch CausalConv1dUpsample
    3. TtCausalHiFTGenerator decode vs PyTorch CausalHiFTGenerator decode
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../ref/CosyVoice")))
from cosyvoice.hifigan.generator import CausalHiFTGenerator
from cosyvoice.transformer.convolution import CausalConv1d, CausalConv1dUpsample

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosy_voice.tt.vocoder.generator import (
    TtCausalConv1d,
    TtCausalConv1dUpsample,
    TtCausalHiFTGenerator,
)


def test_causal_conv1d(device):
    """Test TtCausalConv1d against PyTorch CausalConv1d."""
    torch.manual_seed(42)
    in_ch, out_ch, ks, dilation = 256, 256, 3, 1
    seq_len = 128

    pt_conv = CausalConv1d(in_ch, out_ch, ks, dilation=dilation)
    pt_conv.eval()

    x_pt = torch.randn(1, in_ch, seq_len)
    with torch.no_grad():
        y_pt = pt_conv(x_pt)

    # Extract fused weight (weight_norm may be present)
    w = pt_conv.weight.data.clone()
    b = pt_conv.bias.data.clone()

    tt_conv = TtCausalConv1d(device, w, b, in_ch, out_ch, ks, dilation=dilation)
    x_tt = x_pt.transpose(1, 2).unsqueeze(1)  # [1, 1, L, C]
    x_tt = ttnn.from_torch(x_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = tt_conv(x_tt)
    y_tt_cpu = ttnn.to_torch(y_tt).squeeze(1).transpose(1, 2)

    passing, pcc_msg = comp_pcc(y_pt, y_tt_cpu, 0.99)
    print(f"CausalConv1d PCC: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"


def test_causal_conv1d_upsample(device):
    """Test TtCausalConv1dUpsample against PyTorch CausalConv1dUpsample."""
    torch.manual_seed(42)
    in_ch, out_ch, ks, stride = 512, 256, 16, 8
    seq_len = 32

    pt_up = CausalConv1dUpsample(in_ch, out_ch, ks, stride)
    pt_up.eval()

    x_pt = torch.randn(1, in_ch, seq_len)
    with torch.no_grad():
        y_pt = pt_up(x_pt)

    w = pt_up.weight.data.clone()
    b = pt_up.bias.data.clone()

    tt_up = TtCausalConv1dUpsample(device, w, b, in_ch, out_ch, ks, stride)
    x_tt = x_pt.transpose(1, 2).unsqueeze(1)
    x_tt = ttnn.from_torch(x_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = tt_up(x_tt)
    y_tt_cpu = ttnn.to_torch(y_tt).squeeze(1).transpose(1, 2)

    # Truncate to match (padding differences)
    min_len = min(y_pt.shape[2], y_tt_cpu.shape[2])
    passing, pcc_msg = comp_pcc(y_pt[:, :, :min_len], y_tt_cpu[:, :, :min_len], 0.99)
    print(f"CausalConv1dUpsample PCC: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"


def extract_causal_hift_state_dict(pt_model):
    """Extract fused weights from a CausalHiFTGenerator."""
    sd = {}

    sd["conv_pre.weight"] = pt_model.conv_pre.weight.data.clone()
    sd["conv_pre.bias"] = pt_model.conv_pre.bias.data.clone()

    for i, up in enumerate(pt_model.ups):
        sd[f"ups.{i}.weight"] = up.weight.data.clone()
        sd[f"ups.{i}.bias"] = up.bias.data.clone()

    for i, sd_layer in enumerate(pt_model.source_downs):
        sd[f"source_downs.{i}.weight"] = sd_layer.weight.data.clone()
        sd[f"source_downs.{i}.bias"] = sd_layer.bias.data.clone()

    for i, rb in enumerate(pt_model.source_resblocks):
        for j in range(len(rb.convs1)):
            sd[f"source_resblocks.{i}.convs1.{j}.weight"] = rb.convs1[j].weight.data.clone()
            sd[f"source_resblocks.{i}.convs1.{j}.bias"] = rb.convs1[j].bias.data.clone()
            sd[f"source_resblocks.{i}.convs2.{j}.weight"] = rb.convs2[j].weight.data.clone()
            sd[f"source_resblocks.{i}.convs2.{j}.bias"] = rb.convs2[j].bias.data.clone()
            sd[f"source_resblocks.{i}.activations1.{j}.alpha"] = rb.activations1[j].alpha.data.clone()
            sd[f"source_resblocks.{i}.activations2.{j}.alpha"] = rb.activations2[j].alpha.data.clone()

    for i, rb in enumerate(pt_model.resblocks):
        for j in range(len(rb.convs1)):
            sd[f"resblocks.{i}.convs1.{j}.weight"] = rb.convs1[j].weight.data.clone()
            sd[f"resblocks.{i}.convs1.{j}.bias"] = rb.convs1[j].bias.data.clone()
            sd[f"resblocks.{i}.convs2.{j}.weight"] = rb.convs2[j].weight.data.clone()
            sd[f"resblocks.{i}.convs2.{j}.bias"] = rb.convs2[j].bias.data.clone()
            sd[f"resblocks.{i}.activations1.{j}.alpha"] = rb.activations1[j].alpha.data.clone()
            sd[f"resblocks.{i}.activations2.{j}.alpha"] = rb.activations2[j].alpha.data.clone()

    sd["conv_post.weight"] = pt_model.conv_post.weight.data.clone()
    sd["conv_post.bias"] = pt_model.conv_post.bias.data.clone()

    return sd


@pytest.mark.parametrize("mel_length", [32])
def test_causal_hift_generator_decode(device, mel_length):
    """
    Test the causal decode path against PyTorch CausalHiFTGenerator.
    Bypasses f0/source/STFT by providing pre-computed s_stft directly.
    """
    torch.manual_seed(42)
    batch_size = 1
    in_channels = 80
    base_channels = 512
    upsample_rates = [8, 5, 3]
    upsample_kernel_sizes = [16, 11, 7]
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    source_resblock_kernel_sizes = [7, 7, 11]
    source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    istft_params = {"n_fft": 16, "hop_len": 4}
    conv_pre_look_right = 4

    pt_model = CausalHiFTGenerator(
        in_channels=in_channels,
        base_channels=base_channels,
        nb_harmonics=8,
        sampling_rate=22050,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        istft_params=istft_params,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        source_resblock_kernel_sizes=source_resblock_kernel_sizes,
        source_resblock_dilation_sizes=source_resblock_dilation_sizes,
        lrelu_slope=0.1,
        conv_pre_look_right=conv_pre_look_right,
        f0_predictor=None,
    )
    pt_model.eval()

    x_pt = torch.randn(batch_size, in_channels, mel_length)

    # Compute s_stft length: with center=True STFT, frames = floor(T/hop_len) + 1
    # We need the frames to be compatible with all source_downs outputs matching ups outputs
    total_upsample = int(np.prod(upsample_rates))
    target_stft_frames = mel_length * total_upsample + 1
    n_stft_channels = istft_params["n_fft"] + 2
    s_stft_pt = torch.randn(batch_size, n_stft_channels, target_stft_frames)

    # Run PyTorch decode manually (skip STFT, use s_stft directly)
    with torch.no_grad():
        x_ref = pt_model.conv_pre(x_pt)
        for i in range(pt_model.num_upsamples):
            x_ref = F.leaky_relu(x_ref, pt_model.lrelu_slope)
            x_ref = pt_model.ups[i](x_ref)

            if i == pt_model.num_upsamples - 1:
                x_ref = pt_model.reflection_pad(x_ref)

            si = pt_model.source_downs[i](s_stft_pt)
            si = pt_model.source_resblocks[i](si)
            x_ref = x_ref + si

            xs = None
            for j in range(pt_model.num_kernels):
                if xs is None:
                    xs = pt_model.resblocks[i * pt_model.num_kernels + j](x_ref)
                else:
                    xs += pt_model.resblocks[i * pt_model.num_kernels + j](x_ref)
            x_ref = xs / pt_model.num_kernels

        x_ref = F.leaky_relu(x_ref)
        y_pt = pt_model.conv_post(x_ref)

    # Extract state dict and create TTNN model
    state_dict = extract_causal_hift_state_dict(pt_model)

    tt_model = TtCausalHiFTGenerator(
        device,
        state_dict,
        prefix="",
        in_channels=in_channels,
        base_channels=base_channels,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        source_resblock_kernel_sizes=source_resblock_kernel_sizes,
        source_resblock_dilation_sizes=source_resblock_dilation_sizes,
        istft_params=istft_params,
        lrelu_slope=0.1,
        conv_pre_look_right=conv_pre_look_right,
    )

    # Prepare TTNN inputs
    x_tt = x_pt.transpose(1, 2).unsqueeze(1)
    x_tt = ttnn.from_torch(x_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    s_stft_tt = s_stft_pt.transpose(1, 2).unsqueeze(1)
    s_stft_tt = ttnn.from_torch(s_stft_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = tt_model.decode(x_tt, s_stft_tt)

    y_tt_cpu = ttnn.to_torch(y_tt).squeeze(1).transpose(1, 2)

    min_len = min(y_pt.shape[2], y_tt_cpu.shape[2])
    y_pt_trunc = y_pt[:, :, :min_len]
    y_tt_trunc = y_tt_cpu[:, :, :min_len]

    passing, pcc_msg = comp_pcc(y_pt_trunc, y_tt_trunc, 0.97)
    print(f"CausalHiFTGenerator decode PCC: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"
