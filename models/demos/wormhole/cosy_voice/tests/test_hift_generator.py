"""
Test for the TtHiFTGenerator decode path.

Instantiates the non-causal HiFTGenerator from the reference codebase
with random weights and compares the decode output (up to ISTFT) against
the TTNN TtHiFTGenerator implementation.

The STFT/ISTFT and f0 predictor remain on host CPU and are not tested here.
We test only the neural network portion: conv_pre -> upsamples -> resblocks -> conv_post.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../ref/CosyVoice")))
from cosyvoice.hifigan.generator import HiFTGenerator

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosy_voice.tt.vocoder.generator import TtHiFTGenerator


def extract_hift_state_dict(pt_model):
    """
    Extract fused weights from a HiFTGenerator model,
    bypassing parametrizations/weight_norm issues.
    Returns a flat dict with keys like 'conv_pre.weight', 'conv_pre.bias', etc.
    """
    sd = {}

    # conv_pre
    sd["conv_pre.weight"] = pt_model.conv_pre.weight.data.clone()
    sd["conv_pre.bias"] = pt_model.conv_pre.bias.data.clone()

    # ups (ConvTranspose1d)
    for i, up in enumerate(pt_model.ups):
        sd[f"ups.{i}.weight"] = up.weight.data.clone()
        sd[f"ups.{i}.bias"] = up.bias.data.clone()

    # source_downs (Conv1d)
    for i, sd_layer in enumerate(pt_model.source_downs):
        sd[f"source_downs.{i}.weight"] = sd_layer.weight.data.clone()
        sd[f"source_downs.{i}.bias"] = sd_layer.bias.data.clone()

    # source_resblocks
    for i, rb in enumerate(pt_model.source_resblocks):
        for j in range(len(rb.convs1)):
            sd[f"source_resblocks.{i}.convs1.{j}.weight"] = rb.convs1[j].weight.data.clone()
            sd[f"source_resblocks.{i}.convs1.{j}.bias"] = rb.convs1[j].bias.data.clone()
            sd[f"source_resblocks.{i}.convs2.{j}.weight"] = rb.convs2[j].weight.data.clone()
            sd[f"source_resblocks.{i}.convs2.{j}.bias"] = rb.convs2[j].bias.data.clone()
            sd[f"source_resblocks.{i}.activations1.{j}.alpha"] = rb.activations1[j].alpha.data.clone()
            sd[f"source_resblocks.{i}.activations2.{j}.alpha"] = rb.activations2[j].alpha.data.clone()

    # resblocks
    for i, rb in enumerate(pt_model.resblocks):
        for j in range(len(rb.convs1)):
            sd[f"resblocks.{i}.convs1.{j}.weight"] = rb.convs1[j].weight.data.clone()
            sd[f"resblocks.{i}.convs1.{j}.bias"] = rb.convs1[j].bias.data.clone()
            sd[f"resblocks.{i}.convs2.{j}.weight"] = rb.convs2[j].weight.data.clone()
            sd[f"resblocks.{i}.convs2.{j}.bias"] = rb.convs2[j].bias.data.clone()
            sd[f"resblocks.{i}.activations1.{j}.alpha"] = rb.activations1[j].alpha.data.clone()
            sd[f"resblocks.{i}.activations2.{j}.alpha"] = rb.activations2[j].alpha.data.clone()

    # conv_post
    sd["conv_post.weight"] = pt_model.conv_post.weight.data.clone()
    sd["conv_post.bias"] = pt_model.conv_post.bias.data.clone()

    return sd


@pytest.mark.parametrize("mel_length", [32])
def test_hift_generator_decode(device, mel_length):
    """
    Test the decode path (conv_pre -> upsamples+resblocks -> conv_post)
    of TtHiFTGenerator against the PyTorch HiFTGenerator reference.

    We bypass f0/source/STFT by providing pre-computed s_stft directly.
    """
    torch.manual_seed(42)
    batch_size = 1
    in_channels = 80
    base_channels = 512
    upsample_rates = [8, 8]
    upsample_kernel_sizes = [16, 16]
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    source_resblock_kernel_sizes = [7, 11]
    source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5]]
    istft_params = {"n_fft": 16, "hop_len": 4}

    # Instantiate PyTorch model (non-causal, no f0 predictor)
    pt_model = HiFTGenerator(
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
        f0_predictor=None,
    )
    pt_model.eval()

    # Create dummy mel input [B, C, L] and source STFT [B, n_fft+2, L_src]
    x_pt = torch.randn(batch_size, in_channels, mel_length)

    # Source STFT length must account for the reflection pad (+1) at the last upsample stage.
    # source_downs[0] (stride=8, k=16, pad=4): floor((L_in + 8 - 16)/8) + 1 must equal mel*8=256
    # source_downs[1] (stride=1, k=1):         L_in must equal mel*64 + 1 = 2049
    total_upsample = int(np.prod(upsample_rates))
    s_stft_length = mel_length * total_upsample + 1
    n_stft_channels = istft_params["n_fft"] + 2
    s_stft_pt = torch.randn(batch_size, n_stft_channels, s_stft_length)

    # Run PyTorch decode (only the neural network part, not ISTFT)
    with torch.no_grad():
        # Manually run the decode logic up to conv_post (skip ISTFT)
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
    state_dict = extract_hift_state_dict(pt_model)

    tt_model = TtHiFTGenerator(
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
    )

    # Prepare TTNN inputs: [B, 1, L, C]
    x_tt = x_pt.transpose(1, 2).unsqueeze(1)  # [B, 1, L, 80]
    x_tt = ttnn.from_torch(x_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    s_stft_tt = s_stft_pt.transpose(1, 2).unsqueeze(1)  # [B, 1, L_src, n_fft+2]
    s_stft_tt = ttnn.from_torch(s_stft_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run TTNN decode
    y_tt = tt_model.decode(x_tt, s_stft_tt)

    # Convert output back: [B, 1, L_out, C] -> [B, C, L_out]
    y_tt_cpu = ttnn.to_torch(y_tt)
    y_tt_flat = y_tt_cpu.squeeze(1).transpose(1, 2)

    # Truncate to match PyTorch output length (padding differences)
    min_len = min(y_pt.shape[2], y_tt_flat.shape[2])
    y_pt_trunc = y_pt[:, :, :min_len]
    y_tt_trunc = y_tt_flat[:, :, :min_len]

    # Compare with PCC
    passing, pcc_msg = comp_pcc(y_pt_trunc, y_tt_trunc, 0.98)
    print(f"HiFTGenerator decode PCC: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"
