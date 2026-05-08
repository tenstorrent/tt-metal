# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN Generator core vs PyTorch reference up to `conv_post`."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import ttnn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.demos.kokoro.reference import KokoroConfig
from models.demos.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.demos.kokoro.tt.ttnn_kokoro_generator import TtKokoroGeneratorCore, preprocess_generator_core


@torch.no_grad()
def _torch_generator_prepost(gen, *, x, s, har_per_stage):
    # Mirror Generator.forward, but replace source generation with provided har tensors per stage
    for i in range(gen.num_upsamples):
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x_source = gen.noise_convs[i](har_per_stage[i])
        x_source = gen.noise_res[i](x_source, s)
        x = gen.ups[i](x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
        x = x + x_source
        xs = None
        for j in range(gen.num_kernels):
            y = gen.resblocks[i * gen.num_kernels + j](x, s)
            xs = y if xs is None else xs + y
        x = xs / gen.num_kernels
    x = torch.nn.functional.leaky_relu(x)
    x = gen.conv_post(x)
    return x


def test_ttnn_generator_core_matches_torch(device):
    torch_model = load_decoder_from_huggingface(repo_id=KokoroConfig.repo_id, device="cpu", disable_complex=True)
    gen = torch_model.decoder.generator

    params = preprocess_generator_core(gen, device)
    tt = TtKokoroGeneratorCore(device, params)

    torch.manual_seed(0)
    B = 1
    T = 8
    x = torch.randn(B, 512, T, dtype=torch.float32)  # matches last Decoder block output channels
    s = torch.randn(B, 128, dtype=torch.float32)
    # Create per-stage `har` tensors whose lengths produce x_source matching x after upsample at that stage.
    har_per_stage = []
    cur_len = x.shape[-1]
    for i in range(gen.num_upsamples):
        # after upsample convtranspose
        # compute x length after gen.ups[i]
        u = gen.ups[i]
        k = u.kernel_size[0]
        stride = u.stride[0]
        padding = u.padding[0]
        outpad = u.output_padding[0]
        cur_len = (cur_len - 1) * stride - 2 * padding + k + outpad
        if i == gen.num_upsamples - 1:
            cur_len = cur_len + 1  # reflection_pad((1,0))

        nc = gen.noise_convs[i]
        nk = nc.kernel_size[0]
        ns = nc.stride[0]
        npad = nc.padding[0]
        # choose input length so conv1d output length equals cur_len:
        # out = floor((L_in + 2p - k)/s) + 1  -> pick exact:
        L_in = (cur_len - 1) * ns - 2 * npad + nk
        har_per_stage.append(torch.randn(B, gen.post_n_fft + 2, L_in, dtype=torch.float32))

    ref = _torch_generator_prepost(gen, x=x, s=s, har_per_stage=har_per_stage)

    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = tt(x_bct=x_tt, style_s=s, har_per_stage=har_per_stage)
    out = ttnn.to_torch(out_tt).to(torch.float32)

    # compare on min length
    min_len = min(ref.shape[-1], out.shape[-1])
    ref = ref[..., :min_len]
    out = out[..., :min_len]
    ok, pcc = comp_pcc(ref, out, pcc=0.70)
    assert ok, f"generator core pcc low: {pcc}"
