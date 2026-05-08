# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN vocoder (decoder+generator core logits) vs torch reference.

Waveform-level PCC is extremely sensitive to tiny phase changes; instead we validate
the pre-iSTFT logits tensor (output of `Generator.conv_post`).
"""

from __future__ import annotations

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.kokoro.reference import KokoroConfig
from models.demos.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.demos.kokoro.tt.ttnn_kokoro_istftnet import TtKokoroIstftNetVocoder, preprocess_istftnet_vocoder


@torch.no_grad()
def _torch_decoder_front(decoder, *, asr, f0_pred, n_pred, s):
    F0 = decoder.F0_conv(f0_pred.unsqueeze(1))
    N = decoder.N_conv(n_pred.unsqueeze(1))
    x = torch.cat([asr, F0, N], axis=1)
    x = decoder.encode(x, s)
    asr_res = decoder.asr_res(asr)
    res = True
    for block in decoder.decode:
        if res:
            x = torch.cat([x, asr_res, F0, N], axis=1)
        x = block(x, s)
        if block.upsample_type != "none":
            res = False
    return x, F0


@torch.no_grad()
def _torch_generator_core_logits(gen, *, x, s, f0_curve, har_per_stage):
    # use provided har tensors (post STFT) and run generator up to conv_post
    har = har_per_stage
    for i in range(gen.num_upsamples):
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
        x_source = gen.noise_convs[i](har[i])
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


def test_ttnn_istftnet_vocoder_matches_torch(device):
    torch_model = load_decoder_from_huggingface(repo_id=KokoroConfig.repo_id, device="cpu", disable_complex=True)
    decoder = torch_model.decoder

    params = preprocess_istftnet_vocoder(decoder, device)
    tt = TtKokoroIstftNetVocoder(device, torch_decoder=decoder, params=params)

    torch.manual_seed(0)
    B = 1
    T_asr = 32
    T_f0 = 64
    asr = torch.randn(B, 512, T_asr, dtype=torch.float32)
    f0 = torch.randn(B, T_f0, dtype=torch.float32).abs() * 200.0  # positive-ish Hz curve
    n = torch.randn(B, T_f0, dtype=torch.float32)
    s = torch.randn(B, 256, dtype=torch.float32)

    # Torch reference logits (Generator conv_post output)
    x_feat, f0_curve = _torch_decoder_front(decoder, asr=asr, f0_pred=f0, n_pred=n, s=s[:, :128])
    # build har per stage from torch generator internals
    from models.demos.kokoro.tt.ttnn_kokoro_istftnet import _build_har_per_stage

    har_per_stage = _build_har_per_stage(decoder.generator, f0_curve_bt=f0_curve.squeeze(1), x_len=x_feat.shape[-1])
    ref_logits = _torch_generator_core_logits(
        decoder.generator, x=x_feat, s=s[:, :128], f0_curve=f0, har_per_stage=har_per_stage
    )

    # TT path (end-to-end returns audio, but we want logits; grab from internals)
    asr_tt = ttnn.from_torch(asr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_feat_tt = tt.tt_decoder_front(asr_bct=asr_tt, f0_pred=f0, n_pred=n, style_s=s[:, :128])
    x_logits_tt = tt.tt_generator_core(x_bct=x_feat_tt, style_s=s[:, :128], har_per_stage=har_per_stage)
    out_logits = ttnn.to_torch(x_logits_tt).to(torch.float32)

    min_len = min(ref_logits.shape[-1], out_logits.shape[-1])
    ref_logits = ref_logits[..., :min_len]
    out_logits = out_logits[..., :min_len]
    ok, pcc = comp_pcc(ref_logits, out_logits, pcc=0.70)
    assert ok, f"vocoder logits pcc low: {pcc}"
