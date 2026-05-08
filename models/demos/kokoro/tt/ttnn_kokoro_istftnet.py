# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro ISTFTNet vocoder bring-up:
- Decoder front-end on TTNN (`TtKokoroDecoderFront`)
- Generator core on TTNN (`TtKokoroGeneratorCore`)
- Harmonic source generation + STFT/iSTFT on host (temporary)

This provides an end-to-end `Decoder` equivalent producing waveform audio, with
only the STFT/iSTFT + hn-nsf source still on host.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.kokoro.reference.kokoro_istftnet import CustomSTFT
from models.demos.kokoro.tt.ttnn_kokoro_decoder import (
    DecoderFrontParams,
    TtKokoroDecoderFront,
    preprocess_decoder_front,
)
from models.demos.kokoro.tt.ttnn_kokoro_generator import (
    GeneratorCoreParams,
    TtKokoroGeneratorCore,
    preprocess_generator_core,
)


@dataclass(frozen=True)
class IstftNetVocoderParams:
    decoder_front: DecoderFrontParams
    generator_core: GeneratorCoreParams
    # host-side helpers
    post_n_fft: int
    hop_size: int
    upsample_scale: int


def preprocess_istftnet_vocoder(
    torch_decoder, device: ttnn.Device, *, weights_dtype=ttnn.bfloat16
) -> IstftNetVocoderParams:
    dec_front = preprocess_decoder_front(torch_decoder, device, weights_dtype=weights_dtype)
    gen_core = preprocess_generator_core(torch_decoder.generator, device, weights_dtype=weights_dtype)
    g = torch_decoder.generator
    upsample_scale = int(g.f0_upsamp.scale_factor)  # prod(upsample_rates)*hop
    return IstftNetVocoderParams(
        decoder_front=dec_front,
        generator_core=gen_core,
        post_n_fft=int(g.post_n_fft),
        hop_size=int(g.stft.hop_length),
        upsample_scale=upsample_scale,
    )


def _build_har_per_stage(torch_generator, *, f0_curve_bt: torch.Tensor, x_len: int) -> list[torch.Tensor]:
    """
    Host: mimic `Generator.forward` no_grad section to produce per-stage `har` tensors
    whose time lengths match each stage's `noise_convs[i]` output.
    """
    with torch.no_grad():
        f0_up = torch_generator.f0_upsamp(f0_curve_bt[:, None]).transpose(1, 2)  # [B, T_up, 1]
        har_source, _, _uv = torch_generator.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)  # [B, T_up]
        har_spec, har_phase = torch_generator.stft.transform(har_source)  # [B, F, T_stft]
        har = torch.cat([har_spec, har_phase], dim=1)  # [B, post_n_fft+2, T_stft]

    har_per_stage: list[torch.Tensor] = []
    cur_len = int(x_len)
    for i in range(torch_generator.num_upsamples):
        # After upsample convtranspose (matches reference Generator.ups)
        u = torch_generator.ups[i]
        k = int(u.kernel_size[0])
        stride = int(u.stride[0])
        padding = int(u.padding[0])
        outpad = int(u.output_padding[0])
        cur_len = (cur_len - 1) * stride - 2 * padding + k + outpad
        if i == torch_generator.num_upsamples - 1:
            cur_len = cur_len + 1  # reflection_pad((1,0))

        nc = torch_generator.noise_convs[i]
        nk = int(nc.kernel_size[0])
        ns = int(nc.stride[0])
        npad = int(nc.padding[0])
        # pick L_in so conv1d output len == cur_len
        L_in = (cur_len - 1) * ns - 2 * npad + nk

        if har.shape[-1] >= L_in:
            har_i = har[..., :L_in]
        else:
            pad = L_in - har.shape[-1]
            har_i = torch.nn.functional.pad(har, (0, pad))
        har_per_stage.append(har_i)

    return har_per_stage


class TtKokoroIstftNetVocoder:
    def __init__(self, device: ttnn.Device, *, torch_decoder, params: IstftNetVocoderParams):
        self.device = device
        self.params = params
        self.tt_decoder_front = TtKokoroDecoderFront(device, params.decoder_front)
        self.tt_generator_core = TtKokoroGeneratorCore(device, params.generator_core)
        # host-side: iSTFT inverse
        self._stft = CustomSTFT(
            filter_length=params.post_n_fft, hop_length=params.hop_size, win_length=params.post_n_fft
        ).eval()
        # keep torch modules for host-only helper computations
        self._torch_generator = torch_decoder.generator

    @torch.no_grad()
    def __call__(
        self, *, asr: torch.Tensor, f0_pred: torch.Tensor, n_pred: torch.Tensor, ref_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs are PyTorch (CPU) tensors matching reference `Decoder.forward`.
        Returns waveform audio on CPU.
        """
        # Decoder front wants asr on device; keep bfloat16 on device
        asr_tt = ttnn.from_torch(asr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        x_feat_bct = self.tt_decoder_front(asr_bct=asr_tt, f0_pred=f0_pred, n_pred=n_pred, style_s=ref_s[:, :128])

        # host: build harmonic features (temporary)
        har_per_stage = _build_har_per_stage(
            self._torch_generator, f0_curve_bt=f0_pred, x_len=int(x_feat_bct.shape[-1])
        )

        x_logits_bct = self.tt_generator_core(x_bct=x_feat_bct, style_s=ref_s[:, :128], har_per_stage=har_per_stage)
        x_logits = ttnn.to_torch(x_logits_bct).to(torch.float32)

        # host: final spec/phase + iSTFT inverse
        freq_bins = self.params.post_n_fft // 2 + 1
        spec = torch.exp(x_logits[:, :freq_bins, :])
        phase = torch.sin(x_logits[:, freq_bins:, :])
        audio = self._stft.inverse(spec, phase).squeeze(1)  # [B, T]
        return audio
