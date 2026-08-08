# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hifi_decoder` of coqui/XTTS-v2 (`Xtts.hifigan_decoder`).

Reference submodule: `hifigan_decoder`, a `TTS.tts.layers.xtts.hifigan_decoder.HifiDecoder`.
Its `forward(latents, g)` (the speaker_encoder is used only by `inference`, not here):

    z = interpolate(latents.transpose(1,2), scale=ar_mel/hop, mode='linear').squeeze(1)
    z = interpolate(z,                       scale=out_sr/in_sr, mode='linear').squeeze(0)
    o = waveform_decoder(z, g=g)                                       # HiFi-GAN vocoder

Captured: latents `[1, 12, 1024]`, g `[1, 512, 1]` -> waveform `[1, 1, 13312]`.

The two linear `F.interpolate` steps operate independently per channel along the time
axis, so their composition is a single fixed linear map `M` of shape `[T_in, T_out]`
(here 12 -> 52). We extract `M` exactly by pushing an identity through the SAME
`F.interpolate` chain (a geometry constant — no trained weights, no torch reference
module) and apply it as a native ttnn matmul:

    z[c, t_out] = sum_{t_in} latents[0, t_in, c] * M[t_in, t_out]
                = (latents.transpose(1,2) @ M)

The HiFi-GAN vocoder itself reuses the graduated native `hifigan_generator` port
(`waveform_decoder`), whose forward already accepts the `[1, C, T]` activation and the
`g` d-vector.
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs.hifigan_generator import build as _build_generator


def build(device, torch_module):
    """Bind the vocoder + interpolation geometry and return a native ttnn forward."""
    m = torch_module

    gen_forward = _build_generator(device, m.waveform_decoder)

    ar_mel = float(m.ar_mel_length_compression)
    hop = float(m.output_hop_length)
    in_sr = float(m.input_sample_rate)
    out_sr = float(m.output_sample_rate)

    _m_cache: dict = {}

    def _interp_matrix(t_in):
        if t_in not in _m_cache:
            import torch
            import torch.nn.functional as F

            # Push a time-identity through the exact reference interpolation chain
            # to materialize the [t_in, t_out] linear resampling map.
            x = torch.eye(t_in).unsqueeze(0)  # [1, t_in, t_in]
            mm = F.interpolate(x, scale_factor=[ar_mel / hop], mode="linear").squeeze(1)
            if out_sr != in_sr:
                mm = F.interpolate(mm, scale_factor=[out_sr / in_sr], mode="linear").squeeze(0)
            # float32 resampling map: the interpolation matmul is the vocoder's
            # precision-sensitive front (bf16 here measurably lowered e2e PCC).
            _m_cache[t_in] = ttnn.as_tensor(
                mm.contiguous().to(torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return _m_cache[t_in]

    def forward(latents, g=None, *args, **kwargs):
        t_in = int(latents.shape[1])
        M = _interp_matrix(t_in)
        if latents.get_dtype() != ttnn.float32:
            latents = ttnn.typecast(latents, ttnn.float32)  # f32 interp for a clean vocoder input
        lt = ttnn.transpose(latents, 1, 2)  # [1, C, t_in]
        z = ttnn.matmul(lt, M)  # [1, C, t_out]
        return gen_forward(z, g=g)

    return forward


def hifi_decoder(*args, **kwargs):
    raise RuntimeError(
        "hifi_decoder requires build(device, torch_module) to bind trained weights; "
        "the bare callable has no parameters."
    )
