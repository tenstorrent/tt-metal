# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) XTTS-v2 speaker-encoder mel frontend (``torch_spec``).

Coqui builds this as ``nn.Sequential(PreEmphasis(0.97), torchaudio.MelSpectrogram(
sample_rate=16000, n_fft=512, win_length=400, hop_length=160, hamming, n_mels=64))``.
torchaudio isn't installed here, so we reconstruct it with ``torch.stft`` and the
frontend buffers cached in the checkpoint (``speaker_encoder.torch_spec.*``): the
preemphasis filter ``[-0.97, 1]``, the hamming window ``[400]``, and the HTK mel
filterbank ``[257, 64]``. Output is the linear (power) mel ``[B, 64, T]``; the log
+ instance-norm live in the encoder body (already ported).
"""

import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 64
POWER = 2.0
PREEMPH = 0.97

_PREFIX = "hifigan_decoder.speaker_encoder.torch_spec."


class MelFrontend(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.register_buffer("preemph_filter", state_dict[_PREFIX + "0.filter"])  # [1, 1, 2] = [-0.97, 1]
        self.register_buffer("window", state_dict[_PREFIX + "1.spectrogram.window"])  # [400] hamming
        self.register_buffer("mel_fb", state_dict[_PREFIX + "1.mel_scale.fb"])  # [257, 64]

    def forward(self, waveform):  # [B, samples]
        x = waveform.unsqueeze(1)  # [B, 1, samples]
        x = F.pad(x, (1, 0), "reflect")
        x = F.conv1d(x, self.preemph_filter).squeeze(1)  # [B, samples]

        spec = torch.stft(
            x,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # [B, 257, T]
        power = spec.abs().pow(POWER)  # [B, 257, T]
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)  # [B, 64, T]
        return mel


def build_reference_mel_frontend(state_dict):
    return MelFrontend(state_dict).eval()
