# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from models.experimental.xtts.config import (  # noqa: F401
    SPK_FRONTEND_PREFIX as _PREFIX,
    SPK_HOP_LENGTH as HOP_LENGTH,
    SPK_N_FFT as N_FFT,
    SPK_N_MELS as N_MELS,
    SPK_POWER as POWER,
    SPK_PREEMPH as PREEMPH,
    SPK_SAMPLE_RATE as SAMPLE_RATE,
    SPK_WIN_LENGTH as WIN_LENGTH,
)


class MelFrontend(torch.nn.Module):
    def __init__(self, state_dict):
        """Register preemphasis, STFT window, and mel filterbank buffers."""
        super().__init__()
        self.register_buffer("preemph_filter", state_dict[_PREFIX + "0.filter"])
        self.register_buffer("window", state_dict[_PREFIX + "1.spectrogram.window"])
        self.register_buffer("mel_fb", state_dict[_PREFIX + "1.mel_scale.fb"])

    def forward(self, waveform):
        """Convert waveform to mel spectrogram via preemphasis and STFT."""
        x = waveform.unsqueeze(1)
        x = F.pad(x, (1, 0), "reflect")
        x = F.conv1d(x, self.preemph_filter).squeeze(1)

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
        )
        power = spec.abs().pow(POWER)
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)
        return mel


def build_reference_mel_frontend(state_dict):
    """Build an eval-mode MelFrontend from checkpoint weights."""
    return MelFrontend(state_dict).eval()
