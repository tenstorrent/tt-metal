# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference for pyannote/segmentation-3.0, the speaker segmentation model of
pyannote/speaker-diarization-3.1: SincNet, 4-layer BiLSTM, two linear layers, 7 powerset classes.

Mirrors pyannote.audio's PyanNet without importing pyannote.audio or asteroid-filterbanks."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.pyannote.reference.wespeaker import _pickle_module, _weights_path

HF_REPO_ID = "pyannote/segmentation-3.0"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 10.0
STEP_SECONDS = 1.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
STEP_SAMPLES = int(STEP_SECONDS * SAMPLE_RATE)
NUM_SPEAKERS = 3
# Output frames: SincNet stride 10, then three max-pools of 3 (270 samples per frame).
FRAME_STEP = 0.016875
FRAME_DURATION = 0.0619375

# Powerset class -> active speakers: none, each single speaker, each pair.
POWERSET = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32
)


class SincFilters(nn.Module):
    """asteroid-filterbanks ParamSincFB(80, 251, stride=10): 40 cosine and 40 sine band-pass filters."""

    def __init__(self, n_filters=80, kernel_size=251, stride=10, min_low_hz=50.0, min_band_hz=50.0):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.min_low_hz, self.min_band_hz = min_low_hz, min_band_hz
        half = kernel_size // 2
        self.low_hz_ = nn.Parameter(torch.zeros(n_filters // 2, 1))
        self.band_hz_ = nn.Parameter(torch.zeros(n_filters // 2, 1))
        self.register_buffer("window_", torch.from_numpy(np.hamming(kernel_size)[:half]).float())
        self.register_buffer("n_", 2 * math.pi * torch.arange(-half, 0.0).view(1, -1) / SAMPLE_RATE)

    def filters(self) -> torch.Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, SAMPLE_RATE / 2)
        band = (high - low)[:, 0]
        ft_low, ft_high = low @ self.n_, high @ self.n_
        cos_left = (torch.sin(ft_high) - torch.sin(ft_low)) / (self.n_ / 2) * self.window_
        sin_left = (torch.cos(ft_low) - torch.cos(ft_high)) / (self.n_ / 2) * self.window_
        cos = torch.cat([cos_left, 2 * band.view(-1, 1), torch.flip(cos_left, dims=[1])], dim=1)
        sin = torch.cat([sin_left, torch.zeros_like(band.view(-1, 1)), -torch.flip(sin_left, dims=[1])], dim=1)
        filters = torch.cat([cos, sin], dim=0) / (2 * torch.cat([band, band])[:, None])
        return filters.unsqueeze(1)

    def forward(self, x):
        return F.conv1d(x, self.filters(), stride=self.stride)


class SincNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
        self.sinc = SincFilters()
        self.conv1d = nn.ModuleList([nn.Conv1d(80, 60, 5), nn.Conv1d(60, 60, 5)])
        self.norm1d = nn.ModuleList([nn.InstanceNorm1d(c, affine=True) for c in (80, 60, 60)])

    def forward(self, waveforms):
        x = self.wav_norm1d(waveforms)
        for i, norm in enumerate(self.norm1d):
            x = torch.abs(self.sinc(x)) if i == 0 else self.conv1d[i - 1](x)
            x = F.leaky_relu(norm(F.max_pool1d(x, 3, stride=3)))
        return x


class PyanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sincnet = SincNet()
        self.lstm = nn.LSTM(60, 128, num_layers=4, bidirectional=True, batch_first=True)
        self.linear = nn.ModuleList([nn.Linear(256, 128), nn.Linear(128, 128)])
        self.classifier = nn.Linear(128, len(POWERSET))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """waveforms [batch, 1, samples] -> powerset log-probabilities [batch, frames, 7]."""
        x, _ = self.lstm(self.sincnet(waveforms).transpose(1, 2))
        for linear in self.linear:
            x = F.leaky_relu(linear(x))
        return F.log_softmax(self.classifier(x), dim=-1)


def to_multilabel(log_probs: torch.Tensor) -> torch.Tensor:
    """Powerset log-probabilities [..., 7] -> binary speaker activity [..., 3] of the most likely class."""
    return POWERSET.to(log_probs.device)[log_probs.argmax(dim=-1)]


def load_segmentation() -> PyanNet:
    """PyanNet with the pyannote/segmentation-3.0 weights, in eval mode."""
    path = _weights_path(HF_REPO_ID)
    state = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pickle_module)["state_dict"]
    renamed = {}
    for key, value in state.items():
        key = key.replace("sincnet.conv1d.0.filterbank.", "sincnet.sinc.")
        if key.startswith("sincnet.conv1d."):
            index = int(key.split(".")[2])
            key = key.replace(f"sincnet.conv1d.{index}.", f"sincnet.conv1d.{index - 1}.")
        renamed[key] = value
    model = PyanNet()
    model.load_state_dict(renamed)
    return model.eval()


def window_starts(num_samples: int) -> list[int]:
    """Start samples of the 10 s windows pyannote slides over a waveform at a 1 s step; the last
    window starts where the full windows end and is zero-padded."""
    count = max(0, (num_samples - WINDOW_SAMPLES) // STEP_SAMPLES + 1) if num_samples >= WINDOW_SAMPLES else 0
    starts = [i * STEP_SAMPLES for i in range(count)]
    if num_samples < WINDOW_SAMPLES or (num_samples - WINDOW_SAMPLES) % STEP_SAMPLES > 0:
        starts.append(count * STEP_SAMPLES)
    return starts


def windows(waveform: torch.Tensor, starts_before: int | None = None) -> tuple[torch.Tensor, list[int]]:
    """waveform [samples] -> 10 s windows [num_windows, samples per window] and their start samples.

    With `starts_before`, windows start every second below that sample and the waveform past it is
    context: a chunk of a longer recording followed by the next 10 s gets the windows that a run on the
    whole recording would have, so speaker counts near chunk boundaries do not drop."""
    if starts_before is None:
        starts = window_starts(waveform.shape[-1])
    else:
        starts = list(range(0, starts_before, STEP_SAMPLES))
    chunks = [
        F.pad(waveform[s : s + WINDOW_SAMPLES], (0, max(0, s + WINDOW_SAMPLES - waveform.shape[-1]))) for s in starts
    ]
    return torch.stack(chunks), starts
