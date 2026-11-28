# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference for pyannote/wespeaker-voxceleb-resnet34-LM, the speaker embedding model of
pyannote/speaker-diarization-3.1: Kaldi fbank, ResNet34, weighted statistics pooling, linear.

Mirrors pyannote.audio's WeSpeakerResNet34 without importing pyannote.audio or torchaudio, which the
tt-metal environment does not ship."""

import math
import pickle
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, try_to_load_from_cache

HF_REPO_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
EMBED_DIM = 256
WINDOW_SIZE = 400  # 25 ms
WINDOW_SHIFT = 160  # 10 ms
PADDED_WINDOW_SIZE = 512
PREEMPHASIS = 0.97
LOW_FREQ = 20.0


def _mel_scale(freq):
    return 1127.0 * torch.log(1.0 + freq / 700.0)


def _mel_banks() -> torch.Tensor:
    """Kaldi triangular mel filters, [NUM_MEL_BINS, PADDED_WINDOW_SIZE // 2 + 1]."""
    num_fft_bins = PADDED_WINDOW_SIZE // 2
    fft_bin_width = SAMPLE_RATE / PADDED_WINDOW_SIZE
    mel_low = 1127.0 * math.log(1.0 + LOW_FREQ / 700.0)
    mel_high = 1127.0 * math.log(1.0 + (SAMPLE_RATE / 2) / 700.0)
    delta = (mel_high - mel_low) / (NUM_MEL_BINS + 1)
    b = torch.arange(NUM_MEL_BINS, dtype=torch.float32).unsqueeze(1)
    left, center, right = mel_low + b * delta, mel_low + (b + 1) * delta, mel_low + (b + 2) * delta
    mel = _mel_scale(fft_bin_width * torch.arange(num_fft_bins, dtype=torch.float32)).unsqueeze(0)
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    banks = torch.clamp(torch.minimum(up, down), min=0.0)
    return F.pad(banks, (0, 1))


def compute_fbank(waveforms: torch.Tensor) -> torch.Tensor:
    """waveforms [batch, samples] in [-1, 1] -> mean-centred log mel fbank [batch, frames, 80].

    torchaudio.compliance.kaldi.fbank with pyannote's settings: Hamming window, no dither, no energy,
    snip_edges, power spectrum; then centred over time per waveform."""
    x = waveforms.float() * (1 << 15)
    num_frames = 1 + (x.shape[-1] - WINDOW_SIZE) // WINDOW_SHIFT
    frames = x.unfold(-1, WINDOW_SIZE, WINDOW_SHIFT)[:, :num_frames]
    frames = frames - frames.mean(dim=-1, keepdim=True)
    previous = torch.cat([frames[..., :1], frames[..., :-1]], dim=-1)
    frames = frames - PREEMPHASIS * previous
    window = torch.hamming_window(WINDOW_SIZE, periodic=False, alpha=0.54, beta=0.46)
    frames = F.pad(frames * window, (0, PADDED_WINDOW_SIZE - WINDOW_SIZE))
    power = torch.fft.rfft(frames).abs().pow(2.0)
    fbank = torch.clamp(power @ _mel_banks().T, min=torch.finfo(torch.float32).eps).log()
    return fbank - fbank.mean(dim=1, keepdim=True)


def stats_pool(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """features [batch, dim, frames], weights [batch, speakers, frames'] -> [batch, speakers, 2 * dim].

    Weighted mean and unbiased weighted std per speaker, as pyannote's StatsPool; weights are resized to
    the feature frame count by nearest interpolation first."""
    if weights.shape[-1] != features.shape[-1]:
        weights = F.interpolate(weights, size=features.shape[-1], mode="nearest")
    out = []
    for s in range(weights.shape[1]):
        w = weights[:, s : s + 1, :]
        v1 = w.sum(dim=2) + 1e-8
        mean = torch.sum(features * w, dim=2) / v1
        dx2 = torch.square(features - mean.unsqueeze(2))
        v2 = torch.square(w).sum(dim=2)
        var = torch.sum(dx2 * w, dim=2) / (v1 - v2 / v1 + 1e-8)
        out.append(torch.cat([mean, torch.sqrt(var)], dim=1))
    return torch.stack(out, dim=1)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet34(nn.Module):
    def __init__(self, m_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, m_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        in_planes = m_channels
        layers = []
        for i, (blocks, stride) in enumerate(zip([3, 4, 6, 3], [1, 2, 2, 2])):
            planes = m_channels * 2**i
            layer = []
            for s in [stride] + [1] * (blocks - 1):
                layer.append(BasicBlock(in_planes, planes, s))
                in_planes = planes
            layers.append(nn.Sequential(*layer))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.stats_dim = (NUM_MEL_BINS // 8) * m_channels * 8
        self.seg_1 = nn.Linear(2 * self.stats_dim, EMBED_DIM)

    def trunk(self, fbank: torch.Tensor) -> torch.Tensor:
        """fbank [batch, frames, 80] -> [batch, 256 * 10, frames / 8], features ordered (channel, freq)."""
        x = fbank.permute(0, 2, 1).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x.flatten(1, 2)

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """fbank [batch, frames, 80], weights [batch, speakers, mask frames] -> [batch, speakers, 256]."""
        return self.seg_1(stats_pool(self.trunk(fbank), weights))


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    # The checkpoint also pickles pyannote.audio task metadata; only its tensors are needed.
    def find_class(self, module, name):
        if module.split(".")[0] in ("pyannote", "pytorch_lightning", "lightning_fabric"):
            return _Stub
        return super().find_class(module, name)


_pickle_module = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")


def _weights_path(repo_id: str) -> str:
    """The checkpoint from the local HF cache, else from the hub. A cached gated checkpoint then loads
    with a token that cannot read the repo, or offline."""
    path = try_to_load_from_cache(repo_id, "pytorch_model.bin")
    return path if isinstance(path, str) else hf_hub_download(repo_id, "pytorch_model.bin")


def load_wespeaker() -> ResNet34:
    """ResNet34 with the pyannote/wespeaker-voxceleb-resnet34-LM weights, in eval mode."""
    path = _weights_path(HF_REPO_ID)
    state = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pickle_module)["state_dict"]
    model = ResNet34()
    model.load_state_dict({k.removeprefix("resnet."): v for k, v in state.items() if k.startswith("resnet.")})
    return model.eval()
