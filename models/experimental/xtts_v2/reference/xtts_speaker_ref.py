# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the XTTS-v2 ResNet speaker encoder (Block 2) -> d-vector.

Reimplements coqui's TTS.encoder.models.resnet.ResNetSpeakerEncoder (H/ASP) as self-contained
nn.Modules, stripped of the BaseEncoder/losses/trainer/coqpit training deps. As instantiated in
HifiDecoder: input_dim=64, proj_dim=512, layers=[3,4,6,3], num_filters=[32,64,128,256],
encoder_type="ASP", log_input=True, use_torch_spec=True.

Block boundary for the port is the ResNet CORE: logmel [1,64,T] -> d-vector [1,512]. The mel
front-end (PreEmphasis + torchaudio MelSpectrogram + log) runs on host (needs torchaudio); it is
provided here for completeness but is not the TTNN target.

Coqui inference path (Xtts.get_speaker_embedding):
    wav (sr) -> resample 16kHz -> speaker_encoder.forward(wav, l2_norm=True) -> unsqueeze(-1)
             -> speaker_embedding [1, 512, 1]

CORE forward (from logmel onward):
    x = instancenorm(logmel).unsqueeze(1)          # InstanceNorm1d(64, affine=False)
    x = conv1(x); x = relu(x); x = bn1(x)          # NB: ReLU BEFORE BatchNorm (coqui's order)
    x = layer1..layer4(x)                          # SEBasicBlock ResNet, strides (1,2,2,2)
    x = x.reshape(N, 256*8=2048, T')               # fold freq into channels
    w = attention(x)                               # ASP weights, softmax over time
    mu = sum(x*w,2); sg = sqrt((sum(x^2*w,2)-mu^2).clamp(1e-5)); x = cat([mu,sg],1)  # [N,4096]
    x = fc(x)                                       # Linear(4096 -> 512)
    if l2_norm: x = F.normalize(x, p=2, dim=1)

Run (needs repo root on PYTHONPATH):
    PYTHONPATH=<repo> python models/experimental/xtts_v2/reference/xtts_speaker_ref.py
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.xtts_v2.reference.xtts_gpt_ref import DEFAULT_CKPT, load_full_state, pcc

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "speaker")
SE_PREFIX = "hifigan_decoder.speaker_encoder."

INPUT_DIM = 64
PROJ_DIM = 512
LAYERS = [3, 4, 6, 3]
NUM_FILTERS = [32, 64, 128, 256]
AUDIO_CONFIG = {"fft_size": 512, "win_length": 400, "hop_length": 160, "sample_rate": 16000, "preemphasis": 0.97, "num_mels": 64}


# --------------------------------------------------------------------------------------
# ResNet core modules (op-for-op from coqui resnet.py; names match so the ckpt loads)
# --------------------------------------------------------------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)  # ReLU before BN (coqui's order)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SpeakerEncoderCore(nn.Module):
    """ResNetSpeakerEncoder minus the torch_spec front-end and BaseEncoder plumbing.
    Submodule names match coqui so a sliced checkpoint loads directly."""

    def __init__(self, input_dim=INPUT_DIM, proj_dim=PROJ_DIM, layers=LAYERS, num_filters=NUM_FILTERS):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.inplanes = num_filters[0]
        self.layer1 = self._make_layer(num_filters[0], layers[0])
        self.layer2 = self._make_layer(num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(num_filters[3], layers[3], stride=(2, 2))
        self.instancenorm = nn.InstanceNorm1d(input_dim)  # affine=False -> no params
        outmap_size = int(input_dim / 8)
        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.fc = nn.Linear(num_filters[3] * outmap_size * 2, proj_dim)  # ASP -> *2 (mean+std)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [SEBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, logmel, l2_norm=True):  # logmel [N, 64, T]
        x = self.instancenorm(logmel).unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1, x.size(-1))  # [N, 2048, T']
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), dim=1)  # [N, 4096]
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # [N, 512]
        if l2_norm:
            x = F.normalize(x, p=2, dim=1)
        return x


# --------------------------------------------------------------------------------------
# Host-side mel front-end (not a TTNN target; here for wav -> logmel completeness)
# --------------------------------------------------------------------------------------
class PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.register_buffer("filter", torch.FloatTensor([-coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert x.dim() == 2
        x = F.pad(x.unsqueeze(1), (1, 0), "reflect")
        return F.conv1d(x, self.filter).squeeze(1)


def build_frontend():
    import torchaudio  # lazy: only for wav -> mel

    return nn.Sequential(
        PreEmphasis(AUDIO_CONFIG["preemphasis"]),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=AUDIO_CONFIG["sample_rate"],
            n_fft=AUDIO_CONFIG["fft_size"],
            win_length=AUDIO_CONFIG["win_length"],
            hop_length=AUDIO_CONFIG["hop_length"],
            window_fn=torch.hamming_window,
            n_mels=AUDIO_CONFIG["num_mels"],
        ),
    ).eval()


@torch.no_grad()
def logmel_from_wav(wav_16k, frontend=None):  # wav [1, T] -> logmel [1, 64, T']
    fe = frontend if frontend is not None else build_frontend()
    return (fe(wav_16k) + 1e-6).log()


# --------------------------------------------------------------------------------------
# Weight loading / entry points
# --------------------------------------------------------------------------------------
def load_speaker_state(ckpt_path=DEFAULT_CKPT):
    """speaker_encoder core weights (keyed relative to the encoder), excluding the front-end."""
    full = load_full_state(ckpt_path)
    out = {}
    for k, v in full.items():
        if k.startswith(SE_PREFIX):
            rel = k[len(SE_PREFIX) :]
            if not rel.startswith("torch_spec."):  # front-end runs on host; not in the core
                out[rel] = v
    return out


def build_reference(ckpt_path=DEFAULT_CKPT):
    core = SpeakerEncoderCore()
    missing, unexpected = core.load_state_dict(load_speaker_state(ckpt_path), strict=False)
    assert not missing, f"missing core keys: {missing[:8]}"
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    core.eval()
    return core


@torch.no_grad()
def speaker_embedding(core, logmel):  # -> [1, 512, 1] (matches get_speaker_embedding)
    return core(logmel, l2_norm=True).unsqueeze(-1)


def make_synthetic_logmel(n_frames=128, seed=0):
    """Deterministic logmel [1, 64, T]. Content is irrelevant for op-for-op validation;
    InstanceNorm normalizes it anyway."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, INPUT_DIM, n_frames, generator=g)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=GOLDEN_DIR)
    ap.add_argument("--n-frames", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[ref] loading speaker encoder from {args.ckpt}")
    core = build_reference(args.ckpt)
    logmel = make_synthetic_logmel(args.n_frames)
    dvec = core(logmel, l2_norm=True)  # [1, 512]
    print(f"[ref] logmel {tuple(logmel.shape)} -> d-vector {tuple(dvec.shape)} (norm={dvec.norm().item():.4f})")

    torch.save(logmel, os.path.join(args.out, "logmel_in.pt"))
    torch.save(dvec, os.path.join(args.out, "dvector.pt"))  # unsqueeze(-1) -> [1,512,1] in the pipeline
    torch.save({"n_frames": args.n_frames, "proj_dim": PROJ_DIM}, os.path.join(args.out, "meta.pt"))
    print(f"[ref] wrote speaker goldens to {args.out}")


if __name__ == "__main__":
    main()
