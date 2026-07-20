# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the XTTS-v2 ResNet speaker encoder (Block 2, "Branch B" -> d-vector).

Op-for-op mirror of coqui `TTS.encoder.models.resnet.ResNetSpeakerEncoder` as it is
instantiated inside `TTS.tts.layers.xtts.hifigan_decoder.HifiDecoder`:

    ResNetSpeakerEncoder(input_dim=64, proj_dim=512, log_input=True, use_torch_spec=True,
                         layers=[3,4,6,3], num_filters=[32,64,128,256], encoder_type="ASP",
                         audio_config={fft_size=512, win_length=400, hop_length=160,
                                       sample_rate=16000, preemphasis=0.97, num_mels=64})

Coqui inference path (`Xtts.get_speaker_embedding`):
    audio (sr) -> resample to 16 kHz -> speaker_encoder.forward(audio_16k, l2_norm=True)
               -> .unsqueeze(-1)  ->  speaker_embedding [1, 512, 1]

forward(x, l2_norm):
    x.squeeze_(1)
    x = torch_spec(x)                      # PreEmphasis(0.97) + MelSpectrogram(64 mels)
    x = (x + 1e-6).log()                   # log_input   <-- "logmel"  (TTNN block boundary)
    x = instancenorm(x).unsqueeze(1)       # InstanceNorm1d(64, affine=False) over time
    x = conv1(x); x = relu(x); x = bn1(x)  # NB: relu BEFORE bn (coqui's odd order)
    x = layer1..layer4(x)                  # SEBasicBlock ResNet, strides (1,2,2,2)
    x = x.reshape(B, -1, T')               # [B, 256*8=2048, T']
    w = attention(x)                       # ASP weights, softmax over time
    mu = sum(x*w, 2); sg = sqrt((sum(x^2*w,2)-mu^2).clamp(min=1e-5))
    x = cat([mu, sg], 1)                   # [B, 4096]
    x = fc(x)                              # Linear(4096 -> 512)
    if l2_norm: x = F.normalize(x, p=2, dim=1)

The ResNet *core* (everything from logmel onward) is a plain-torch nn.Module with no TTS /
torchaudio dependency, so it imports and runs in the tt-metal python_env. The mel front-end
(preemphasis + STFT + mel filterbank + log) needs torchaudio and is built lazily; it is used
only when computing the mel from a raw waveform (coqui venv). The TTNN test starts from the
captured `logmel` golden and only touches the core.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experimental.xtts_v2.reference.xtts_gpt_ref import DEFAULT_CKPT, load_full_state

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden", "speaker")

SE_PREFIX = "hifigan_decoder.speaker_encoder."

INPUT_DIM = 64
PROJ_DIM = 512
LAYERS = [3, 4, 6, 3]
NUM_FILTERS = [32, 64, 128, 256]
AUDIO_CONFIG = {
    "fft_size": 512,
    "win_length": 400,
    "hop_length": 160,
    "sample_rate": 16000,
    "preemphasis": 0.97,
    "num_mels": 64,
}


def load_speaker_state(ckpt_path=DEFAULT_CKPT):
    """Speaker-encoder weights with the 'hifigan_decoder.speaker_encoder.' prefix stripped."""
    full = load_full_state(ckpt_path)
    return {k[len(SE_PREFIX) :]: v.float() for k, v in full.items() if k.startswith(SE_PREFIX)}


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
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetSpeakerEncoder(nn.Module):
    """The ResNet core (logmel -> embedding). No torchaudio dependency."""

    def __init__(self):
        super().__init__()
        self.encoder_type = "ASP"
        self.input_dim = INPUT_DIM
        self.log_input = True
        self.proj_dim = PROJ_DIM

        self.conv1 = nn.Conv2d(1, NUM_FILTERS[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(NUM_FILTERS[0])

        self.inplanes = NUM_FILTERS[0]
        self.layer1 = self._make_layer(NUM_FILTERS[0], LAYERS[0])
        self.layer2 = self._make_layer(NUM_FILTERS[1], LAYERS[1], stride=(2, 2))
        self.layer3 = self._make_layer(NUM_FILTERS[2], LAYERS[2], stride=(2, 2))
        self.layer4 = self._make_layer(NUM_FILTERS[3], LAYERS[3], stride=(2, 2))

        self.instancenorm = nn.InstanceNorm1d(INPUT_DIM)

        outmap_size = int(self.input_dim / 8)  # 8
        self.attention = nn.Sequential(
            nn.Conv1d(NUM_FILTERS[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, NUM_FILTERS[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        out_dim = NUM_FILTERS[3] * outmap_size * 2  # ASP -> 4096
        self.fc = nn.Linear(out_dim, PROJ_DIM)

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

    def forward(self, logmel, l2_norm=False, return_intermediates=False):
        """logmel: [1, 64, T] (post-log mel). Returns [1, 512] (or dict of intermediates)."""
        inter = {}
        x = self.instancenorm(logmel).unsqueeze(1)  # [1,1,64,T]
        inter["instancenorm"] = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        inter["conv1"] = x

        x = self.layer1(x)
        inter["layer1"] = x
        x = self.layer2(x)
        inter["layer2"] = x
        x = self.layer3(x)
        inter["layer3"] = x
        x = self.layer4(x)
        inter["layer4"] = x

        x = x.reshape(x.size(0), -1, x.size(-1))  # [1, 2048, T'']
        inter["reshape"] = x
        w = self.attention(x)
        inter["attn_w"] = w
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)  # [1, 4096]
        inter["pool"] = x
        x = self.fc(x)
        inter["fc"] = x
        if l2_norm:
            x = F.normalize(x, p=2, dim=1)
        inter["emb"] = x
        if return_intermediates:
            return x, inter
        return x


class _PreEmphasis(nn.Module):
    def __init__(self, coefficient=0.97):
        super().__init__()
        self.register_buffer("filter", torch.FloatTensor([-coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        assert len(x.size()) == 2
        x = F.pad(x.unsqueeze(1), (1, 0), "reflect")
        return F.conv1d(x, self.filter).squeeze(1)


class SpeakerReference:
    """Full reference: raw 16 kHz waveform -> d-vector [1, 512, 1].

    The ResNet core loads with no torchaudio dependency; the mel front-end is built lazily
    (torchaudio) and is only needed by `logmel` / `speaker_embedding`.
    """

    def __init__(self, ckpt_path=DEFAULT_CKPT):
        self.core = ResNetSpeakerEncoder()
        sd = load_speaker_state(ckpt_path)
        core_sd = {k: v for k, v in sd.items() if not k.startswith("torch_spec.")}
        missing, unexpected = self.core.load_state_dict(core_sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert not missing, f"missing keys: {missing}"
        self.core.eval()
        self._torch_spec = None

    def _build_frontend(self):
        import torchaudio  # lazy: only needed for the mel front-end

        self._torch_spec = nn.Sequential(
            _PreEmphasis(AUDIO_CONFIG["preemphasis"]),
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
    def logmel(self, audio_16k):  # [1, T] -> [1, 64, T']
        if self._torch_spec is None:
            self._build_frontend()
        x = audio_16k.clone()
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        spec = self._torch_spec(x)
        return (spec + 1e-6).log()

    @torch.no_grad()
    def speaker_embedding(self, audio_16k):  # [1, T] -> [1, 512, 1]
        emb = self.core(self.logmel(audio_16k), l2_norm=True)
        return emb.unsqueeze(-1)


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


if __name__ == "__main__":
    ref = SpeakerReference()
    logmel_g = torch.load(os.path.join(GOLDEN_DIR, "logmel.pt"))
    emb_g = torch.load(os.path.join(GOLDEN_DIR, "speaker_embedding.pt"))

    # core path from the golden logmel (no torchaudio needed)
    emb_core = ref.core(logmel_g, l2_norm=True).unsqueeze(-1)
    print(f"core(logmel)  speaker_embedding {tuple(emb_core.shape)}  PCC vs coqui = {_pcc(emb_core, emb_g):.6f}")

    # full path from raw audio (needs torchaudio)
    try:
        audio = torch.load(os.path.join(GOLDEN_DIR, "audio_16k.pt"))
        logmel = ref.logmel(audio)
        emb = ref.speaker_embedding(audio)
        print(f"logmel {tuple(logmel.shape)}  PCC vs coqui = {_pcc(logmel, logmel_g):.6f}")
        print(f"full(audio)   speaker_embedding {tuple(emb.shape)}  PCC vs coqui = {_pcc(emb, emb_g):.6f}")
    except ModuleNotFoundError as e:
        print(f"(front-end skipped: {e})")
