from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from safetensors.torch import load_file

SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
DEFAULT_MODEL_URL = "https://huggingface.co/mert-kurttutan/rmvpe/resolve/main/rmvpe.safetensors"


def get_model_path() -> str:
    model_path = Path(__file__).parent.parent / "data/rmvpe.safetensors"
    return str(model_path)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        self.mel_basis = torch.from_numpy(
            mel(
                sr=sampling_rate,
                n_fft=n_fft,
                n_mels=n_mel_channels,
                fmin=mel_fmin,
                fmax=mel_fmax,
                htk=True,
            )
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def _get_hann_window(self, keyshift, win_length_new):
        if keyshift not in self.hann_window:
            self.hann_window[keyshift] = torch.hann_window(win_length_new)
        return self.hann_window[keyshift]

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(self.n_fft * factor)
        win_length_new = int(self.win_length * factor)
        hop_length_new = int(self.hop_length * speed)

        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self._get_hann_window(keyshift, win_length_new),
            center=center,
            return_complex=True,
        )
        magnitude = torch.abs(fft)

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            b, c, h, w = x.shape
            x_reshaped = x.view(b, c, -1).transpose(1, 2)
            residual = self.shortcut(x_reshaped).transpose(1, 2).view(b, -1, h, w)
        else:
            residual = x
        out = self.conv(x)
        return out + residual


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        if stride == (1, 2):
            out_padding = (0, 1)
        elif stride == (2, 2):
            out_padding = (1, 1)
        elif stride == (2, 1):
            out_padding = (1, 0)
        else:
            out_padding = (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            skip, x = self.layers[i](x)
            concat_tensors.append(skip)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)])
        for _ in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class RMVPE(nn.Module):
    def __init__(
        self,
        hop_length,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Sigmoid())

    def forward(self, x):
        assert x.ndim == 2, "Input audio should be a 2D tensor of shape (num_batches, audio_length)"
        mel = self.mel(x)
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.unet(mel)
        if n_pad > 0:
            x = x[:, :, :-n_pad, :]
        x = self.cnn(x).transpose(1, 2).flatten(-2)
        return self.fc(x)


def to_local_average_cents_old(salience, thred=0.5):
    batch_size, n_features, n_bins = salience.shape
    salience = salience.reshape(batch_size * n_features, n_bins)
    if not hasattr(to_local_average_cents, "cents_mapping"):
        to_local_average_cents.cents_mapping = torch.linspace(0, 7180, 360) + 1997.3794084376191

    average_cents = []
    for i in range(salience.shape[0]):
        salience_i = salience[i, :]
        center_i = int(torch.argmax(salience_i).item())
        start = max(0, center_i - 4)
        end = min(len(salience_i), center_i + 5)
        salience_window = salience_i[start:end]
        cents_window = to_local_average_cents.cents_mapping[start:end]
        product_sum = torch.sum(salience_window * cents_window)
        weight_sum = torch.sum(salience_window)
        average_cents.append(product_sum / weight_sum if torch.max(salience_window) > thred else 0)

    return torch.stack(average_cents).reshape(batch_size, n_features)


def to_local_average_cents(salience, thred=0.5):
    batch_size, n_features, n_bins = salience.shape
    salience = salience.reshape(batch_size * n_features, n_bins)

    average_cents = []
    max_salience = torch.max(salience, dim=1)
    center_i_tensor = max_salience.indices
    max_cents_tensor = max_salience.values
    salience_window_tensor = torch.zeros((batch_size * n_features, 9), device=salience.device)
    cents_window_tensor = torch.zeros((batch_size * n_features, 9), device=salience.device)
    start_tensor = torch.clamp(center_i_tensor - 4, min=0)
    end_tensor = torch.clamp(center_i_tensor + 5, max=n_bins)
    index_tensor = torch.arange(9, device=salience.device).unsqueeze(0) + start_tensor.unsqueeze(1)
    end_mask = index_tensor < end_tensor.unsqueeze(1)
    safe_index_tensor = index_tensor.clamp(max=n_bins - 1)
    salience_window_tensor = torch.gather(salience, 1, safe_index_tensor) * end_mask
    cents_window_tensor = 1997.3794084376191 + (7180 / 359) * index_tensor

    salience_sum = torch.sum(salience_window_tensor, dim=1)
    product_sum = torch.sum(salience_window_tensor * cents_window_tensor, dim=1)
    average_cents_tensor = torch.where(
        max_cents_tensor > thred, product_sum / salience_sum, torch.tensor(0.0, device=salience.device)
    )
    return average_cents_tensor.reshape(batch_size, n_features)


class RMVPEPitchAlgorithm:
    def __init__(
        self, sample_rate: int = SAMPLE_RATE, hop_size: int = 160, fmin: float = MEL_FMIN, fmax: float = MEL_FMAX
    ):
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

        self.model_hop_length = 160
        model_path = get_model_path()
        self.load_model(model_path)

    def load_model(self, model_path):
        model = RMVPE(self.model_hop_length, 4, 1, (2, 2))
        model_path = Path(model_path)
        if model_path.suffix == ".safetensors":
            state_dict = load_file(str(model_path), device="cpu")
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.to(torch.float32)

        if self.sample_rate != SAMPLE_RATE:
            from scipy.signal import resample

            target_length = int(len(audio) * SAMPLE_RATE / self.sample_rate)
            audio = resample(audio.cpu().numpy(), target_length, axis=1)
            audio = torch.from_numpy(audio).float().contiguous()

        return audio

    def _extract_raw_pitch_and_periodicity(
        self, audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_processed = self._preprocess_audio(audio)

        with torch.no_grad():
            pitch_pred = self.model(audio_processed)

        cents = to_local_average_cents(pitch_pred, thred=0.5)
        f0 = torch.where(cents > 0, 10 * (2 ** (cents / 1200)), torch.tensor(0.0, device=cents.device))
        periodicity = torch.max(pitch_pred, dim=2).values
        return f0, periodicity

    def extract_pitch(self, audio: torch.Tensor) -> torch.Tensor:
        target_length = (audio.shape[1] + self.hop_size - 1) // self.hop_size
        pitch, periodicity = self._extract_raw_pitch_and_periodicity(audio)
        pitch, _ = self._sanity_check(pitch, periodicity)
        if pitch.shape[1] >= target_length:
            pitch = pitch[:, :target_length]
        else:
            pitch = F.pad(pitch, (0, target_length - pitch.shape[1]), mode="reflect")
        return pitch

    def _sanity_check(self, pitch: torch.Tensor, periodicity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        voiced = periodicity > 0
        pitch[~voiced] = 0.0
        pitch[voiced] = torch.clamp(pitch[voiced], self.fmin, self.fmax)

        periodicity = torch.clamp(periodicity, 0.0, 1.0)
        return pitch, periodicity
