from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from safetensors.torch import load_file

import ttnn
from models.demos.rvc.tt_impl.batchnorm2d import BatchNorm2d
from models.demos.rvc.tt_impl.conv2d import Conv2d
from models.demos.rvc.tt_impl.convtranspose2d import ConvTranspose2d
from models.demos.rvc.tt_impl.gru import TorchWrappedGRU as GRU
from models.demos.rvc.tt_impl.linear import Linear

SAMPLE_RATE = 16000
N_CLASS = 360
N_MELS = 128
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 1024
DEFAULT_MODEL_URL = "https://huggingface.co/mert-kurttutan/rmvpe/resolve/main/rmvpe.safetensors"


def check_basic_stats(name: str, x: torch.Tensor) -> None:
    x = x.detach()
    print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"{name}: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}, std={x.std().item()}")
    print(f"{name}: absmax={x.abs().max().item()}")
    print(
        f"{name}: isfinite={torch.isfinite(x).all().item()}, "
        f"has_nan={torch.isnan(x).any().item()}, has_inf={torch.isinf(x).any().item()}"
    )


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


class ConvBlockRes:
    def __init__(self, device, in_channels: int, out_channels: int, momentum: float = 0.01):
        self.device = device
        self.conv1 = Conv2d(
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = BatchNorm2d(device=device, num_features=out_channels, momentum=momentum)
        self.conv2 = Conv2d(
            device=device,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = BatchNorm2d(device=device, num_features=out_channels, momentum=momentum)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Linear(device=device, in_features=in_channels, out_features=out_channels)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        self.conv1.load_state_dict(state_dict, "conv.0", module_prefix)
        self.bn1.load_state_dict(state_dict, "conv.1", module_prefix)
        self.conv2.load_state_dict(state_dict, "conv.3", module_prefix)
        self.bn2.load_state_dict(state_dict, "conv.4", module_prefix)
        if self.shortcut is not None:
            self.shortcut.load_state_dict(state_dict, "shortcut", module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # residual = self.shortcut(x) if self.shortcut is not None else x
        if self.shortcut is not None:
            b, h, w, c = x.shape
            # x_reshaped = x.view(b, h * w, c)
            x_reshaped = ttnn.reshape(x, (b, h * w, c))
            x_reshaped = ttnn.to_memory_config(x_reshaped, ttnn.DRAM_MEMORY_CONFIG)
            residual = self.shortcut(x_reshaped)
            residual = ttnn.reshape(residual, (b, h, w, -1))
        else:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = ttnn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = ttnn.relu(out)
        return ttnn.add(out, residual)


class ResEncoderBlock:
    def __init__(
        self, device, in_channels: int, out_channels: int, kernel_size, n_blocks: int = 1, momentum: float = 0.01
    ):
        self.device = device
        self.n_blocks = n_blocks
        self.conv = [ConvBlockRes(device, in_channels, out_channels, momentum)]
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(device, out_channels, out_channels, momentum))
        self.kernel_size = kernel_size

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        for i, block in enumerate(self.conv):
            block.load_state_dict(state_dict, f"{module_prefix}conv.{i}.")

    def __call__(self, x: ttnn.Tensor):
        for i, block in enumerate(self.conv):
            x = block(x)
        if self.kernel_size is not None:
            batch_size, height, width, channels = x.shape
            x_i = ttnn.reshape(x, (1, 1, batch_size * height * width, channels))
            x_o = ttnn.avg_pool2d(
                input_tensor=x_i,
                batch_size=batch_size,
                input_h=height,
                input_w=width,
                channels=channels,
                kernel_size=list(self.kernel_size),
                stride=list(self.kernel_size),
                padding=[0, 0],
                ceil_mode=False,
                count_include_pad=True,
                divisor_override=None,
                dtype=ttnn.bfloat16,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            out_height = height // self.kernel_size[0]
            out_width = width // self.kernel_size[1]
            x_o = ttnn.reshape(x_o, (batch_size, out_height, out_width, channels))
            return x, x_o

        return x


class ResDecoderBlock:
    def __init__(self, device, in_channels: int, out_channels: int, stride, n_blocks: int = 1, momentum: float = 0.01):
        self.device = device
        if stride == (1, 2):
            out_padding = (0, 1)
        elif stride == (2, 2):
            out_padding = (1, 1)
        elif stride == (2, 1):
            out_padding = (1, 0)
        else:
            out_padding = (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = ConvTranspose2d(
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            output_padding=out_padding,
            bias=False,
        )
        self.bn1 = BatchNorm2d(device=device, num_features=out_channels, momentum=momentum)
        self.conv2 = [ConvBlockRes(device, out_channels * 2, out_channels, momentum)]
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(device, out_channels, out_channels, momentum))

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        self.conv1.load_state_dict(state_dict, "conv1.0", module_prefix)
        self.bn1.load_state_dict(state_dict, "conv1.1", module_prefix)
        for i, block in enumerate(self.conv2):
            block.load_state_dict(state_dict, f"{module_prefix}conv2.{i}.")

    def __call__(self, x: ttnn.Tensor, concat_tensor: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.conv1(x)
        x = self.bn1(x)
        x = ttnn.relu(x)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        concat_tensor = ttnn.to_layout(concat_tensor, ttnn.TILE_LAYOUT)
        x = ttnn.concat([x, concat_tensor], dim=3)
        for block in self.conv2:
            x = block(x)
        return x


class Encoder:
    def __init__(
        self,
        device,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size,
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ):
        self.device = device
        self.n_encoders = n_encoders
        self.bn = BatchNorm2d(device=device, num_features=in_channels, momentum=momentum)
        self.layers = []
        for _ in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(device, in_channels, out_channels, kernel_size, n_blocks, momentum=momentum)
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        self.bn.load_state_dict(state_dict, "bn", module_prefix)
        for i, layer in enumerate(self.layers):
            layer.load_state_dict(state_dict, f"{module_prefix}layers.{i}.")

    def __call__(self, x: ttnn.Tensor):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            concat_tensors.append(skip)
        return x, concat_tensors


class Intermediate:
    def __init__(
        self, device, in_channels: int, out_channels: int, n_inters: int, n_blocks: int, momentum: float = 0.01
    ):
        self.n_inters = n_inters
        self.layers = [ResEncoderBlock(device, in_channels, out_channels, None, n_blocks, momentum)]
        for _ in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(device, out_channels, out_channels, None, n_blocks, momentum))

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        for i, layer in enumerate(self.layers):
            layer.load_state_dict(state_dict, f"{module_prefix}layers.{i}.")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class Decoder:
    def __init__(self, device, in_channels: int, n_decoders: int, stride, n_blocks: int, momentum: float = 0.01):
        self.layers = []
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(device, in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        for i, layer in enumerate(self.layers):
            layer.load_state_dict(state_dict, f"{module_prefix}layers.{i}.")

    def __call__(self, x: ttnn.Tensor, concat_tensors: list[ttnn.Tensor]) -> ttnn.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet:
    def __init__(
        self,
        device,
        kernel_size,
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        self.device = device
        self.encoder = Encoder(device, in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            device, self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(device, self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        prefix = "" if module_prefix is None else module_prefix
        self.encoder.load_state_dict(state_dict, f"{prefix}encoder.")
        self.intermediate.load_state_dict(state_dict, f"{prefix}intermediate.")
        self.decoder.load_state_dict(state_dict, f"{prefix}decoder.")

    def __call__(self, x: ttnn.Tensor | torch.Tensor) -> ttnn.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU:
    def __init__(self, device, input_features, hidden_features, num_layers):
        self.gru = GRU(
            device=device,
            input_size=input_features,
            hidden_size=hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        prefix = "" if module_prefix is None else module_prefix
        self.gru.load_state_dict(state_dict, key="gru", module_prefix=prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        out, _ = self.gru(x)
        return out


class BiGRUTorch(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class RMVPE:
    def __init__(
        self,
        device: ttnn.Device,
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
        self.device = device
        if self.device.get_num_devices() > 1:
            self.input_mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=0)
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        else:
            self.input_mesh_mapper = None
            self.output_mesh_composer = None
        self.n_gru = n_gru
        self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)
        self.unet = DeepUnet(device, kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = Conv2d(
            device=device,
            in_channels=en_out_channels,
            out_channels=3,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.mel.eval()
        if n_gru:
            self.bigru = BiGRU(device, 3 * N_MELS, 256, n_gru)
            self.fc_linear = Linear(device=device, in_features=512, out_features=N_CLASS)
        else:
            self.bigru = None
            self.fc_linear = Linear(device=device, in_features=3 * N_MELS, out_features=N_CLASS)

    def __call__(self, x):
        assert x.ndim == 2, "Input audio should be a 2D tensor of shape (num_batches, audio_length)"
        mel = self.mel(x)
        n_frames = mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, n_pad), mode="constant")
        mel = mel.transpose(-1, -2).unsqueeze(1)
        mel_tt = ttnn.from_torch(
            mel.permute(0, 2, 3, 1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=self.input_mesh_mapper,
        )
        x = self.unet(mel_tt)
        if n_pad > 0:
            x = x[:, :-n_pad, :, :]
        x = self.cnn(x)
        x = ttnn.permute(x, (0, 1, 3, 2))
        x = ttnn.reshape(x, (x.shape[0], x.shape[1], -1))
        if self.bigru is not None:
            x = self.bigru(x)
        out = self.fc_linear(x)
        out = ttnn.sigmoid(out)
        return out

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        self.unet.load_state_dict(state_dict, "unet.")
        self.cnn.load_state_dict(state_dict, "cnn")
        if self.bigru is not None:
            self.bigru.load_state_dict(state_dict, "fc.0.")
            self.fc_linear.load_state_dict(state_dict, key="1", module_prefix="fc.")
        else:
            self.fc_linear.load_state_dict(state_dict, key="0", module_prefix="fc.")


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
        self,
        device: ttnn.Device,
        sample_rate: int = SAMPLE_RATE,
        hop_size: int = 160,
        fmin: float = MEL_FMIN,
        fmax: float = MEL_FMAX,
    ):
        if fmin >= fmax:
            raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if hop_size <= 0:
            raise ValueError(f"Hop size must be positive, got {hop_size}")

        self.device = device
        if self.device.get_num_devices() > 1:
            self.input_mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=0)
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        else:
            self.input_mesh_mapper = None
            self.output_mesh_composer = None
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax

        self.model_hop_length = 160
        model_path = get_model_path()
        self.load_model(model_path)

    def load_model(self, model_path):
        model = RMVPE(self.device, self.model_hop_length, 4, 1, (2, 2))
        model_path = Path(model_path)
        if model_path.suffix == ".safetensors":
            state_dict = load_file(str(model_path), device="cpu")
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        model.load_state_dict(state_dict, strict=True)
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

        pitch_pred_torch = ttnn.to_torch(pitch_pred, mesh_composer=self.output_mesh_composer).to(torch.float32)

        cents = to_local_average_cents(pitch_pred_torch, thred=0.5)

        f0 = torch.where(cents > 0, 10 * (2 ** (cents / 1200)), torch.tensor(0.0, device=cents.device))
        periodicity = torch.max(pitch_pred_torch, dim=2).values
        return f0, periodicity

    def extract_pitch(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
