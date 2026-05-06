# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import ttnn
from models.demos.rvc.tt_impl.batchnorm1d import BatchNorm1d
from models.demos.rvc.tt_impl.conv1d import Conv1d
from models.demos.rvc.tt_impl.linear import Linear

CENTS_PER_BIN = 20
PITCH_BINS = 360
MAX_FMAX = 2006.0
SAMPLE_RATE = 16000
WINDOW_SIZE = 1024


def viterbi(logits):
    """Sample observations using viterbi decoding."""
    if not hasattr(viterbi, "transition"):
        xx, yy = np.meshgrid(range(PITCH_BINS), range(PITCH_BINS))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        viterbi.transition = transition

    with torch.no_grad():
        probs = F.softmax(logits, dim=1)

    sequences = probs.cpu().numpy()
    bins = np.array([librosa.sequence.viterbi(sequence, viterbi.transition).astype(np.int64) for sequence in sequences])
    bins = torch.tensor(bins)
    return bins, bins_to_frequency(bins)


def bins_to_cents(bins):
    """Converts pitch bins to cents."""
    cents = CENTS_PER_BIN * bins + 1997.3794084376191
    noise = scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size())
    return cents + cents.new_tensor(noise)


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz."""
    cents = CENTS_PER_BIN * bins + 1997.3794084376191
    noise = scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size())
    cents = cents + cents.new_tensor(noise)
    return 10 * 2 ** (cents / 1200)


def argmax(logits):
    """Sample observations by taking the argmax."""
    bins = logits.argmax(dim=1)
    return bins, bins_to_frequency(bins)


def weighted_argmax(logits):
    """Sample observations using a weighted sum near the argmax."""
    bins = logits.argmax(dim=1)
    start = torch.clamp(bins - 4, min=0)
    end = torch.clamp(bins + 5, max=logits.size(1))

    for batch in range(logits.size(0)):
        for time in range(logits.size(2)):
            logits[batch, : start[batch, time], time] = -float("inf")
            logits[batch, end[batch, time] :, time] = -float("inf")

    if not hasattr(weighted_argmax, "weights"):
        weights = bins_to_cents(torch.arange(PITCH_BINS))
        weighted_argmax.weights = weights[None, :, None]

    with torch.no_grad():
        probs = torch.sigmoid(logits)

    cents = (weighted_argmax.weights * probs).sum(dim=1) / probs.sum(dim=1)
    return bins, 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins."""
    cents = 1200 * torch.log2(frequency / 10.0)
    bins = (cents - 1997.3794084376191) / CENTS_PER_BIN
    return quantize_fn(bins).int()


def preprocess(audio, sample_rate, hop_length=None, batch_size=None, pad=True):
    """Convert audio to model input frames."""
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    if pad:
        total_frames = 1 + int(audio.size(1) // hop_length)
        audio = F.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else:
        total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

    batch_size = total_frames if batch_size is None else batch_size

    for i in range(0, total_frames, batch_size):
        start = max(0, i * hop_length)
        end = min(audio.size(1), (i + batch_size - 1) * hop_length + WINDOW_SIZE)

        frames = F.unfold(
            audio[:, None, None, start:end],
            kernel_size=(1, WINDOW_SIZE),
            stride=(1, hop_length),
        )
        frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)
        frames -= frames.mean(dim=1, keepdim=True)
        frames /= torch.max(torch.tensor(1e-10), frames.std(dim=1, keepdim=True))
        yield frames


def resample(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate == SAMPLE_RATE:
        return audio

    target_length = int(audio.size(1) * SAMPLE_RATE / sample_rate)
    resampled = librosa.resample(audio.cpu().numpy(), orig_sr=sample_rate, target_sr=SAMPLE_RATE, axis=1)
    if resampled.shape[1] != target_length:
        resampled = librosa.util.fix_length(resampled, size=target_length, axis=1)
    return torch.from_numpy(resampled).to(audio.dtype)


def compute_periodicity(probabilities, bins):
    """Computes the periodicity from the network output and pitch bins."""
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
    bins_stacked = bins.reshape(-1, 1).to(torch.int64)
    periodicity = probs_stacked.gather(1, bins_stacked)
    return periodicity.reshape(probabilities.size(0), probabilities.size(2))


def filter_by_confidence(
    pitch: torch.Tensor,
    periodicity: torch.Tensor,
    threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if threshold is None:
        return pitch, periodicity

    voiced_mask = periodicity >= threshold
    filtered_pitch = torch.where(voiced_mask, pitch, torch.zeros_like(pitch))
    filtered_periodicity = torch.where(voiced_mask, periodicity, torch.zeros_like(periodicity))
    return filtered_pitch, filtered_periodicity


def postprocess(
    probabilities,
    fmin=0.0,
    fmax=MAX_FMAX,
    decoder=viterbi,
    return_periodicity=False,
    confidence_threshold: float | None = None,
):
    """Convert model output to F0 and periodicity."""
    minidx = frequency_to_bins(torch.tensor(fmin))
    maxidx = frequency_to_bins(torch.tensor(fmax), torch.ceil)

    probabilities[:, :minidx] = -float("inf")
    probabilities[:, maxidx:] = -float("inf")
    # return decoder(probabilities).transpose(1,2)
    bins, pitch = decoder(probabilities)
    if not return_periodicity:
        return pitch

    periodicity = compute_periodicity(probabilities, bins)
    pitch, periodicity = filter_by_confidence(pitch, periodicity, confidence_threshold)
    return pitch, periodicity


class ConvBlock:
    def __init__(self, device: ttnn.Device, in_channels, out_channels, kernel_size, stride, padding):
        self.device = device
        self.out_channels = out_channels
        self.conv = Conv1d(
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = BatchNorm1d(
            device=device,
            num_features=out_channels,
            eps=0.0010000000474974513,
            momentum=0.0,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        prefix = "" if module_prefix is None else module_prefix
        self.conv.load_state_dict(state_dict, "conv", prefix)
        self.batch_norm.load_state_dict(state_dict, "batch_norm", prefix)

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv(input_tensor)
        x = ttnn.relu(x)
        x = self.batch_norm(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        batch_size, input_length, _ = x.shape
        x = ttnn.unsqueeze(x, dim=1)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=batch_size,
            input_h=1,
            input_w=input_length,
            channels=self.out_channels,
            kernel_size=[1, 2],
            stride=[1, 2],
            padding=[0, 0],
            dilation=[1, 1],
            ceil_mode=False,
            dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        output_length = (input_length - 2) // 2 + 1
        out = ttnn.reshape(out, (batch_size, 1, output_length, self.out_channels))
        return ttnn.squeeze(out, dim=1)


class Crepe:
    """TTNN CREPE model definition."""

    def __init__(self, device: ttnn.Device):
        self.device = device
        if self.device.get_num_devices() > 1:
            self.input_mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=0)
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        else:
            self.input_mesh_mapper = None
            self.output_mesh_composer = None

        in_channels = [1, 128, 16, 16, 16, 32]
        out_channels = [128, 16, 16, 16, 32, 64]
        self.in_features = 256

        kernel_sizes = [512] + 5 * [64]
        strides = [4] + 5 * [1]
        paddings = [(254, 254)] + 5 * [(31, 32)]

        self.convs = [
            ConvBlock(device, in_channels[i], out_channels[i], kernel_sizes[i], strides[i], paddings[i])
            for i in range(6)
        ]
        self.classifier = Linear(device=device, in_features=self.in_features, out_features=PITCH_BINS)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        for i, conv in enumerate(self.convs):
            conv.load_state_dict(state_dict, module_prefix=f"convs.{i}.")
        self.classifier.load_state_dict(state_dict, key="classifier")

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.unsqueeze(-1).contiguous()
        tt_x = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.input_mesh_mapper,
        )
        for conv in self.convs:
            tt_x = conv(tt_x)

        tt_x = ttnn.to_memory_config(tt_x, ttnn.DRAM_MEMORY_CONFIG)
        tt_x = ttnn.reshape(tt_x, (-1, self.in_features))

        tt_x = self.classifier(tt_x)

        tt_x = ttnn.sigmoid(tt_x)
        return ttnn.to_torch(tt_x, mesh_composer=self.output_mesh_composer).to(torch.float32)


def load_crepe(device: ttnn.Device):
    """Load local model weights from the project data directory."""
    weights_path = Path(__file__).resolve().parent.parent / "data" / "assets" / f"crepe-tiny.safetensors"
    model = Crepe(device=device)
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    return model


class CrepePredictor:
    def __init__(self, device: ttnn.Device | None = None):
        if device is None:
            raise ValueError("TTNN CrepePredictor requires a TT device.")
        self.device = device
        self.model = load_crepe(device=device)

    def predict(
        self,
        audio,
        sample_rate,
        hop_length=None,
        fmin=50.0,
        fmax=MAX_FMAX,
        decoder=viterbi,
        return_periodicity=False,
        batch_size=None,
        pad=True,
        confidence_threshold: float | None = None,
    ):
        """Performs pitch estimation with the predictor's loaded model."""
        results = []
        batch_size = 16
        with torch.no_grad():
            generator = preprocess(audio, sample_rate, hop_length, batch_size, pad)
            for frames in generator:
                probabilities = self.model(frames)
                # return probabilities
                probabilities = probabilities.reshape(audio.size(0), -1, PITCH_BINS).transpose(1, 2)
                # return probabilities
                result = postprocess(
                    probabilities,
                    fmin,
                    fmax,
                    decoder,
                    return_periodicity=return_periodicity,
                    confidence_threshold=confidence_threshold,
                )

                if isinstance(result, tuple):
                    result = (result[0], result[1])

                results.append(result)
        if return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)
        return torch.cat(results, 1)
