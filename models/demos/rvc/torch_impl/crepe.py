# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import librosa
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

CENTS_PER_BIN = 20  # cents

PITCH_BINS = 360
MAX_FMAX = 2006.0  # hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples


def viterbi(logits):
    """Sample observations using viterbi decoding"""
    # return logits.transpose(1,2), logits.transpose(1,2)
    # Create viterbi transition matrix
    if not hasattr(viterbi, "transition"):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        viterbi.transition = transition

    # Normalize logits
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)

    # return probs.transpose(1,2), probs.transpose(1,2)

    # Convert to numpy
    sequences = probs.cpu().numpy()

    # Perform viterbi decoding
    bins = np.array([librosa.sequence.viterbi(sequence, viterbi.transition).astype(np.int64) for sequence in sequences])

    # Convert to pytorch
    bins = torch.tensor(bins)

    # Convert to frequency in Hz
    return bins, bins_to_frequency(bins)


class CrepePredictor:
    def __init__(self):
        self.model = load_crepe()

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
            generator = preprocess(
                audio,
                sample_rate,
                hop_length,
                batch_size,
                pad,
            )
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
                else:
                    result = result

                results.append(result)

        if return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)

        return torch.cat(results, 1)


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise
    noise = scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size())
    return cents + cents.new_tensor(noise)


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    cents = CENTS_PER_BIN * bins + 1997.3794084376191
    noise = scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size())
    cents = cents + cents.new_tensor(noise)
    return 10 * 2 ** (cents / 1200)


def argmax(logits):
    """Sample observations by taking the argmax"""
    bins = logits.argmax(dim=1)

    # Convert to frequency in Hz
    return bins, bins_to_frequency(bins)


def weighted_argmax(logits):
    """Sample observations using weighted sum near the argmax"""
    # Find center of analysis window
    bins = logits.argmax(dim=1)

    # Find bounds of analysis window
    start = torch.clamp(bins - 4, min=0)
    end = torch.clamp(bins + 5, max=logits.size(1))

    # Mask out everything outside of window
    for batch in range(logits.size(0)):
        for time in range(logits.size(2)):
            logits[batch, : start[batch, time], time] = -float("inf")
            logits[batch, end[batch, time] :, time] = -float("inf")

    # Construct weights
    if not hasattr(weighted_argmax, "weights"):
        weights = bins_to_cents(torch.arange(360))
        weighted_argmax.weights = weights[None, :, None]

    # Convert to probabilities
    with torch.no_grad():
        probs = torch.sigmoid(logits)

    # Apply weights
    cents = (weighted_argmax.weights * probs).sum(dim=1) / probs.sum(dim=1)

    # Convert to frequency in Hz
    return bins, 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=torch.floor):
    """Convert frequency in Hz to pitch bins"""
    cents = 1200 * torch.log2(frequency / 10.0)
    bins = (cents - 1997.3794084376191) / CENTS_PER_BIN
    return quantize_fn(bins).int()


def preprocess(audio, sample_rate, hop_length=None, batch_size=None, pad=True):
    """Convert audio to model input

    Arguments
        audio (torch.tensor [shape=(1, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        frames (torch.tensor [shape=(1 + int(time // hop_length), 1024)])
    """
    # Default hop length of 10 ms
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Resample
    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    # Maybe pad
    if pad:
        total_frames = 1 + int(audio.size(1) // hop_length)
        audio = F.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else:
        total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):
        # Batch indices
        start = max(0, i * hop_length)
        end = min(audio.size(1), (i + batch_size - 1) * hop_length + WINDOW_SIZE)

        # Chunk
        frames = F.unfold(audio[:, None, None, start:end], kernel_size=(1, WINDOW_SIZE), stride=(1, hop_length))

        # shape=(1 + int(time / hop_length, 1024)
        frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE)

        # Mean-center
        frames -= frames.mean(dim=1, keepdim=True)

        # Scale
        # Note: during silent frames, this produces very large values. But
        # this seems to be what the network expects.
        frames /= torch.max(torch.tensor(1e-10), frames.std(dim=1, keepdim=True))

        yield frames


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.padding = padding
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.batch_norm = torch.nn.BatchNorm1d(num_features=out_channels, eps=0.0010000000474974513, momentum=0.0)

    def forward(self, input):
        x = F.pad(input, self.padding)
        x = self.conv(x)
        x = F.relu(x)
        x = self.batch_norm(x)
        return F.max_pool1d(x, 2, 2)


class Crepe(torch.nn.Module):
    """Crepe model definition"""

    def __init__(self):
        super().__init__()
        in_channels = [1, 128, 16, 16, 16, 32]
        out_channels = [128, 16, 16, 16, 32, 64]
        self.in_features = 256

        # Shared layer parameters
        kernel_sizes = [512] + 5 * [64]
        strides = [4] + 5 * [1]
        paddings = [(254, 254)] + 5 * [(31, 32)]
        # Layer definitions
        self.convs = torch.nn.ModuleList(
            [ConvBlock(in_channels[i], out_channels[i], kernel_sizes[i], strides[i], paddings[i]) for i in range(6)]
        )

        self.classifier = torch.nn.Linear(in_features=self.in_features, out_features=PITCH_BINS)

    def forward(self, x):
        x = x[:, None, :]
        # Forward pass through first five layers
        for conv in self.convs:
            x = conv(x)
        # shape=(batch, self.in_features)
        x = x.permute(0, 2, 1).reshape(-1, self.in_features)
        # Compute logits
        x = self.classifier(x)
        output = torch.sigmoid(x)
        return output


def load_crepe():
    """Load local model weights from the project assets directory."""
    weights_path = Path(__file__).resolve().parent.parent / "data" / "assets" / "crepe-tiny.safetensors"
    model = Crepe()
    from safetensors.torch import load_file

    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_periodicity(probabilities, bins):
    """Computes the periodicity from the network output and pitch bins"""
    # shape=(batch * time / hop_length, 360)
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)

    # shape=(batch * time / hop_length, 1)
    bins_stacked = bins.reshape(-1, 1).to(torch.int64)

    # Use maximum logit over pitch bins as periodicity
    periodicity = probs_stacked.gather(1, bins_stacked)

    # shape=(batch, time / hop_length)
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
    """Convert model output to F0 and periodicity

    Arguments
        probabilities (torch.tensor [shape=(1, 360, time / hop_length)])
            The probabilities for each pitch bin inferred by the network
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        viterbi (bool)
            Whether to use viterbi decoding
        return_periodicity (bool)
            Whether to also return the network confidence

    Returns
        pitch (torch.tensor [shape=(1, 1 + int(time // hop_length))])
        periodicity (torch.tensor [shape=(1, 1 + int(time // hop_length))])
    """

    # Convert frequency range to pitch bin range
    minidx = frequency_to_bins(torch.tensor(fmin))
    maxidx = frequency_to_bins(torch.tensor(fmax), torch.ceil)

    # Remove frequencies outside of allowable range
    probabilities[:, :minidx] = -float("inf")
    probabilities[:, maxidx:] = -float("inf")

    # Perform argmax or viterbi sampling
    # return decoder(probabilities).transpose(1, 2)
    bins, pitch = decoder(probabilities)

    if not return_periodicity:
        return pitch

    # Compute periodicity from probabilities and decoded pitch bins
    periodicity = compute_periodicity(probabilities, bins)
    pitch, periodicity = filter_by_confidence(pitch, periodicity, confidence_threshold)
    return pitch, periodicity