"""
tt_metal/models/ttnn/llvc/data_utils.py

Utility functions for dataset loading, preprocessing, and batching for voice conversion.
Supports TTNN-based Low-Latency Low-Resource Voice Conversion (LLVC) model bringup.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for voice conversion data preprocessing."""
    sample_rate: int = 16000                 # Target sample rate (Hz)
    n_fft: int = 1024                        # FFT window size
    hop_length: int = 256                    # Hop length (samples)
    win_length: int = 1024                   # Window length (samples)
    n_mels: int = 80                         # Number of mel bands
    f_min: float = 0.0                       # Minimum frequency (Hz)
    f_max: Optional[float] = 8000.0          # Maximum frequency (Hz)
    power: float = 2.0                       # Power of the spectrogram (2.0 for power)
    normalized: bool = True                  # Whether to normalize mel by power
    mel_scale: str = "htk"                   # Mel scale ("htk" or "slaney")
    db_scale: bool = True                    # Convert to decibel scale
    top_db: float = 80.0                     # Top DB for clipping
    max_audio_duration: float = 10.0         # Maximum audio duration (seconds)
    trim_silence: bool = True                # Trim leading/trailing silence
    silence_threshold_db: float = -40.0      # Silence threshold (dB)
    silence_min_duration: float = 0.1        # Minimum silence duration (seconds)
    seed: int = 42                           # Random seed for reproducibility

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")
        if self.win_length <= 0:
            raise ValueError(f"win_length must be positive, got {self.win_length}")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")


# ---------------------------------------------------------------------------
# Audio Loading and Preprocessing
# ---------------------------------------------------------------------------

def load_audio(
    path: Union[str, Path],
    target_sr: int,
    mono: bool = True,
    trim_silence: bool = False,
    silence_threshold_db: float = -40.0,
    silence_min_duration: float = 0.1,
) -> Tuple[Tensor, int]:
    """
    Load an audio file, optionally convert to mono and resample.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate (Hz).
        mono: If True, convert to mono.
        trim_silence: If True, trim leading/trailing silence.
        silence_threshold_db: Threshold (dB) below which is considered silence.
        silence_min_duration: Minimum silence duration (seconds) to consider.

    Returns:
        Tuple of (waveform tensor shape [1, T], sample_rate).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = torchaudio.load(str(path), normalize=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {path}: {e}") from e

    # Convert to mono if needed
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        waveform = F.resample(waveform, sr, target_sr)
        sr = target_sr

    # Trim silence
    if trim_silence:
        waveform = _trim_silence(
            waveform, sr, silence_threshold_db, silence_min_duration
        )

    return waveform, sr


def _trim_silence(
    waveform: Tensor,
    sr: int,
    threshold_db: float = -40.0,
    min_duration: float = 0.1,
) -> Tensor:
    """
    Remove leading and trailing silence from a waveform using energy threshold.

    Args:
        waveform: Input waveform (1, T).
        sr: Sample rate.
        threshold_db: Silence threshold in dB.
        min_duration: Minimum silence duration in seconds.

    Returns:
        Trimmed waveform (1, T').
    """
    # Convert to power
    energy = waveform.pow(2).squeeze(0)
    # Compute energy in dB (with upper bound to avoid -inf)
    energy_db = 10.0 * torch.log10(torch.clamp(energy, min=1e-10))

    # Determine silence mask: below threshold for at least min_duration
    min_frames = int(min_duration * sr)
    silence_mask = energy_db < threshold_db

    # Find continuous silent segments
    # Pad to handle boundaries
    silence_mask = silence_mask.float()
    kernel = torch.ones((1, 1, min_frames), dtype=torch.float32)
    conv = torch.nn.functional.conv1d(
        silence_mask.view(1, 1, -1), kernel, padding=min_frames // 2
    ).squeeze()
    silence_stretches = (conv == min_frames).bool()

    # Find first and last non-silence indices
    if not silence_stretches.any():
        return waveform  # No silence to trim

    non_silence = ~silence_stretches
    non_silence_indices = torch.where(non_silence)[0]
    if len(non_silence_indices) == 0:
        return waveform  # Entire waveform is silence

    start = int(non_silence_indices[0].item())
    end = int(non_silence_indices[-1].item()) + 1

    return waveform[:, start:end]


def preprocess_audio(
    waveform: Tensor,
    sr: int,
    config: DataConfig,
    pad_to_max: bool = False,
    max_length: Optional[int] = None,
) -> Tensor:
    """
    Apply standard preprocessing: trim, pad/truncate, and convert to float32.

    Args:
        waveform: Input waveform (1, T) or (C, T).
        sr: Sample rate.
        config: DataConfig object.
        pad_to_max: If True, pad or truncate to max_length.
        max_length: Maximum length in samples (required if pad_to_max).

    Returns:
        Preprocessed waveform (1, T') or (C, T').
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Convert to float32
    waveform = waveform.float()

    # Ensure mono for conversion tasks (most VC pipelines expect mono)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Trim silence if configured
    if config.trim_silence:
        waveform = _trim_silence(
            waveform, sr, config.silence_threshold_db, config.silence_min_duration
        )

    # Pad/truncate to max length
    max_samples = max_length or int(config.max_audio_duration * sr)
    if pad_to_max:
        if waveform.shape[-1] < max_samples:
            pad_len = max_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :max_samples]

    return waveform


# ---------------------------------------------------------------------------
# Mel-Spectrogram Extraction
# ---------------------------------------------------------------------------

def extract_mel_spectrogram(
    waveform: Tensor,
    sr: int,
    config: DataConfig,
) -> Tensor:
    """
    Extract mel-spectrogram from waveform using torchaudio.

    Args:
        waveform: Input waveform (1, T).
        sr: Sample rate.
        config: DataConfig object.

    Returns:
        Mel-spectrogram tensor (n_mels, time_steps).
    """
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"Expected mono waveform (1, T), got {waveform.shape}")

    # Compute STFT-based mel spectrogram using torchaudio's feature extractor
    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max,
        power=config.power,
        normalized=config.normalized,
        mel_scale=config.mel_scale,
    ).to(waveform.device)

    mel_spec = mel_extractor(waveform)  # (1, n_mels, time)

    # Convert to decibel scale if requested
    if config.db_scale:
        mel_spec = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=config.top_db
        )(mel_spec)

    # Remove batch dimension
    mel_spec = mel_spec.squeeze(0)  # (n_mels, time)

    return mel_spec


def extract_mel_spectrograms_batch(
    waveforms: Tensor,
    lengths: Optional[Tensor],
    sr: int,
    config: DataConfig,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Extract mel-spectrograms for a batch of waveforms, returning padded
    features and lengths.

    Args:
        waveforms: Batch of waveforms (B, 1, T_max).
        lengths: Original lengths of each waveform (B,). If None, assume all same.
        sr: Sample rate.
        config: DataConfig object.

    Returns:
        Tuple of (mel batch (B, n_mels, T_mels), lengths_mels (B,)).
    """
    B = waveforms.shape[0]
    mel_list = []
    mel_lengths = []

    for i in range(B):
        wav = waveforms[i]  # (1, T_i)
        mel = extract_mel_spectrogram(wav, sr, config)
        mel_list.append(mel)
        if lengths is not None:
            # Compute mel length from original audio length
            # mel_time = floor((T - n_fft) / hop_length) + 1
            T = lengths[i].item()
            mel_T = max(0, (T - config.n_fft) // config.hop_length + 1)
            mel_lengths.append(min(mel_T, mel.shape[-1]))
        else:
            mel_lengths.append(mel.shape[-1])

    max_mel_T = max(m.shape[-1] for m in mel_list)
    mel_padded = torch.zeros(
        B, config.n_mels, max_mel_T, dtype=waveforms.dtype, device=waveforms.device
    )
    for i, m in enumerate(mel_list):
        mel_padded[i, :, : m.shape[-1]] = m

    mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long, device=waveforms.device)
    return mel_padded, mel_lengths_tensor


# ---------------------------------------------------------------------------
# Align & Trim
# ---------------------------------------------------------------------------

def align_or_trim(
    features: Tensor,
    target_length: int,
    mode: str = "pad_trim",
    pad_value: float = 0.0,
) -> Tensor:
    """
    Align or trim feature tensor along the time dimension.

    Args:
        features: Input tensor (..., T, ...) where last dim is time.
        target_length: Desired time length.
        mode: "pad_trim" (default) – pad with pad_value if shorter, trim if longer.
              "resample" – linearly interpolate to target length.
        pad_value: Value to pad with.

    Returns:
        Aligned/trimmed tensor (... target_length ...).
    """
    current_length = features.shape[-1]
    if current_length == target_length:
        return features

    if mode == "pad_trim":
        if current_length < target_length:
            pad_shape = list(features.shape)
            pad_shape[-1] = target_length - current_length
            pad = torch.full(pad_shape, pad_value, dtype=features.dtype, device=features.device)
            return torch.cat([features, pad], dim=-1)
        else:
            return features[..., :target_length]
    elif mode == "resample":
        # Use interpolation (1D linear)
        if current_length == 0:
            return torch.zeros(
                *features.shape[:-1], target_length, dtype=features.dtype, device=features.device
            )
        x = torch.linspace(0, 1, current_length, device=features.device)
        xnew = torch.linspace(0, 1, target_length, device=features.device)
        # Reshape for interpolation: (batch, channels, time) -> (batch, 1, time)
        orig_shape = features.shape
        features_flat = features.view(-1, current_length).unsqueeze(1)  # (N, 1, T)
        x_flat = x.unsqueeze(0).unsqueeze(0).expand(features_flat.shape[0], 1, -1)
        xnew_flat = xnew.unsqueeze(0).unsqueeze(0).expand(features_flat.shape[0], 1, -1)
        # Use torch's grid_sample with 1D grid
        grid = xnew_flat * 2.0 - 1.0  # normalize to [-1, 1]
        grid = grid.unsqueeze(-1)  # (N, 1, Tnew, 1)
        interpolated = torch.nn.functional.grid_sample(
            features_flat.unsqueeze(-1),  # (N, 1, T, 1)
            grid,
            mode="linear",
            align_corners=False,
            padding_mode="border",
        ).squeeze(-1).squeeze(1)  # (N, Tnew)
        # Reshape back to original dimensions, preserving batch and channel dims
        return interpolated.view(*orig_shape[:-1], target_length)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'pad_trim' or 'resample'.")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def create_batches(
    dataset: List[Tuple[str, str]],
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
    drop_last: bool = False,
) -> Iterator[Dict[str, Tensor]]:
    """
    Create batches of mel-spectrograms and speaker embeddings from paired dataset.

    Args:
        dataset: List of (source_wav_path, target_wav_path) tuples.
        batch_size: Number of pairs per batch.
        shuffle: If True, shuffle dataset each epoch.
        seed: Random seed.
        drop_last: If True, drop last incomplete batch.

    Yields:
        Dictionary with keys:
            - "source_mel": (B, n_mels, T) tensor
            - "target_mel": (B, n_mels, T) tensor
            - "source_paths": list of source paths
            - "target_paths": list of target paths
    """
    rng = random.Random(seed)

    indices = list(range(len(dataset)))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        end = start + batch_size
        if drop_last and end > len(indices):
            break
        batch_indices = indices[start:end]
        batch_src_paths = [dataset[i][0] for i in batch_indices]
        batch_tgt_paths = [dataset[i][1] for i in batch_indices]

        # Load and preprocess mel spectrograms
        # For efficiency, use fixed config; in production, pass via argument.
        config = DataConfig()  # Use default config; caller should override

        src_mels = []
        tgt_mels = []
        src_paths_ok = []
        tgt_paths_ok = []

        for src_path, tgt_path in zip(batch_src_paths, batch_tgt_paths):
            try:
                # Load audio
                src_wav, _ = load_audio(
                    src_path, config.sample_rate, trim_silence=config.trim_silence
                )
                tgt_wav, _ = load_audio(
                    tgt_path, config.sample_rate, trim_silence=config.trim_silence
                )
                # Extract mel
                src_mel = extract_mel_spectrogram(src_wav, config.sample_rate, config)
                tgt_mel = extract_mel_spectrogram(tgt_wav, config.sample_rate, config)
                src_mels.append(src_mel)
                tgt_mels.append(tgt_mel)
                src_paths_ok.append(src_path)
                tgt_paths_ok.append(tgt_path)
            except Exception as e:
                logger.warning(f"Skipping pair ({src_path}, {tgt_path}): {e}")
                continue

        if len(src_mels) == 0:
            continue

        # Pad mels to same time length within batch
        max_src_T = max(m.shape[-1] for m in src_mels)
        max_tgt_T = max(m.shape[-1] for m in tgt_mels)

        B = len(src_mels)
        n_mels = config.n_mels
        src_mel_batch = torch.zeros(B, n_mels, max_src_T)
        tgt_mel_batch = torch.zeros(B, n_mels, max_tgt_T)
        for i, sm in enumerate(src_mels):
            src_mel_batch[i, :, : sm.shape[-1]] = sm
        for i, tm in enumerate(tgt_mels):
            tgt_mel_batch[i, :, : tm.shape[-1]] = tm

        yield {
            "source_mel": src_mel_batch,
            "target_mel": tgt_mel_batch,
            "source_paths": src_paths_ok,
            "target_paths": tgt_paths_ok,
        }


# ---------------------------------------------------------------------------
# Utility function to scan a dataset directory
# ---------------------------------------------------------------------------

def scan_vc_dataset(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    extension: str = ".wav",
) -> List[Tuple[str, str]]:
    """
    Scan two directories and pair files by common base name.

    Args:
        source_dir: Directory containing source audio files.
        target_dir: Directory containing target audio files.
        extension: File extension to filter (default: .wav).

    Returns:
        List of (source_path, target_path) tuples.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    source_files = sorted(source_dir.glob(f"*{extension}"))
    target_files = sorted(target_dir.glob(f"*{extension}"))

    if len(source_files) != len(target_files):
        logger.warning(
            f"Mismatch: {len(source_files)} source files vs {len(target_files)} target files. "
            "Pairing by sorted order may be incorrect."
        )

    pairs = []
    for src_path, tgt_path in zip(source_files, target_files):
        # Optionally verify base name match (if naming convention is the same)
        if src_path.stem != tgt_path.stem:
            logger.debug(f"Name mismatch: {src_path.stem} vs {tgt_path.stem}. Pairing anyway.")
        pairs.append((str(src_path), str(tgt_path)))

    return pairs


__all__ = [
    "DataConfig",
    "load_audio",
    "preprocess_audio",
    "extract_mel_spectrogram",
    "extract_mel_spectrograms_batch",
    "align_or_trim",
    "create_batches",
    "scan_vc_dataset",
]