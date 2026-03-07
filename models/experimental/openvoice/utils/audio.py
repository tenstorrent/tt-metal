# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Audio processing utilities for OpenVoice.

Handles mel spectrogram computation, audio loading, and processing.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Default audio parameters (from OpenVoice config)
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 256
DEFAULT_WIN_LENGTH = 1024
DEFAULT_N_MELS = 80


def load_audio(
    path: Union[str, Path],
    sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file.

    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono if True

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required for audio loading: pip install librosa")

    audio, sample_rate = librosa.load(str(path), sr=sr, mono=mono)
    return audio, sample_rate


def save_audio(
    path: Union[str, Path],
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
):
    """
    Save audio to file.

    Args:
        path: Output path
        audio: Audio array (mono or stereo)
        sr: Sample rate
    """
    try:
        import numpy as np
        import soundfile as sf

        # Ensure audio is numpy array and handle dimensionality
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        audio = np.asarray(audio)
        # Squeeze extra dimensions for mono audio
        audio = np.squeeze(audio)
        # Ensure contiguous for soundfile
        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio)
        # Force float32 for proper saving
        audio = audio.astype(np.float32)
        sf.write(str(path), audio, sr)
    except ImportError:
        raise ImportError("soundfile required for audio saving: pip install soundfile")


def spectrogram_torch(
    y: "torch.Tensor",
    n_fft: int,
    sampling_rate: int,
    hop_length: int,
    win_length: int,
    center: bool = False,
) -> "torch.Tensor":
    """
    Compute magnitude spectrogram using PyTorch.

    This matches OpenVoice's mel_processing.spectrogram_torch function.

    Args:
        y: Audio tensor [B, T] or [T]
        n_fft: FFT size
        sampling_rate: Sample rate (unused but kept for API compatibility)
        hop_length: Hop length
        win_length: Window length
        center: Whether to pad signal

    Returns:
        Magnitude spectrogram [B, n_fft//2+1, T']
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    # Ensure 2D
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # Hann window
    hann_window = torch.hann_window(win_length, device=y.device, dtype=y.dtype)

    # Pad if needed
    if not center:
        # Pad to ensure we get expected output length
        pad_amount = (n_fft - hop_length) // 2
        y = F.pad(y.unsqueeze(1), (pad_amount, pad_amount), mode="reflect").squeeze(1)

    # STFT
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Magnitude
    spec = torch.abs(spec)

    return spec


def spectrogram_numpy(
    y: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    win_length: int = DEFAULT_WIN_LENGTH,
) -> np.ndarray:
    """
    Compute magnitude spectrogram using librosa/numpy.

    Args:
        y: Audio array
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length

    Returns:
        Magnitude spectrogram [n_fft//2+1, T']
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required: pip install librosa")

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=False,
    )

    return np.abs(stft)


def mel_spectrogram(
    y: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    n_mels: int = DEFAULT_N_MELS,
    hop_length: int = DEFAULT_HOP_LENGTH,
    win_length: int = DEFAULT_WIN_LENGTH,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Compute mel spectrogram.

    Args:
        y: Audio array
        sr: Sample rate
        n_fft: FFT size
        n_mels: Number of mel bands
        hop_length: Hop length
        win_length: Window length
        fmin: Minimum frequency
        fmax: Maximum frequency

    Returns:
        Mel spectrogram [n_mels, T']
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required: pip install librosa")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        center=False,
    )

    return mel


def audio_to_spectrogram(
    audio_path: Union[str, Path],
    config: dict,
) -> "torch.Tensor":
    """
    Load audio and compute spectrogram for model input.

    Args:
        audio_path: Path to audio file
        config: Model config with audio parameters

    Returns:
        Spectrogram tensor [1, n_fft//2+1, T]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    # Get config params
    data_cfg = config.get("data", config)
    sr = data_cfg.get("sampling_rate", DEFAULT_SAMPLE_RATE)
    n_fft = data_cfg.get("filter_length", DEFAULT_N_FFT)
    hop_length = data_cfg.get("hop_length", DEFAULT_HOP_LENGTH)
    win_length = data_cfg.get("win_length", DEFAULT_WIN_LENGTH)

    # Load audio
    audio, _ = load_audio(audio_path, sr=sr)

    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio)

    # Compute spectrogram
    spec = spectrogram_torch(
        audio_tensor,
        n_fft=n_fft,
        sampling_rate=sr,
        hop_length=hop_length,
        win_length=win_length,
        center=False,
    )

    return spec


class AudioProcessor:
    """
    Audio processor with cached parameters.

    Provides consistent audio processing matching OpenVoice's requirements.
    """

    def __init__(self, config: dict):
        """
        Initialize processor from config.

        Args:
            config: Model config dict
        """
        data_cfg = config.get("data", config)

        self.sample_rate = data_cfg.get("sampling_rate", DEFAULT_SAMPLE_RATE)
        self.n_fft = data_cfg.get("filter_length", DEFAULT_N_FFT)
        self.hop_length = data_cfg.get("hop_length", DEFAULT_HOP_LENGTH)
        self.win_length = data_cfg.get("win_length", DEFAULT_WIN_LENGTH)
        self.n_mels = data_cfg.get("n_mel_channels", DEFAULT_N_MELS)

        # Derived
        self.spec_channels = self.n_fft // 2 + 1

    def load_audio(self, path: Union[str, Path]) -> np.ndarray:
        """Load audio at correct sample rate."""
        audio, _ = load_audio(path, sr=self.sample_rate)
        return audio

    def audio_to_spec(self, audio: Union[np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
        """
        Convert audio to spectrogram.

        Args:
            audio: Audio array or tensor

        Returns:
            Spectrogram [1, spec_channels, T]
        """
        if isinstance(audio, np.ndarray):
            audio = torch.FloatTensor(audio)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        spec = spectrogram_torch(
            audio,
            n_fft=self.n_fft,
            sampling_rate=self.sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
        )

        return spec

    def load_and_process(self, path: Union[str, Path]) -> "torch.Tensor":
        """
        Load audio file and convert to spectrogram.

        Args:
            path: Audio file path

        Returns:
            Spectrogram tensor [1, spec_channels, T]
        """
        audio = self.load_audio(path)
        return self.audio_to_spec(audio)
