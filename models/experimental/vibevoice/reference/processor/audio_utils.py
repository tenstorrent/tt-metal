from typing import Optional

import numpy as np


class AudioNormalizer:
    """
    Audio normalization class for VibeVoice tokenizer.

    This class provides audio normalization to ensure consistent input levels
    for the VibeVoice tokenizer while maintaining audio quality.
    """

    def __init__(self, target_dB_FS: float = -25, eps: float = 1e-6):
        """
        Initialize the audio normalizer.

        Args:
            target_dB_FS (float): Target dB FS level for the audio. Default: -25
            eps (float): Small value to avoid division by zero. Default: 1e-6
        """
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def tailor_dB_FS(self, audio: np.ndarray) -> tuple:
        """
        Adjust the audio to the target dB FS level.

        Args:
            audio (np.ndarray): Input audio signal

        Returns:
            tuple: (normalized_audio, rms, scalar)
        """
        rms = np.sqrt(np.mean(audio**2))
        scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
        normalized_audio = audio * scalar
        return normalized_audio, rms, scalar

    def avoid_clipping(self, audio: np.ndarray, scalar: Optional[float] = None) -> tuple:
        """
        Avoid clipping by scaling down if necessary.

        Args:
            audio (np.ndarray): Input audio signal
            scalar (float, optional): Explicit scaling factor

        Returns:
            tuple: (normalized_audio, scalar)
        """
        if scalar is None:
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                scalar = max_val + self.eps
            else:
                scalar = 1.0

        return audio / scalar, scalar

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize the audio by adjusting to target dB FS and avoiding clipping.

        Args:
            audio (np.ndarray): Input audio signal

        Returns:
            np.ndarray: Normalized audio signal
        """
        # First adjust to target dB FS
        audio, _, _ = self.tailor_dB_FS(audio)
        # Then avoid clipping
        audio, _ = self.avoid_clipping(audio)
        return audio
