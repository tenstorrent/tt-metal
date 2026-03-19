# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

import torch
import numpy as np
import torchaudio
from typing import Optional, Union, Tuple, List
import math


class MelSpectrogramProcessor:
    """Mel-spectrogram processing utilities for SpeechT5 model."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mels: int = 80,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 power: float = 2.0,
                 normalized: bool = False,
                 center: bool = True,
                 pad_mode: str = "reflect"):
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        
        # Initialize mel filter bank
        self.mel_scale = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=self.f_max,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode
        )
    
    def compute_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel-spectrogram."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        mel_spec = self.mel_scale(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        return mel_spec
    
    def normalize_mel_spectrogram(self, mel_spec: torch.Tensor, 
                                 mean: Optional[torch.Tensor] = None,
                                 std: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Normalize mel-spectrogram using z-score normalization."""
        if mean is None:
            mean = mel_spec.mean()
        if std is None:
            std = mel_spec.std()
        
        return (mel_spec - mean) / (std + 1e-8)
    
    def pad_or_trim_mel(self, mel_spec: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad or trim mel-spectrogram to target length."""
        current_length = mel_spec.size(-1)
        
        if current_length > target_length:
            # Trim
            return mel_spec[..., :target_length]
        elif current_length < target_length:
            # Pad
            pad_amount = target_length - current_length
            return torch.nn.functional.pad(mel_spec, (0, pad_amount), mode='constant', value=0)
        else:
            return mel_spec


class SpeakerXVectorProcessor:
    """Speaker x-vector embedding processing utilities."""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
    
    def load_speaker_embedding(self, embedding_path: str) -> torch.Tensor:
        """Load speaker embedding from file."""
        if embedding_path.endswith('.npy'):
            embedding = np.load(embedding_path)
            return torch.from_numpy(embedding).float()
        elif embedding_path.endswith('.pt') or embedding_path.endswith('.pth'):
            return torch.load(embedding_path)
        else:
            raise ValueError(f"Unsupported embedding format: {embedding_path}")
    
    def normalize_speaker_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """L2 normalize speaker embedding."""
        return torch.nn.functional.normalize(embedding, p=2, dim=-1)
    
    def interpolate_speaker_embeddings(self, 
                                     source_embedding: torch.Tensor,
                                     target_embedding: torch.Tensor,
                                     alpha: float = 0.5) -> torch.Tensor:
        """Interpolate between source and target speaker embeddings."""
        assert 0.0 <= alpha <= 1.0, "Alpha must be between 0 and 1"
        
        interpolated = (1 - alpha) * source_embedding + alpha * target_embedding
        return self.normalize_speaker_embedding(interpolated)
    
    def expand_speaker_embedding(self, embedding: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Expand speaker embedding to match sequence length."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        # Expand to [batch_size, seq_length, embedding_dim]
        return embedding.unsqueeze(1).expand(-1, seq_length, -1)


class AudioPreprocessor:
    """Audio preprocessing utilities for SpeechT5."""
    
    def __init__(self, sample_rate: int = 16000, target_sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        
        if sample_rate != target_sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sample_rate
            )
        else:
            self.resampler = None
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and return waveform and sample rate."""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform, sr
    
    def preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess audio waveform."""
        # Resample if necessary
        if self.resampler is not None and sample_rate != self.target_sample_rate:
            waveform = self.resampler(waveform)
        
        # Normalize audio
        waveform = self.normalize_audio(waveform)
        
        # Remove silence
        waveform = self.trim_silence(waveform)
        
        return waveform
    
    def normalize_audio(self, waveform: torch.Tensor, target_level: float = -23.0) -> torch.Tensor:
        """Normalize audio to target level in dB."""
        # RMS normalization
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            current_db = 20 * torch.log10(rms)
            gain_db = target_level - current_db
            gain_linear = 10 ** (gain_db / 20)
            waveform = waveform * gain_linear
        
        return waveform
    
    def trim_silence(self, waveform: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Trim silence from beginning and end of audio."""
        # Find non-silent regions
        energy = waveform.abs()
        non_silent = energy > threshold
        
        if non_silent.any():
            start_idx = non_silent.argmax().item()
            end_idx = len(non_silent) - non_silent.flip(0).argmax().item()
            waveform = waveform[..., start_idx:end_idx]
        
        return waveform


class AudioPostprocessor:
    """Audio postprocessing utilities for SpeechT5."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def denormalize_mel_spectrogram(self, mel_spec: torch.Tensor,
                                   mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize mel-spectrogram."""
        return mel_spec * std + mean
    
    def apply_griffin_lim(self, mel_spec: torch.Tensor,
                         n_fft: int = 1024,
                         hop_length: int = 256,
                         n_iter: int = 32) -> torch.Tensor:
        """Apply Griffin-Lim algorithm to convert mel-spectrogram to waveform."""
        # Convert log mel to linear
        mel_spec = torch.exp(mel_spec)
        
        # Initialize Griffin-Lim transform
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=n_iter,
            power=2.0
        )
        
        # Convert mel to waveform (this is a simplified approach)
        # In practice, you'd need a proper mel-to-linear conversion
        waveform = griffin_lim(mel_spec)
        
        return waveform
    
    def save_audio(self, waveform: torch.Tensor, output_path: str,
                   sample_rate: Optional[int] = None) -> None:
        """Save waveform to audio file."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure waveform is in correct format
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Clamp values to prevent clipping
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        torchaudio.save(output_path, waveform, sample_rate)


class SpeechT5Utils:
    """Main utility class combining all SpeechT5 preprocessing utilities."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 speaker_embedding_dim: int = 512):
        
        self.mel_processor = MelSpectrogramProcessor(
            sample_rate=sample_rate,
            n_mels=n_mels
        )
        self.speaker_processor = SpeakerXVectorProcessor(
            embedding_dim=speaker_embedding_dim
        )
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate
        )
        self.audio_postprocessor = AudioPostprocessor(
            sample_rate=sample_rate
        )
    
    def prepare_input_features(self, 
                              audio_path: str,
                              speaker_embedding_path: str,
                              max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare input features for SpeechT5 model."""
        # Load and preprocess audio
        waveform, sr = self.audio_preprocessor.load_audio(audio_path)
        waveform = self.audio_preprocessor.preprocess_audio(waveform, sr)
        
        # Compute mel-spectrogram
        mel_spec = self.mel_processor.compute_mel_spectrogram(waveform)
        mel_spec = self.mel_processor.normalize_mel_spectrogram(mel_spec)
        
        # Pad or trim if max_length specified
        if max_length is not None:
            mel_spec = self.mel_processor.pad_or_trim_mel(mel_spec, max_length)
        
        # Load and process speaker embedding
        speaker_embedding = self.speaker_processor.load_speaker_embedding(speaker_embedding_path)
        speaker_embedding = self.speaker_processor.normalize_speaker_embedding(speaker_embedding)
        
        # Expand speaker embedding to match mel-spectrogram sequence length
        seq_length = mel_spec.size(-1)
        speaker_embedding = self.speaker_processor.expand_speaker_embedding(
            speaker_embedding, seq_length
        )
        
        return mel_spec, speaker_embedding