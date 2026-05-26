# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS Speaker Encoder: ECAPA-TDNN for extracting speaker embeddings.

The Speaker Encoder is a relatively small CNN (12M params) that runs once per
utterance for voice cloning. It processes a mel spectrogram from reference audio
and produces a fixed-length speaker embedding that is injected into the Talker.

Implementation strategy: the ECAPA-TDNN runs on the host (CPU) using the
PyTorch reference model, and the resulting speaker embedding is transferred to
TT device. This is practical because:
  1. The model is small (12M params) and runs once per utterance
  2. Mel spectrogram extraction is already CPU-side (STFT + mel filterbank)
  3. The complex Conv1d operations (dilated, Res2Net, attentive pooling) would
     require significant ttnn.conv1d tuning for minimal latency benefit

The speaker embedding is returned as a ttnn tensor ready for injection into
the Talker's hidden state via addition.
"""

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.qwen3_tts.reference.speaker_encoder_ref import SpeakerEncoderReference


def mel_spectrogram(
    audio: np.ndarray,
    sr: int = 24000,
    n_fft: int = 1024,
    num_mels: int = 128,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: int = 0,
    fmax: int = 12000,
) -> torch.Tensor:
    """Compute mel spectrogram from audio waveform (CPU).

    Args:
        audio: [num_samples] numpy array, float32, normalized to [-1, 1]
        sr: sample rate (must be 24000)
        n_fft: FFT size
        num_mels: number of mel bins
        hop_size: hop size for STFT
        win_size: window size for STFT
        fmin: minimum frequency for mel filterbank
        fmax: maximum frequency for mel filterbank

    Returns:
        mel: [1, T, num_mels] torch tensor (time-major, ready for speaker encoder)
    """
    from librosa.filters import mel as librosa_mel_fn

    y = torch.from_numpy(audio).float().unsqueeze(0)

    mel_basis = torch.from_numpy(
        librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    ).float()
    hann_window = torch.hann_window(win_size)

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window, center=False, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel.transpose(1, 2)  # [1, T, num_mels]


class SpeakerEncoder:
    """
    Speaker Encoder wrapper: runs ECAPA-TDNN on host, returns ttnn tensor.

    Usage:
        encoder = SpeakerEncoder.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", mesh_device)
        speaker_emb_tt = encoder.encode(audio_numpy, sr=24000)
        # speaker_emb_tt is [1, 1, 1, enc_dim] on device, ready for Talker injection
    """

    def __init__(self, model: SpeakerEncoderReference, mesh_device, dtype=torch.bfloat16):
        self.model = model
        self.model.eval()
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.enc_dim = model.enc_dim

    @classmethod
    def from_pretrained(cls, model_name_or_path, mesh_device, dtype=torch.bfloat16):
        model = SpeakerEncoderReference.from_pretrained(model_name_or_path)
        model = model.to(dtype)
        return cls(model, mesh_device, dtype)

    @classmethod
    def from_state_dict(cls, state_dict, mesh_device, dtype=torch.bfloat16):
        """Load from a pre-filtered state dict (keys with 'speaker_encoder.' prefix stripped)."""
        model = SpeakerEncoderReference.from_hf_state_dict(state_dict)
        model = model.to(dtype)
        return cls(model, mesh_device, dtype)

    @torch.inference_mode()
    def encode(self, audio: np.ndarray, sr: int = 24000) -> "ttnn.Tensor":
        """Extract speaker embedding from audio and transfer to TT device.

        Args:
            audio: [num_samples] numpy array, float32
            sr: sample rate (must be 24000)

        Returns:
            ttnn.Tensor [1, 1, 1, enc_dim] on device (bfloat16, TILE_LAYOUT)
        """
        assert sr == 24000, f"Expected 24kHz audio, got {sr}Hz"

        mel = mel_spectrogram(audio, sr=sr).to(self.dtype)
        logger.info(f"Mel spectrogram: {mel.shape} ({mel.shape[1]} frames)")

        embedding = self.model(mel)  # [1, enc_dim]
        logger.info(f"Speaker embedding: {embedding.shape}, norm={embedding.norm():.4f}")

        emb_tt = ttnn.from_torch(
            embedding.float().unsqueeze(0).unsqueeze(0),  # [1, 1, 1, enc_dim]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return emb_tt

    @torch.inference_mode()
    def extract_embedding(self, audio: np.ndarray, sr: int = 24000) -> torch.Tensor:
        """Extract speaker embedding as a CPU torch tensor (for saving).

        Args:
            audio: [num_samples] numpy array, float32
            sr: sample rate (must be 24000)

        Returns:
            torch.Tensor [1, enc_dim] float32
        """
        assert sr == 24000, f"Expected 24kHz audio, got {sr}Hz"
        mel = mel_spectrogram(audio, sr=sr).to(self.dtype)
        embedding = self.model(mel)  # [1, enc_dim]
        return embedding.float()

    def save_embedding(self, audio: np.ndarray, sr: int, path: str) -> torch.Tensor:
        """Extract speaker embedding from audio and save to .safetensors file.

        Args:
            audio: [num_samples] numpy array, float32
            sr: sample rate (must be 24000)
            path: output file path (.safetensors)

        Returns:
            torch.Tensor [1, enc_dim] the saved embedding
        """
        from safetensors.torch import save_file

        embedding = self.extract_embedding(audio, sr)
        save_file(
            {"speaker_embedding": embedding},
            path,
            metadata={"enc_dim": str(self.enc_dim), "sample_rate": str(sr)},
        )
        logger.info(f"Saved speaker embedding to {path} (enc_dim={self.enc_dim}, norm={embedding.norm():.4f})")
        return embedding

    @staticmethod
    def load_embedding(path: str, mesh_device) -> "ttnn.Tensor":
        """Load saved speaker embedding and transfer to TT device.

        Args:
            path: path to .safetensors file
            mesh_device: TT mesh device

        Returns:
            ttnn.Tensor [1, 1, 1, enc_dim] on device (bfloat16, TILE_LAYOUT)
        """
        from safetensors.torch import load_file

        data = load_file(path)
        embedding = data["speaker_embedding"]  # [1, enc_dim]
        logger.info(f"Loaded speaker embedding from {path}: {embedding.shape}, norm={embedding.norm():.4f}")

        emb_tt = ttnn.from_torch(
            embedding.unsqueeze(0).unsqueeze(0),  # [1, 1, 1, enc_dim]
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        return emb_tt

    @torch.inference_mode()
    def encode_mel(self, mel: torch.Tensor) -> "ttnn.Tensor":
        """Extract speaker embedding from pre-computed mel spectrogram.

        Args:
            mel: [B, T, mel_dim] mel spectrogram tensor

        Returns:
            ttnn.Tensor [1, 1, B, enc_dim] on device
        """
        mel = mel.to(self.dtype)
        embedding = self.model(mel)  # [B, enc_dim]

        emb_tt = ttnn.from_torch(
            embedding.float().unsqueeze(0).unsqueeze(0),  # [1, 1, B, enc_dim]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return emb_tt
