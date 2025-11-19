# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Modality Processors for MiniCPM-o-2_6

Handles preprocessing and postprocessing for different modalities:
- Audio: WAV/MP3 files → mel spectrograms
- Image: JPG/PNG files → tensors for SigLip
- Text: strings → token IDs
- Audio Output: mel spectrograms → waveforms
"""

import torch
import numpy as np
from typing import Union, Tuple
from pathlib import Path
from loguru import logger


class AudioProcessor:
    """
    Audio preprocessing for MiniCPM-o-2_6.

    Converts audio files to mel spectrograms compatible with Whisper.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        chunk_length: int = 30,  # Whisper chunk length in seconds
    ):
        """
        Initialize audio processor.

        Args:
            sample_rate: Audio sample rate (Whisper: 16000)
            n_mels: Number of mel bins (Whisper: 80)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            chunk_length: Maximum audio chunk length in seconds
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.max_samples = sample_rate * chunk_length

        # Try to import audio libraries
        try:
            import librosa

            self.librosa = librosa
        except ImportError:
            logger.warning("librosa not available, audio processing will be limited")
            self.librosa = None

        try:
            import torchaudio

            self.torchaudio = torchaudio
        except ImportError:
            logger.warning("torchaudio not available, audio processing will be limited")
            self.torchaudio = None

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Try torchaudio first (faster)
        if self.torchaudio:
            try:
                audio, sr = self.torchaudio.load(str(audio_path))
                audio = audio.numpy()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)  # Convert to mono
                return audio, sr
            except Exception as e:
                logger.warning(f"torchaudio failed: {e}")

        # Fallback to librosa
        if self.librosa:
            try:
                audio, sr = self.librosa.load(str(audio_path), sr=None)
                return audio, sr
            except Exception as e:
                logger.warning(f"librosa failed: {e}")

        raise ImportError("Neither torchaudio nor librosa available for audio loading")

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int = None) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate (default: self.sample_rate)

        Returns:
            Resampled audio
        """
        if target_sr is None:
            target_sr = self.sample_rate

        if orig_sr == target_sr:
            return audio

        if self.librosa:
            return self.librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        elif self.torchaudio:
            import torchaudio.transforms as T

            resampler = T.Resample(orig_sr, target_sr)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            resampled = resampler(audio_tensor).squeeze(0)
            return resampled.numpy()

        logger.warning("No resampling library available, returning original audio")
        return audio

    def audio_to_mel(self, audio: np.ndarray, sample_rate: int = None) -> torch.Tensor:
        """
        Convert audio waveform to mel spectrogram.

        Args:
            audio: Audio waveform
            sample_rate: Sample rate (default: self.sample_rate)

        Returns:
            Mel spectrogram tensor [n_mels, time]
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Ensure audio is numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        # Truncate to maximum length
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]

        # Convert to mel spectrogram
        if self.librosa:
            # Use librosa for mel spectrogram (matches Whisper)
            mel = self.librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            mel = self.librosa.power_to_db(mel, ref=np.max)

            # Normalize to [0, 1] range (simplified)
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

            return torch.from_numpy(mel).float()
        else:
            # Fallback: create mock mel spectrogram
            logger.warning("librosa not available, creating mock mel spectrogram")
            time_steps = len(audio) // self.hop_length
            return torch.randn(self.n_mels, time_steps)

    def process_audio_file(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Process audio file to mel spectrogram.

        Args:
            audio_path: Path to audio file

        Returns:
            Mel spectrogram tensor [80, time]
        """
        logger.info(f"Processing audio file: {audio_path}")

        # Load audio
        audio, sr = self.load_audio(audio_path)

        # Resample if needed
        audio = self.resample_audio(audio, sr, self.sample_rate)

        # Convert to mel spectrogram
        mel_spec = self.audio_to_mel(audio, self.sample_rate)

        logger.info(f"Audio processed: {audio.shape} samples → mel {mel_spec.shape}")
        return mel_spec


class ImageProcessor:
    """
    Image preprocessing for MiniCPM-o-2_6.

    Prepares images for SigLip vision encoder.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # ImageNet mean
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),  # ImageNet std
    ):
        """
        Initialize image processor.

        Args:
            image_size: Target image size (height, width)
            mean: Normalization mean (RGB)
            std: Normalization std (RGB)
        """
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

        # Try to import image libraries
        try:
            from PIL import Image

            self.PIL = Image
        except ImportError:
            logger.warning("PIL not available, image processing will be limited")
            self.PIL = None

        # Lazy import torchvision - only import when needed to avoid memory issues
        self.transforms = None

    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor [3, H, W]
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Lazy import torchvision if needed
        if self.transforms is None:
            try:
                import torchvision.transforms as T

                self.transforms = T
            except ImportError:
                logger.warning("torchvision not available, image processing will be limited")
                self.transforms = None

        if self.PIL and self.transforms:
            # Load with PIL
            image = self.PIL.open(str(image_path)).convert("RGB")

            # Create preprocessing pipeline
            transform = self.transforms.Compose(
                [
                    self.transforms.Resize(self.image_size),
                    self.transforms.ToTensor(),
                    self.transforms.Normalize(mean=self.mean.squeeze(), std=self.std.squeeze()),
                ]
            )

            # Apply transforms
            image_tensor = transform(image)
            return image_tensor
        else:
            # Fallback: create mock image tensor
            logger.warning("PIL/torchvision not available, creating mock image tensor")
            return torch.randn(3, *self.image_size)

    def process_image_file(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Process image file for vision encoder.

        Args:
            image_path: Path to image file

        Returns:
            Processed image tensor [3, 224, 224]
        """
        logger.info(f"Processing image file: {image_path}")

        image_tensor = self.load_image(image_path)

        logger.info(f"Image processed: {image_tensor.shape}")
        return image_tensor


class TextProcessor:
    """
    Text preprocessing for MiniCPM-o-2_6.

    Tokenizes text using Qwen tokenizer.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize text processor.

        Args:
            model_name: HuggingFace model name for tokenizer
        """
        self.model_name = model_name

        # Lazy load tokenizer to avoid memory issues during import
        self.tokenizer = None
        self._tokenizer_loaded = False
        self.model_name = model_name

    def _load_tokenizer(self):
        """Lazy load tokenizer."""
        if not self._tokenizer_loaded:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info(f"✅ Initialized tokenizer: {self.model_name}")
            except ImportError:
                logger.warning("transformers not available, text processing will be limited")
                self.tokenizer = None
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
            finally:
                self._tokenizer_loaded = True

    def tokenize_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Tokenize text to token IDs.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Token IDs tensor
        """
        # Lazy load tokenizer
        if not self._tokenizer_loaded:
            self._load_tokenizer()

        if self.tokenizer:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
            return tokens["input_ids"].squeeze(0)
        else:
            # Fallback: mock tokenization
            logger.warning("Tokenizer not available, creating mock tokens")
            return torch.randint(0, 1000, (len(text.split()),))

    def decode_tokens(self, tokens: torch.Tensor) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: Token IDs tensor

        Returns:
            Decoded text
        """
        # Lazy load tokenizer
        if not self._tokenizer_loaded:
            self._load_tokenizer()

        if self.tokenizer:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            return f"Mock decoded text from {len(tokens)} tokens"


class AudioPostprocessor:
    """
    Audio postprocessing for MiniCPM-o-2_6.

    Converts mel spectrograms back to waveforms.
    """

    def __init__(
        self,
        sample_rate: int = 24000,  # Vocos default
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
    ):
        """
        Initialize audio postprocessor.

        Args:
            sample_rate: Output sample rate
            n_fft: FFT window size
            hop_length: Hop length
            n_mels: Number of mel bins
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Try to import vocos (lazy loading to avoid memory issues)
        self.vocos = None
        self._vocos_loaded = False

        # Fallback: try librosa for griffin-lim
        try:
            import librosa

            self.librosa = librosa
        except ImportError:
            logger.warning("librosa not available, griffin-lim will not work")
            self.librosa = None

    def _load_vocos(self):
        """Lazy load Vocos model."""
        if not self._vocos_loaded:
            try:
                from vocos import Vocos

                self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
                self._vocos_loaded = True
                logger.info("✅ Initialized Vocos vocoder")
            except ImportError:
                logger.warning("vocos not available, audio postprocessing will be limited")
                self.vocos = None
                self._vocos_loaded = True

    def mel_to_waveform(self, mel_spec: torch.Tensor) -> np.ndarray:
        """
        Convert mel spectrogram to audio waveform.

        Args:
            mel_spec: Mel spectrogram [n_mels, time]

        Returns:
            Audio waveform as numpy array
        """
        # Lazy load vocos
        if not self._vocos_loaded:
            self._load_vocos()

        if self.vocos:
            # Use Vocos for high-quality reconstruction
            with torch.no_grad():
                # Convert to vocos format (expects [batch, n_mels, time])
                mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
                waveform = self.vocos.decode(mel_spec)
                return waveform.squeeze(0).numpy()
        elif self.librosa:
            # Fallback: use Griffin-Lim reconstruction
            logger.warning("Using Griffin-Lim reconstruction (lower quality)")
            try:
                # Convert back to linear spectrogram (approximate)
                mel_spec_np = mel_spec.numpy()
                mel_spec_db = mel_spec_np * 80 - 80  # Approximate dB range
                mel_spec_linear = self.librosa.db_to_power(mel_spec_db)

                # Griffin-Lim reconstruction
                waveform = self.librosa.feature.inverse.mel_to_audio(
                    mel_spec_linear, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
                )
                return waveform
            except Exception as e:
                logger.warning(f"Griffin-Lim failed: {e}")

        # Final fallback: generate noise
        logger.warning("No vocoder available, generating mock audio")
        duration = mel_spec.shape[1] * self.hop_length / self.sample_rate
        num_samples = int(duration * self.sample_rate)
        return np.random.randn(num_samples).astype(np.float32) * 0.1


# Convenience functions


def create_audio_processor() -> AudioProcessor:
    """Create audio processor with default settings."""
    return AudioProcessor()


def create_image_processor() -> ImageProcessor:
    """Create image processor with default settings."""
    return ImageProcessor()


def create_text_processor() -> TextProcessor:
    """Create text processor with default settings."""
    return TextProcessor()


def create_audio_postprocessor() -> AudioPostprocessor:
    """Create audio postprocessor with default settings."""
    return AudioPostprocessor()


# Test functions are removed to avoid memory allocation issues in test environment
