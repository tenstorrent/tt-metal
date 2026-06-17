"""
Real-time voice conversion inference using TTNN APIs.

This module provides a low-latency voice conversion pipeline that runs on
Tenstorrent hardware via the TTNN library. The pipeline includes audio
preprocessing (mel-spectrogram extraction), model inference on device, and
waveform reconstruction using a Griffin‑Lim or WaveGlow vocoder.

The inference class is designed for streaming: call ``convert_chunk()`` with
audio buffers as they arrive. Final conversion can be obtained via ``flush()``.

Usage::

    converter = VoiceConverter(
        model_path="path/to/model.pt",
        config_path="path/to/config.yaml",
        device_id=0
    )
    output_audio = converter.convert(input_audio, sr=16000)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import torch
import yaml

# TTNN library imports (hypothetical API, adjust to actual TTNN bindings)
try:
    import ttnn
    import ttnn.ops as tops
    from ttnn.device import Device
    from ttnn.tensor import Tensor as TTNNTensor
except ImportError:
    # Mock for development/testing without hardware
    class _TTNNTensor:
        pass

    class _Device:
        pass

    class _MockTTNN:
        @staticmethod
        def to_device(t: Any, device: Any) -> Any:
            return t

        @staticmethod
        def from_device(t: Any) -> Any:
            return t

        @staticmethod
        def to_torch(t: Any) -> Any:
            return t

    class _MockOps:
        @staticmethod
        def matmul(a: Any, b: Any) -> Any:
            return a @ b

    ttnn = _MockTTNN()
    tops = _MockOps()
    Device = _Device
    TTNNTensor = _TTNNTensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class VoiceConverterError(Exception):
    """Base exception for VoiceConverter errors."""


class VoiceConverterInitError(VoiceConverterError):
    """Raised when initialization fails."""


class VoiceConverterInferenceError(VoiceConverterError):
    """Raised during conversion failure."""


class VoiceConverterValidationError(VoiceConverterError):
    """Raised when input validation fails."""


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class VocoderType(Enum):
    """Supported vocoder types.

    Attributes:
        GRIFFIN_LIM: Classic Griffin-Lim algorithm (no learned model).
        WAVEGLOW: WaveGlow vocoder (requires external model file).
        MELGAN: Future melGAN support (placeholder).
        HIFIGAN: Future HiFi-GAN support (placeholder).
    """
    GRIFFIN_LIM = "griffin_lim"
    WAVEGLOW = "waveglow"
    MELGAN = "melgan"        # not yet implemented
    HIFIGAN = "hifi-gan"    # not yet implemented


class DeviceMemoryLayout(Enum):
    """TTNN memory layout options."""
    TILE = "TILE"
    ROW_MAJOR = "ROW_MAJOR"


class DeviceDataType(Enum):
    """TTNN data type options."""
    BFLOAT16 = "BFLOAT16"
    FLOAT32 = "FLOAT32"


@dataclass
class VoiceConverterConfig:
    """Configuration for voice conversion inference.

    Parameters
    ----------
    model_path : str
        Path to the trained voice conversion TorchScript model.
    config_path : str
        Path to YAML configuration for mel extraction and model.
    device_id : int
        TTNN device ID (0, 1, ...).
    sample_rate : int
        Audio sample rate (Hz).
    hop_length : int
        Hop length for STFT.
    win_length : int
        Window length for STFT.
    n_fft : int
        FFT size.
    n_mels : int
        Number of mel filterbanks.
    f_min : float
        Minimum frequency for mel filters.
    f_max : Optional[float]
        Maximum frequency for mel filters (None for Nyquist).
    power : float
        Power of the spectrogram (1.0 for magnitude).
    normalized : bool
        Whether to normalize mel spectrogram.
    center : bool
        Whether to center the FFT window.
    pad_mode : str
        Padding mode for STFT.
    vocoder_type : Union[str, VocoderType]
        Vocoder to use (griffin_lim or waveglow).
    vocoder_path : Optional[str]
        Path to WaveGlow model (if vocoder_type is waveglow).
    use_fp16 : bool
        Whether to use half precision on TTNN device.
    batch_size : int
        Batch size for inference (currently only 1).
    max_input_length : int
        Maximum number of samples to process in one chunk (e.g., 3 seconds at 16kHz).
    ttnn_layout : Union[str, DeviceMemoryLayout]
        TTNN tensor layout (e.g., "TILE").
    ttnn_dtype : Union[str, DeviceDataType]
        TTNN data type (e.g., "BFLOAT16").
    overlap_samples : int
        Number of overlapping samples for streaming (reduces boundary artifacts).
    enable_profiling : bool
        If True, log per-step timing.
    """

    # Required paths
    model_path: str = "models/llvc.pt"
    config_path: str = "configs/llvc.yaml"

    # Device settings
    device_id: int = 0
    use_fp16: bool = True
    batch_size: int = 1

    # Audio processing parameters
    sample_rate: int = 16000
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = 8000.0
    power: float = 1.0
    normalized: bool = True
    center: bool = True
    pad_mode: str = "reflect"

    # Vocoder
    vocoder_type: Union[str, VocoderType] = VocoderType.GRIFFIN_LIM
    vocoder_path: Optional[str] = None

    # Streaming / chunking
    max_input_length: int = 48000  # 3 seconds at 16kHz
    overlap_samples: int = 256

    # TTNN tensor options
    ttnn_layout: Union[str, DeviceMemoryLayout] = DeviceMemoryLayout.TILE
    ttnn_dtype: Union[str, DeviceDataType] = DeviceDataType.BFLOAT16

    # Diagnostics
    enable_profiling: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Raise VoiceConverterValidationError for any invalid config values.

        This method is idempotent.
        """
        # Validate model paths exist
        if not Path(self.model_path).is_file():
            raise VoiceConverterValidationError(
                f"Model file not found: {self.model_path}"
            )
        if not Path(self.config_path).is_file():
            raise VoiceConverterValidationError(
                f"Config file not found: {self.config_path}"
            )

        # Validate numeric parameters
        if self.sample_rate <= 0:
            raise VoiceConverterValidationError(
                f"Sample rate must be positive, got {self.sample_rate}"
            )
        if self.hop_length <= 0:
            raise VoiceConverterValidationError(
                f"Hop length must be positive, got {self.hop_length}"
            )
        if self.win_length <= 0:
            raise VoiceConverterValidationError(
                f"Window length must be positive, got {self.win_length}"
            )
        if self.n_fft <= 0:
            raise VoiceConverterValidationError(
                f"FFT size must be positive, got {self.n_fft}"
            )
        if self.n_mels <= 0:
            raise VoiceConverterValidationError(
                f"Number of mel bands must be positive, got {self.n_mels}"
            )
        if self.max_input_length <= 0:
            raise VoiceConverterValidationError(
                f"Max input length must be positive, got {self.max_input_length}"
            )
        if self.overlap_samples < 0:
            raise VoiceConverterValidationError(
                f"Overlap samples must be non-negative, got {self.overlap_samples}"
            )
        if self.batch_size <= 0:
            raise VoiceConverterValidationError(
                f"Batch size must be positive, got {self.batch_size}"
            )
        if self.device_id < 0:
            raise VoiceConverterValidationError(
                f"Device ID must be non-negative, got {self.device_id}"
            )

        # Validate vocoder type
        if isinstance(self.vocoder_type, str):
            try:
                self.vocoder_type = VocoderType(self.vocoder_type)
            except ValueError:
                valid_types = [v.value for v in VocoderType]
                raise VoiceConverterValidationError(
                    f"Invalid vocoder type '{self.vocoder_type}'. "
                    f"Valid options: {valid_types}"
                )
        if self.vocoder_type == VocoderType.WAVEGLOW and not self.vocoder_path:
            raise VoiceConverterValidationError(
                "vocoder_path must be provided when vocoder_type is WAVEGLOW"
            )
        if self.vocoder_type == VocoderType.GRIFFIN_LIM and self.vocoder_path:
            logger.warning("vocoder_path is ignored for Griffin-Lim vocoder")

        # Validate TTNN layout and dtype
        if isinstance(self.ttnn_layout, str):
            try:
                self.ttnn_layout = DeviceMemoryLayout(self.ttnn_layout)
            except ValueError:
                valid = [v.value for v in DeviceMemoryLayout]
                raise VoiceConverterValidationError(
                    f"Invalid ttnn_layout '{self.ttnn_layout}'. Valid: {valid}"
                )
        if isinstance(self.ttnn_dtype, str):
            try:
                self.ttnn_dtype = DeviceDataType(self.ttnn_dtype)
            except ValueError:
                valid = [v.value for v in DeviceDataType]
                raise VoiceConverterValidationError(
                    f"Invalid ttnn_dtype '{self.ttnn_dtype}'. Valid: {valid}"
                )

        # Validate pad_mode
        valid_pad_modes = {"constant", "reflect", "replicate", "circular"}
        if self.pad_mode not in valid_pad_modes:
            raise VoiceConverterValidationError(
                f"pad_mode '{self.pad_mode}' not supported. Valid: {valid_pad_modes}"
            )

        logger.info("VoiceConverterConfig validated successfully.")


# ---------------------------------------------------------------------------
# Helper: loading YAML config
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file safely.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary of configuration parameters.

    Raises:
        VoiceConverterValidationError: If file cannot be read or parsed.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise VoiceConverterValidationError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise VoiceConverterValidationError(
            f"Failed to parse YAML config {config_path}: {e}"
        ) from e
    except OSError as e:
        raise VoiceConverterValidationError(
            f"Failed to read config file {config_path}: {e}"
        ) from e

    if not isinstance(config, dict):
        raise VoiceConverterValidationError(
            f"YAML file {config_path} must contain a dictionary, got {type(config)}"
        )

    logger.debug(f"Loaded config from {config_path}: {len(config)} keys")
    return config


# ---------------------------------------------------------------------------
# Mel-spectrogram extractor (torch-based)
# ---------------------------------------------------------------------------

class MelSpectrogramExtractor:
    """Mel-spectrogram extraction from raw audio waveforms.

    Uses torchaudio's MelSpectrogram or a pure torch implementation
    for lower latency. Supports variable-length inputs.

    Parameters
    ----------
    config : VoiceConverterConfig
        Configuration with all audio processing parameters.
    """

    def __init__(self, config: VoiceConverterConfig) -> None:
        self._config = config
        self._build_transform()

    def _build_transform(self) -> None:
        """Build the mel spectrogram transform.

        Uses torchaudio if available; otherwise falls back to a torch-based
        implementation.
        """
        try:
            import torchaudio
            self._transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self._config.sample_rate,
                n_fft=self._config.n_fft,
                hop_length=self._config.hop_length,
                win_length=self._config.win_length,
                n_mels=self._config.n_mels,
                f_min=self._config.f_min,
                f_max=self._config.f_max,
                power=self._config.power,
                normalized=self._config.normalized,
                center=self._config.center,
                pad_mode=self._config.pad_mode,
            )
            self._to_db = torchaudio.transforms.AmplitudeToDB()
        except ImportError:
            logger.warning("torchaudio not available; using pure torch mel extractor (slower)")
            self._transform = self._torch_mel_spectrogram
            self._to_db = self._torch_amplitude_to_db

        # Move to device (CPU initially; actual device transfer handled separately)
        self._transform = self._transform.to("cpu")
        self._to_db = self._to_db.to("cpu")

    def _torch_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch mel spectrogram (fallback when torchaudio missing)."""
        # STFT
        spec = torch.stft(
            waveform,
            n_fft=self._config.n_fft,
            hop_length=self._config.hop_length,
            win_length=self._config.win_length,
            window=torch.hann_window(self._config.win_length),
            center=self._config.center,
            pad_mode=self._config.pad_mode,
            normalized=self._config.normalized,
            return_complex=True,
        )
        mag = torch.abs(spec) ** self._config.power if self._config.power != 1.0 else torch.abs(spec)

        # Mel filterbank
        mel_basis = self._create_mel_filterbank()
        mel = torch.matmul(mel_basis, mag)
        return mel

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix (approximation)."""
        # Placeholder: real implementation would use librosa or torchaudio
        n_freqs = self._config.n_fft // 2 + 1
        fb = torch.zeros(self._config.n_mels, n_freqs)
        # Simplified linear mel scale (not accurate)
        for i in range(self._config.n_mels):
            fb[i, i * n_freqs // self._config.n_mels] = 1.0
        return fb

    @staticmethod
    def _torch_amplitude_to_db(spec: torch.Tensor) -> torch.Tensor:
        """Convert amplitude spectrogram to dB scale."""
        return 20.0 * torch.log10(torch.clamp(spec, min=1e-7))

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from waveform.

        Args:
            waveform: Shape (batch, channels, samples) or (samples,).

        Returns:
            Mel spectrogram tensor of shape (batch, n_mels, time_frames) or (n_mels, time_frames).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1,1,T)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (batch, 1, T)

        mel = self._transform(waveform)
        mel = self._to_db(mel)
        return mel.squeeze(1)  # (batch, n_mels, T)


# ---------------------------------------------------------------------------
# Vocoder implementations
# ---------------------------------------------------------------------------

class BaseVocoder:
    """Base class for vocoder implementations."""

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize audio from mel spectrogram.

        Args:
            mel: Shape (batch, n_mels, T).

        Returns:
            Waveform tensor of shape (batch, samples).
        """
        raise NotImplementedError


class GriffinLimVocoder(BaseVocoder):
    """Griffin-Lim vocoder (iterative phase reconstruction)."""

    def __init__(self, config: VoiceConverterConfig) -> None:
        self._config = config
        self._n_iter: int = 32  # number of Griffin-Lim iterations

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize using Grffin-Lim algorithm.

        Args:
            mel: Mel spectrogram (batch, n_mels, T).

        Returns:
            Waveform tensor (batch, samples).
        """
        # Invert mel to linear spectrogram using pseudo-inverse of mel filterbank
        # For simplicity, assume mel basis is invertible (approximation)
        # Real use would use a neural vocoder; this is placeholder.
        logger.debug(f"Griffin-Lim synthesis: mel shape {mel.shape}")
        batch, n_mels, T = mel.shape
        # Placeholder: return random noise of appropriate length
        length = T * self._config.hop_length
        return torch.randn(batch, length)


class WaveGlowVocoder(BaseVocoder):
    """WaveGlow vocoder (flow-based neural vocoder)."""

    def __init__(self, config: VoiceConverterConfig) -> None:
        self._config = config
        self._model = self._load_model(config.vocoder_path)

    @staticmethod
    def _load_model(vocoder_path: Optional[str]) -> torch.nn.Module:
        """Load WaveGlow model from path."""
        if not vocoder_path:
            raise VoiceConverterValidationError("vocoder_path required for WaveGlow")
        path = Path(vocoder_path)
        if not path.is_file():
            raise VoiceConverterValidationError(f"WaveGlow model not found: {path}")
        model = torch.jit.load(str(path))
        model.eval()
        logger.info(f"Loaded WaveGlow model from {path}")
        return model

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        """Synthesize using WaveGlow.

        Args:
            mel: Mel spectrogram (batch, n_mels, T).

        Returns:
            Waveform tensor (batch, samples).
        """
        with torch.no_grad():
            audio = self._model(mel)
        return audio


# ---------------------------------------------------------------------------
# VoiceConverter class (main)
# ---------------------------------------------------------------------------

class VoiceConverter:
    """Real-time voice conversion inference engine using TTNN.

    The converter loads a voice conversion model (mel-to-mel or mel-to-voice)
    and a vocoder, then provides a streaming interface for low-latency
    inference on Tenstorrent hardware.

    Parameters
    ----------
    model_path : str
        Path to voice conversion TorchScript model.
    config_path : str
        Path to YAML config for audio parameters.
    device_id : int
        TTNN device ID.
    **kwargs : Any
        Additional overrides for VoiceConverterConfig.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device_id: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # Build config from default + file + overrides
        config = VoiceConverterConfig()
        # Override from file
        if config_path is not None:
            config.config_path = config_path
        if model_path is not None:
            config.model_path = model_path
        if device_id is not None:
            config.device_id = device_id

        # Load file config
        file_cfg = load_yaml_config(config.config_path)
        for key, value in file_cfg.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Apply constructor overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Ignoring unknown config key '{key}'")

        # Validate final config
        config._validate()

        self._config = config
        self._initialize_device()
        self._load_model()
        self._initialize_vocoder()
        self._initialize_mel_extractor()

        # Streaming state
        self._buffer: torch.Tensor = torch.tensor([], dtype=torch.float32)
        self._overlap_buffer: torch.Tensor = torch.tensor([], dtype=torch.float32)

        logger.info(
            f"VoiceConverter initialized (device={self._device}, "
            f"model={config.model_path}, vocoder={config.vocoder_type.value})"
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_device(self) -> None:
        """Initialize TTNN device.

        Raises:
            VoiceConverterInitError: If device cannot be opened.
        """
        try:
            self._device: Device = ttnn.open_device(self._config.device_id)
            logger.info(f"Opened TTNN device {self._device}")
        except Exception as e:
            raise VoiceConverterInitError(
                f"Failed to open TTNN device {self._config.device_id}: {e}"
            ) from e

    def _load_model(self) -> None:
        """Load and compile voice conversion model to TTNN device.

        Raises:
            VoiceConverterInitError: If model loading fails.
        """
        model_path = Path(self._config.model_path)
        if not model_path.is_file():
            raise VoiceConverterInitError(f"Model file not found: {model_path}")

        try:
            self._model: torch.nn.Module = torch.jit.load(str(model_path))
            self._model.eval()
            logger.info(f"Loaded TorchScript model from {model_path}")
        except Exception as e:
            raise VoiceConverterInitError(
                f"Failed to load model from {model_path}: {e}"
            ) from e

        # Move model to device (TTNN handles placement)
        # Placeholder: in real implementation, convert to TTNN tensor ops
        self._model = self._model.to("cpu")  # actual TTNN placement would differ

    def _initialize_vocoder(self) -> None:
        """Initialize vocoder based on config."""
        vtype = self._config.vocoder_type
        if vtype == VocoderType.GRIFFIN_LIM:
            self._vocoder: BaseVocoder = GriffinLimVocoder(self._config)
        elif vtype == VocoderType.WAVEGLOW:
            self._vocoder = WaveGlowVocoder(self._config)
        else:
            raise VoiceConverterInitError(f"Vocoder type {vtype} not implemented")

    def _initialize_mel_extractor(self) -> None:
        """Initialize mel spectrogram extraction."""
        self._mel_extractor = MelSpectrogramExtractor(self._config)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_audio_input(
        self, audio: Union[np.ndarray, torch.Tensor, List[float]], sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """Validate and convert audio input to torch tensor.

        Args:
            audio: Input audio as numpy array, list, or torch tensor.
            sample_rate: Expected sample rate.

        Returns:
            Float tensor of shape (1, T).

        Raises:
            VoiceConverterValidationError: On invalid input.
        """
        if isinstance(audio, np.ndarray):
            tensor = torch.from_numpy(audio).float()
        elif isinstance(audio, list):
            tensor = torch.tensor(audio, dtype=torch.float32)
        elif isinstance(audio, torch.Tensor):
            tensor = audio.float()
        else:
            raise VoiceConverterValidationError(
                f"Unsupported audio type: {type(audio)}"
            )

        # Ensure 1-D or 2-D
        if tensor.dim() == 0:
            raise VoiceConverterValidationError("Audio must be at least 1-D")
        if tensor.dim() > 2:
            raise VoiceConverterValidationError("Audio must be 1-D or 2-D")
        if tensor.dim() == 2 and tensor.size(0) > 1:
            # If multiple channels, take mean
            tensor = tensor.mean(dim=0, keepdim=True)

        # Check for NaN/Inf
        if not torch.isfinite(tensor).all():
            raise VoiceConverterValidationError("Audio contains NaN or Inf values")

        # Check sample rate if provided
        if sample_rate is not None and sample_rate != self._config.sample_rate:
            logger.warning(
                f"Input sample rate {sample_rate} differs from config {self._config.sample_rate}; "
                f"resampling not implemented – using as-is"
            )

        # Ensure shape (1, T)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # (1, T)

        # Clamp length
        if tensor.size(-1) > self._config.max_input_length:
            logger.warning(
                f"Input length {tensor.size(-1)} exceeds max_input_length "
                f"{self._config.max_input_length}; truncating"
            )
            tensor = tensor[:, :self._config.max_input_length]

        return tensor

    # ------------------------------------------------------------------
    # Core conversion methods
    # ------------------------------------------------------------------

    def convert(
        self,
        audio: Union[np.ndarray, torch.Tensor, List[float]],
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Perform full voice conversion on a complete audio buffer.

        This method processes the entire audio at once. For streaming use
        ``convert_chunk()`` and ``flush()``.

        Args:
            audio: Input audio waveform (1-D or 2-D).
            sample_rate: Optional sample rate for validation.

        Returns:
            Converted audio as (T,) numpy array.

        Raises:
            VoiceConverterInferenceError: If inference fails.
        """
        audio_tensor = self._validate_audio_input(audio, sample_rate)
        logger.info(f"Full conversion: input shape {audio_tensor.shape}")

        try:
            # 1. Compute mel spectrogram
            mel = self._mel_extractor.extract(audio_tensor)  # (1, n_mels, T)

            # 2. Move mel to device and run model
            mel_device = ttnn.to_device(mel, self._device)
            with torch.no_grad():
                if self._config.enable_profiling:
                    t0 = time.perf_counter()
                output_mel = self._model(mel_device)
                if self._config.enable_profiling:
                    t1 = time.perf_counter()
                    logger.info(f"Model inference took {t1-t0:.4f}s")
            output_mel = ttnn.from_device(output_mel)

            # 3. Synthesize waveform
            audio_out = self._vocoder.synthesize(output_mel)  # (1, samples)

            return audio_out.squeeze(0).numpy()
        except Exception as e:
            raise VoiceConverterInferenceError(
                f"Conversion failed: {e}"
            ) from e

    def convert_chunk(
        self,
        chunk: Union[np.ndarray, torch.Tensor, List[float]],
        sample_rate: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Process a chunk of audio in streaming fashion.

        Call this method with audio buffers as they arrive. The method buffers
        audio internally and returns a converted chunk when enough data is
        accumulated. Call ``flush()`` at the end to get remaining audio.

        Args:
            chunk: Audio chunk (1-D).
            sample_rate: Optional sample rate.

        Returns:
            Converted audio chunk as numpy array, or None if insufficient data.

        Raises:
            VoiceConverterInferenceError: If inference fails.
        """
        chunk_tensor = self._validate_audio_input(chunk, sample_rate)
        # Append to buffer
        self._buffer = torch.cat([self._buffer, chunk_tensor], dim=-1)

        # Process only if buffer length >= max_input_length
        if self._buffer.size(-1) < self._config.max_input_length:
            return None

        # Take a chunk with overlap
        take_len = self._config.max_input_length
        if self._overlap_buffer.numel() > 0:
            # Concatenate with previous overlap
            combined = torch.cat([self._overlap_buffer, self._buffer], dim=-1)
            take_len = min(combined.size(-1), self._config.max_input_length)
            chunk_to_process = combined[:, -take_len:]
        else:
            chunk_to_process = self._buffer[:, -take_len:]

        # Save overlap for next
        overlap_len = self._config.overlap_samples
        self._overlap_buffer = chunk_to_process[:, -overlap_len:].clone()

        # Remove processed samples from buffer (except overlap)
        self._buffer = self._buffer[:, :-overlap_len] if overlap_len > 0 else torch.tensor([])

        # Convert this chunk
        converted = self._convert_chunk(chunk_to_process)
        return converted

    def _convert_chunk(self, audio_chunk: torch.Tensor) -> Optional[np.ndarray]:
        """Internal chunk conversion (same logic as convert)."""
        try:
            mel = self._mel_extractor.extract(audio_chunk)
            mel_device = ttnn.to_device(mel, self._device)
            with torch.no_grad():
                if self._config.enable_profiling:
                    t_start = time.perf_counter()
                out_mel = self._model(mel_device)
                if self._config.enable_profiling:
                    logger.debug(f"Chunk model inference: {time.perf_counter()-t_start:.4f}s")
            out_mel = ttnn.from_device(out_mel)
            audio_out = self._vocoder.synthesize(out_mel)
            return audio_out.squeeze(0).numpy()
        except Exception as e:
            raise VoiceConverterInferenceError(
                f"Chunk conversion failed: {e}"
            ) from e

    def flush(self) -> Optional[np.ndarray]:
        """Flush the streaming buffer and return remaining converted audio.

        Call after all chunks have been fed via ``convert_chunk()``.

        Returns:
            Converted audio for the remaining buffer, or None if buffer empty.
        """
        if self._buffer.numel() == 0:
            return None
        # Process remaining buffer
        result = self._convert_chunk(self._buffer)
        self._buffer = torch.tensor([], dtype=torch.float32)
        self._overlap_buffer = torch.tensor([], dtype=torch.float32)
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release TTNN device and cleanup resources."""
        try:
            ttnn.close_device(self._device)
            logger.info("TTNN device closed")
        except Exception as e:
            logger.error(f"Error closing device: {e}")
        self._model = None
        self._vocoder = None
        self._buffer = torch.tensor([])
        self._overlap_buffer = torch.tensor([])

    def __enter__(self) -> "VoiceConverter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Main guard (example usage)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Example config
    cfg = VoiceConverterConfig(
        model_path="models/llvc.pt",
        config_path="configs/llvc.yaml",
        device_id=0,
        vocoder_type="griffin_lim",
        enable_profiling=True,
    )

    try:
        converter = VoiceConverter(
            model_path=cfg.model_path,
            config_path=cfg.config_path,
            device_id=cfg.device_id,
        )
        # Dummy audio: 1 second of silence
        dummy_audio = np.zeros(16000, dtype=np.float32)
        out = converter.convert(dummy_audio, sample_rate=16000)
        print(f"Output length: {len(out)} samples")

        # Streaming test
        for i in range(10):
            chunk = np.random.randn(1600).astype(np.float32)
            result = converter.convert_chunk(chunk)
            if result is not None:
                print(f"Chunk {i}: {len(result)} samples")
        final = converter.flush()
        if final is not None:
            print(f"Flush: {len(final)} samples")

        converter.close()
    except VoiceConverterError as e:
        logger.error(f"Voice conversion failed: {e}")
    except Exception as e:
        logger.exception("Unexpected error")