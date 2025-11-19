# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Hybrid Pipeline Utilities for MiniCPM-o-2_6

Utilities for seamless tensor conversions between PyTorch and TTNN
in the hybrid pipeline that uses PyTorch for Qwen and TTNN for other components.
"""

import torch
import ttnn
from typing import Union, Optional
from loguru import logger
import numpy as np
from pathlib import Path

LIBROSA_AVAILABLE = False
TORCHVISION_AVAILABLE = False
Image = None
T = None
librosa = None


def pytorch_to_ttnn(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Convert PyTorch tensor to TTNN tensor with proper formatting.

    Args:
        tensor: PyTorch tensor [batch, seq, hidden] or [batch, seq]
        device: TTNN device
        dtype: Target data type (default: bfloat16)
        layout: Target layout (default: TILE_LAYOUT)
        memory_config: Memory configuration (default: None for DRAM)

    Returns:
        TTNN tensor on device
    """
    # Ensure tensor is contiguous and on CPU
    tensor = tensor.contiguous().cpu()

    # Convert to TTNN
    ttnn_tensor = ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

    return ttnn_tensor


def ttnn_to_pytorch(
    tensor: ttnn.Tensor,
    dtype: torch.dtype = torch.float32,
    mesh_device: Optional[ttnn.Device] = None,
    expected_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert TTNN tensor back to PyTorch tensor.

    Args:
        tensor: TTNN tensor
        dtype: Target PyTorch dtype (default: float32)

    Returns:
        PyTorch tensor on CPU
    """
    # If the tensor is sharded across a mesh, attempt to compose it back to a single
    # PyTorch tensor. Be replica-aware: Gemma and some helpers replicate weights/outputs
    # across devices, which can make ConcatMeshToTensor produce concatenated replicas
    # (e.g., doubling the channel dimension). Try multiple composers and if the
    # composed tensor appears to be replicated, slice down to a single replica.
    pytorch_tensor = None
    if mesh_device is not None:
        num_devices = None
        try:
            num_devices = int(mesh_device.get_num_devices())
        except Exception:
            num_devices = None

        # Candidate concat dims to try (common values observed in codebase)
        candidate_dims = [-1, 0, 1, 2, 3]
        last_exception = None
        ttnn_shape = None
        try:
            ttnn_shape = tuple(tensor.shape)
        except Exception:
            ttnn_shape = None

        for dim in candidate_dims:
            try:
                composer = ttnn.ConcatMeshToTensor(mesh_device, dim=dim)
                cand = ttnn.to_torch(tensor, mesh_composer=composer)
                # If dtype conversion requested, apply later
                cand_shape = tuple(cand.shape)
                # If we have the original TTNN logical shape, prefer exact match
                if ttnn_shape is not None and cand_shape == ttnn_shape:
                    pytorch_tensor = cand
                    break

                # If composer created replicated concatenation: detect axis where size is multiple of expected
                if ttnn_shape is not None:
                    # Align from the right (trailing dims). Detect if any trailing dim in cand is an integer
                    # multiple of the corresponding ttnn dim (replicated concatenation), and slice it down.
                    matched = False
                    for i in range(1, min(len(cand_shape), len(ttnn_shape)) + 1):
                        cand_axis = -i
                        ttnn_axis = -i
                        try:
                            t_dim = ttnn_shape[ttnn_axis]
                            c_dim = cand_shape[cand_axis]
                            # If expected_dim provided, prefer slicing down to expected_dim
                            if expected_dim is not None and c_dim % expected_dim == 0 and (c_dim // expected_dim) > 1:
                                slices = [slice(None)] * len(cand_shape)
                                slices[cand_axis] = slice(0, expected_dim)
                                pytorch_tensor = cand[tuple(slices)]
                                logger.debug(
                                    f"ttnn_to_pytorch: sliced to expected_dim on axis {cand_axis}, result {pytorch_tensor.shape}"
                                )
                                matched = True
                                break
                            if t_dim > 0 and c_dim % t_dim == 0 and (c_dim // t_dim) > 1:
                                # Slice to a single replica along this axis (take first portion)
                                slices = [slice(None)] * len(cand_shape)
                                slices[cand_axis] = slice(0, t_dim)
                                pytorch_tensor = cand[tuple(slices)]
                                logger.debug(
                                    f"ttnn_to_pytorch: detected replicated concat on axis {cand_axis}, sliced to {pytorch_tensor.shape}"
                                )
                                matched = True
                                break
                        except Exception:
                            continue
                    if matched:
                        break

                # If no ttnn_shape to compare against, heuristically prefer tensors where
                # one dimension equals tensor.shape[-1] or has expected layout. Accept first.
                if pytorch_tensor is None:
                    pytorch_tensor = cand
                    break
            except Exception as exc:
                last_exception = exc
                logger.debug(f"ttnn_to_pytorch: ConcatMeshToTensor dim={dim} failed: {exc}")

        # If all composers failed, fall back to plain conversion (may raise)
        if pytorch_tensor is None:
            if last_exception is not None:
                logger.debug(f"ttnn_to_pytorch: all composers failed, last error: {last_exception}")
            pytorch_tensor = ttnn.to_torch(tensor)
    else:
        # No mesh device provided, do direct conversion
        pytorch_tensor = ttnn.to_torch(tensor)

    # Convert to target dtype
    if dtype != pytorch_tensor.dtype:
        pytorch_tensor = pytorch_tensor.to(dtype)

    return pytorch_tensor


def validate_tensor_shapes(pytorch_tensor: torch.Tensor, ttnn_tensor: ttnn.Tensor, name: str = "tensor") -> bool:
    """
    Validate that PyTorch and TTNN tensors have compatible shapes.

    Args:
        pytorch_tensor: PyTorch tensor
        ttnn_tensor: TTNN tensor
        name: Name for logging

    Returns:
        True if shapes are compatible
    """
    pytorch_shape = pytorch_tensor.shape
    ttnn_shape = ttnn_tensor.shape

    # Handle different shape representations
    if len(ttnn_shape) == 4 and len(pytorch_shape) == 3:
        # TTNN: [1, batch*seq, hidden, 1] -> PyTorch: [batch, seq, hidden]
        expected_ttnn_shape = (1, pytorch_shape[0] * pytorch_shape[1], pytorch_shape[2], 1)
        if ttnn_shape == expected_ttnn_shape:
            return True
    elif len(ttnn_shape) == 3 and len(pytorch_shape) == 3:
        # Direct 3D match
        if ttnn_shape == pytorch_shape:
            return True

    logger.warning(f"Shape mismatch for {name}: PyTorch {pytorch_shape} vs TTNN {ttnn_shape}")
    return False


def convert_multimodal_embeddings_to_qwen_format(
    encoder_hidden_states: Union[torch.Tensor, ttnn.Tensor], device: Optional[ttnn.Device] = None
) -> torch.Tensor:
    """
    Convert multimodal embeddings (vision/audio) to format expected by PyTorch Qwen.

    The Qwen cross-attention expects [batch, seq, hidden] format.

    Args:
        encoder_hidden_states: Multimodal embeddings from TTNN components
        device: TTNN device (only needed if input is TTNN tensor)

    Returns:
        PyTorch tensor in [batch, seq, hidden] format for Qwen
    """
    if isinstance(encoder_hidden_states, ttnn.Tensor):
        # Convert TTNN -> PyTorch
        embeddings = ttnn_to_pytorch(encoder_hidden_states)
    else:
        embeddings = encoder_hidden_states

    # Ensure 3D format [batch, seq, hidden]
    if len(embeddings.shape) == 4:
        # TTNN format [1, batch*seq, hidden, 1] -> [batch, seq, hidden]
        batch_seq, hidden = embeddings.shape[1], embeddings.shape[2]
        # Assume batch=1 for now (can be extended)
        batch_size = 1
        seq_len = batch_seq // batch_size
        embeddings = embeddings.view(batch_size, seq_len, hidden).squeeze(0)
    elif len(embeddings.shape) == 2:
        # [seq, hidden] -> [1, seq, hidden]
        embeddings = embeddings.unsqueeze(0)

    return embeddings


def convert_qwen_hidden_states_to_ttnn_format(
    hidden_states: torch.Tensor, device: ttnn.Device, dtype: ttnn.DataType = ttnn.bfloat16
) -> ttnn.Tensor:
    """
    Convert Qwen hidden states to format expected by TTNN TTS components.

    TTNN components (ChatTTS, DVAE) expect specific tensor formats.

    Args:
        hidden_states: PyTorch tensor from Qwen [batch, seq, hidden]
        device: TTNN device
        dtype: Target TTNN dtype

    Returns:
        TTNN tensor in format expected by downstream components
    """
    # For ChatTTS decoder, we need [batch, seq, hidden] format
    # TTNN linear layers often expect 4D tensors: [batch, 1, seq, hidden]
    if len(hidden_states.shape) == 3:
        # Add dimension for TTNN: [batch, seq, hidden] -> [batch, 1, seq, hidden]
        hidden_states_4d = hidden_states.unsqueeze(1)
    else:
        hidden_states_4d = hidden_states

    # Convert to TTNN
    ttnn_hidden = pytorch_to_ttnn(hidden_states_4d, device, dtype=dtype)

    return ttnn_hidden


def prepare_audio_input_for_whisper(audio_tensor: torch.Tensor, device: ttnn.Device) -> torch.Tensor:
    """
    Prepare audio input for TTNN Whisper encoder.

    Args:
        audio_tensor: Raw audio tensor [batch, samples] or [samples]
        device: TTNN device

    Returns:
        Torch tensor ready for Whisper encoder (mel spectrograms)
    """
    # Whisper expects mel spectrograms (time-major), compute if raw audio provided.
    # Prefer librosa when available.
    # Check if input is already a mel spectrogram [batch, 80, time_steps] or [80, time_steps]
    if isinstance(audio_tensor, torch.Tensor) and len(audio_tensor.shape) == 3 and audio_tensor.shape[1] == 80:
        return audio_tensor.transpose(1, 2)  # [batch, time_steps, 80]
    if isinstance(audio_tensor, torch.Tensor) and len(audio_tensor.shape) == 2 and audio_tensor.shape[0] == 80:
        return audio_tensor.transpose(0, 1).unsqueeze(0)  # [1, time_steps, 80]

    # Lazy-import librosa to avoid heavy imports at module load time
    global librosa, LIBROSA_AVAILABLE
    try:
        import librosa as _librosa

        librosa = _librosa
        LIBROSA_AVAILABLE = True
    except Exception:
        LIBROSA_AVAILABLE = False
    if not LIBROSA_AVAILABLE:
        logger.warning(
            "prepare_audio_input_for_whisper: librosa not available - using deterministic dummy mel spectrogram"
        )
    if not LIBROSA_AVAILABLE:
        # deterministic dummy mel so tests are reproducible
        if isinstance(audio_tensor, torch.Tensor):
            batch_size = audio_tensor.shape[0] if audio_tensor.dim() > 1 else 1
        else:
            batch_size = 1
        num_samples = (
            audio_tensor.shape[1] if isinstance(audio_tensor, torch.Tensor) and audio_tensor.dim() > 1 else 16000
        )
        time_steps = max(1, num_samples // 320)
        mel_spec = torch.zeros(batch_size, time_steps, 80, dtype=torch.float32)
        return mel_spec

    # Convert input to numpy and handle batch
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio_tensor)

    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, 0)

    batch_mels = []
    for wav in audio_np:
        # librosa expects float32
        wav = wav.astype(np.float32)
        # Resample if needed (assume input at 16000 is desired)
        try:
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=16000,
                n_fft=400,
                hop_length=160,
                win_length=400,
                n_mels=80,
                window="hann",
                power=2.0,
                center=True,
                pad_mode="reflect",
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            # Normalize to 0..1
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
            # mel_norm shape: [80, time_steps] -> transpose to [time_steps, 80]
            batch_mels.append(torch.from_numpy(mel_norm).float().transpose(0, 1))
        except Exception as exc:
            logger.warning(f"librosa mel computation failed: {exc}. Using zeros mel.")
            batch_mels.append(torch.zeros(1, 80, dtype=torch.float32).transpose(0, 1))

    # Stack to [batch, time_steps, 80]
    mel_batch = torch.nn.utils.rnn.pad_sequence(batch_mels, batch_first=True)
    return mel_batch


def prepare_image_input_for_siglip(image_tensor: torch.Tensor, device: ttnn.Device) -> torch.Tensor:
    """
    Prepare image input for TTNN SigLip encoder.

    Args:
        image_tensor: Image tensor [batch, channels, height, width]
        device: TTNN device

    Returns:
        Torch tensor ready for SigLip encoder (preprocessing happens in encoder)
    """
    # Convert raw image tensor or PIL.Image to SigLip input tensor.
    # If torchvision is available, use standard transforms (resize, center crop, normalize).
    # Lazy-import torchvision/PIL to avoid heavy imports at module load time
    global Image, T, TORCHVISION_AVAILABLE
    try:
        from PIL import Image as _Image
        import torchvision.transforms as _T

        Image = _Image
        T = _T
        TORCHVISION_AVAILABLE = True
    except Exception:
        TORCHVISION_AVAILABLE = False

    if TORCHVISION_AVAILABLE:
        logger.debug("prepare_image_input_for_siglip: Using torchvision preprocessing")
        transform = T.Compose(
            [
                T.ConvertImageDtype(torch.float32) if hasattr(T, "ConvertImageDtype") else T.ToTensor(),
                T.Resize(256),
                T.CenterCrop(224),
                # ensure tensor shape [C, H, W] in 0..1 then normalize
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # If input is a torch tensor [B, C, H, W] assume range 0..1 or -1..1; convert appropriately
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:
                # apply transforms per-image
                processed = []
                for i in range(image_tensor.shape[0]):
                    img = image_tensor[i]
                    # Ensure float in 0..1
                    if img.dtype != torch.float32:
                        img = img.float()
                    if img.min() < 0:
                        img = (img + 1.0) / 2.0
                    # Resize/normalize: convert to PIL for consistency
                    try:
                        pil = T.ToPILImage()(img.clamp(0, 1))
                        proc = transform(pil)
                    except Exception:
                        proc = T.CenterCrop(224)(T.Resize(256)(img))
                        proc = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(proc)
                    processed.append(proc.unsqueeze(0))
                out = torch.cat(processed, dim=0)
                return out
            elif image_tensor.dim() == 3:
                img = image_tensor
                if img.dtype != torch.float32:
                    img = img.float()
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                pil = T.ToPILImage()(img.clamp(0, 1))
                return transform(pil).unsqueeze(0)

    # Fallback: if PIL available and input is path-like or PIL.Image
    if Image is not None:
        try:
            if isinstance(image_tensor, (str, Path)):
                img = Image.open(str(image_tensor)).convert("RGB")
            elif isinstance(image_tensor, Image.Image):
                img = image_tensor.convert("RGB")
            else:
                # If it's a numpy array
                if isinstance(image_tensor, np.ndarray):
                    img = Image.fromarray(image_tensor.astype("uint8")).convert("RGB")
                else:
                    # As last resort, return input tensor as-is
                    logger.warning("prepare_image_input_for_siglip: unknown image input type, returning as-is")
                    return image_tensor

            # Basic resize/crop/normalize pipeline
            img = img.resize((256, 256))
            img = img.crop((16, 16, 240, 240))  # center crop 224
            arr = np.asarray(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W]
            # Normalize using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return tensor.unsqueeze(0)
        except Exception as exc:
            logger.warning(f"prepare_image_input_for_siglip fallback failed: {exc}. Returning original tensor")
            return image_tensor

    # If nothing else worked, return as-is
    logger.warning("prepare_image_input_for_siglip: torchvision/PIL not available - returning original tensor")
    return image_tensor


def log_tensor_info(tensor: Union[torch.Tensor, ttnn.Tensor], name: str):
    """
    Log tensor information for debugging.

    Args:
        tensor: Tensor to log
        name: Name for logging
    """
    if isinstance(tensor, torch.Tensor):
        logger.info(f"{name}: PyTorch shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    elif isinstance(tensor, ttnn.Tensor):
        logger.info(f"{name}: TTNN shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        logger.info(f"{name}: Unknown tensor type {type(tensor)}")


"""
Hybrid Pipeline Utilities for MiniCPM-o-2_6

Utilities for seamless tensor conversions between PyTorch and TTNN
in the hybrid pipeline that uses PyTorch for Qwen and TTNN for other components.
"""

import torch
import ttnn
from typing import Union, Optional
from loguru import logger
import numpy as np
from pathlib import Path

LIBROSA_AVAILABLE = False
TORCHVISION_AVAILABLE = False
Image = None
T = None
librosa = None


def pytorch_to_ttnn(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Convert PyTorch tensor to TTNN tensor with proper formatting.

    Args:
        tensor: PyTorch tensor [batch, seq, hidden] or [batch, seq]
        device: TTNN device
        dtype: Target data type (default: bfloat16)
        layout: Target layout (default: TILE_LAYOUT)
        memory_config: Memory configuration (default: None for DRAM)

    Returns:
        TTNN tensor on device
    """
    # Ensure tensor is contiguous and on CPU
    tensor = tensor.contiguous().cpu()

    # Convert to TTNN
    ttnn_tensor = ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)

    return ttnn_tensor


def validate_tensor_shapes(pytorch_tensor: torch.Tensor, ttnn_tensor: ttnn.Tensor, name: str = "tensor") -> bool:
    """
    Validate that PyTorch and TTNN tensors have compatible shapes.

    Args:
        pytorch_tensor: PyTorch tensor
        ttnn_tensor: TTNN tensor
        name: Name for logging

    Returns:
        True if shapes are compatible
    """
    pytorch_shape = pytorch_tensor.shape
    ttnn_shape = ttnn_tensor.shape

    # Handle different shape representations
    if len(ttnn_shape) == 4 and len(pytorch_shape) == 3:
        # TTNN: [1, batch*seq, hidden, 1] -> PyTorch: [batch, seq, hidden]
        expected_ttnn_shape = (1, pytorch_shape[0] * pytorch_shape[1], pytorch_shape[2], 1)
        if ttnn_shape == expected_ttnn_shape:
            return True
    elif len(ttnn_shape) == 3 and len(pytorch_shape) == 3:
        # Direct 3D match
        if ttnn_shape == pytorch_shape:
            return True

    logger.warning(f"Shape mismatch for {name}: PyTorch {pytorch_shape} vs TTNN {ttnn_shape}")
    return False


def convert_multimodal_embeddings_to_qwen_format(
    encoder_hidden_states: Union[torch.Tensor, ttnn.Tensor], device: Optional[ttnn.Device] = None
) -> torch.Tensor:
    """
    Convert multimodal embeddings (vision/audio) to format expected by PyTorch Qwen.

    The Qwen cross-attention expects [batch, seq, hidden] format.

    Args:
        encoder_hidden_states: Multimodal embeddings from TTNN components
        device: TTNN device (only needed if input is TTNN tensor)

    Returns:
        PyTorch tensor in [batch, seq, hidden] format for Qwen
    """
    if isinstance(encoder_hidden_states, ttnn.Tensor):
        # Convert TTNN -> PyTorch
        embeddings = ttnn_to_pytorch(encoder_hidden_states)
    else:
        embeddings = encoder_hidden_states

    # Ensure 3D format [batch, seq, hidden]
    if len(embeddings.shape) == 4:
        # TTNN format [1, batch*seq, hidden, 1] -> [batch, seq, hidden]
        batch_seq, hidden = embeddings.shape[1], embeddings.shape[2]
        # Assume batch=1 for now (can be extended)
        batch_size = 1
        seq_len = batch_seq // batch_size
        embeddings = embeddings.view(batch_size, seq_len, hidden).squeeze(0)
    elif len(embeddings.shape) == 2:
        # [seq, hidden] -> [1, seq, hidden]
        embeddings = embeddings.unsqueeze(0)

    return embeddings


def convert_qwen_hidden_states_to_ttnn_format(
    hidden_states: torch.Tensor, device: ttnn.Device, dtype: ttnn.DataType = ttnn.bfloat16
) -> ttnn.Tensor:
    """
    Convert Qwen hidden states to format expected by TTNN TTS components.

    TTNN components (ChatTTS, DVAE) expect specific tensor formats.

    Args:
        hidden_states: PyTorch tensor from Qwen [batch, seq, hidden]
        device: TTNN device
        dtype: Target TTNN dtype

    Returns:
        TTNN tensor in format expected by downstream components
    """
    # For ChatTTS decoder, we need [batch, seq, hidden] format
    # TTNN linear layers often expect 4D tensors: [batch, 1, seq, hidden]
    if len(hidden_states.shape) == 3:
        # Add dimension for TTNN: [batch, seq, hidden] -> [batch, 1, seq, hidden]
        hidden_states_4d = hidden_states.unsqueeze(1)
    else:
        hidden_states_4d = hidden_states

    # Convert to TTNN
    ttnn_hidden = pytorch_to_ttnn(hidden_states_4d, device, dtype=dtype)

    return ttnn_hidden


def prepare_audio_input_for_whisper(audio_tensor: torch.Tensor, device: ttnn.Device) -> torch.Tensor:
    """
    Prepare audio input for TTNN Whisper encoder.

    Args:
        audio_tensor: Raw audio tensor [batch, samples] or [samples]
        device: TTNN device

    Returns:
        Torch tensor ready for Whisper encoder (mel spectrograms)
    """
    # Whisper expects mel spectrograms (time-major), compute if raw audio provided.
    # Prefer librosa when available.
    # Check if input is already a mel spectrogram [batch, 80, time_steps] or [80, time_steps]
    if isinstance(audio_tensor, torch.Tensor) and len(audio_tensor.shape) == 3 and audio_tensor.shape[1] == 80:
        return audio_tensor.transpose(1, 2)  # [batch, time_steps, 80]
    if isinstance(audio_tensor, torch.Tensor) and len(audio_tensor.shape) == 2 and audio_tensor.shape[0] == 80:
        return audio_tensor.transpose(0, 1).unsqueeze(0)  # [1, time_steps, 80]

    # Lazy-import librosa to avoid heavy imports at module load time
    global librosa, LIBROSA_AVAILABLE
    try:
        import librosa as _librosa

        librosa = _librosa
        LIBROSA_AVAILABLE = True
    except Exception:
        LIBROSA_AVAILABLE = False
    if not LIBROSA_AVAILABLE:
        logger.warning(
            "prepare_audio_input_for_whisper: librosa not available - using deterministic dummy mel spectrogram"
        )
    if not LIBROSA_AVAILABLE:
        # deterministic dummy mel so tests are reproducible
        if isinstance(audio_tensor, torch.Tensor):
            batch_size = audio_tensor.shape[0] if audio_tensor.dim() > 1 else 1
        else:
            batch_size = 1
        num_samples = (
            audio_tensor.shape[1] if isinstance(audio_tensor, torch.Tensor) and audio_tensor.dim() > 1 else 16000
        )
        time_steps = max(1, num_samples // 320)
        mel_spec = torch.zeros(batch_size, time_steps, 80, dtype=torch.float32)
        return mel_spec

    # Convert input to numpy and handle batch
    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio_tensor)

    if audio_np.ndim == 1:
        audio_np = np.expand_dims(audio_np, 0)

    batch_mels = []
    for wav in audio_np:
        # librosa expects float32
        wav = wav.astype(np.float32)
        # Resample if needed (assume input at 16000 is desired)
        try:
            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=16000,
                n_fft=400,
                hop_length=160,
                win_length=400,
                n_mels=80,
                window="hann",
                power=2.0,
                center=True,
                pad_mode="reflect",
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            # Normalize to 0..1
            mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
            # mel_norm shape: [80, time_steps] -> transpose to [time_steps, 80]
            batch_mels.append(torch.from_numpy(mel_norm).float().transpose(0, 1))
        except Exception as exc:
            logger.warning(f"librosa mel computation failed: {exc}. Using zeros mel.")
            batch_mels.append(torch.zeros(1, 80, dtype=torch.float32).transpose(0, 1))

    # Stack to [batch, time_steps, 80]
    mel_batch = torch.nn.utils.rnn.pad_sequence(batch_mels, batch_first=True)
    return mel_batch


def prepare_image_input_for_siglip(image_tensor: torch.Tensor, device: ttnn.Device) -> torch.Tensor:
    """
    Prepare image input for TTNN SigLip encoder.

    Args:
        image_tensor: Image tensor [batch, channels, height, width]
        device: TTNN device

    Returns:
        Torch tensor ready for SigLip encoder (preprocessing happens in encoder)
    """
    # Convert raw image tensor or PIL.Image to SigLip input tensor.
    # If torchvision is available, use standard transforms (resize, center crop, normalize).
    # Lazy-import torchvision/PIL to avoid heavy imports at module load time
    global Image, T, TORCHVISION_AVAILABLE
    try:
        from PIL import Image as _Image
        import torchvision.transforms as _T

        Image = _Image
        T = _T
        TORCHVISION_AVAILABLE = True
    except Exception:
        TORCHVISION_AVAILABLE = False

    if TORCHVISION_AVAILABLE:
        logger.debug("prepare_image_input_for_siglip: Using torchvision preprocessing")
        transform = T.Compose(
            [
                T.ConvertImageDtype(torch.float32) if hasattr(T, "ConvertImageDtype") else T.ToTensor(),
                T.Resize(256),
                T.CenterCrop(224),
                # ensure tensor shape [C, H, W] in 0..1 then normalize
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # If input is a torch tensor [B, C, H, W] assume range 0..1 or -1..1; convert appropriately
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:
                # apply transforms per-image
                processed = []
                for i in range(image_tensor.shape[0]):
                    img = image_tensor[i]
                    # Ensure float in 0..1
                    if img.dtype != torch.float32:
                        img = img.float()
                    if img.min() < 0:
                        img = (img + 1.0) / 2.0
                    # Resize/normalize: convert to PIL for consistency
                    try:
                        pil = T.ToPILImage()(img.clamp(0, 1))
                        proc = transform(pil)
                    except Exception:
                        proc = T.CenterCrop(224)(T.Resize(256)(img))
                        proc = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(proc)
                    processed.append(proc.unsqueeze(0))
                out = torch.cat(processed, dim=0)
                return out
            elif image_tensor.dim() == 3:
                img = image_tensor
                if img.dtype != torch.float32:
                    img = img.float()
                if img.min() < 0:
                    img = (img + 1.0) / 2.0
                pil = T.ToPILImage()(img.clamp(0, 1))
                return transform(pil).unsqueeze(0)

    # Fallback: if PIL available and input is path-like or PIL.Image
    if Image is not None:
        try:
            if isinstance(image_tensor, (str, Path)):
                img = Image.open(str(image_tensor)).convert("RGB")
            elif isinstance(image_tensor, Image.Image):
                img = image_tensor.convert("RGB")
            else:
                # If it's a numpy array
                if isinstance(image_tensor, np.ndarray):
                    img = Image.fromarray(image_tensor.astype("uint8")).convert("RGB")
                else:
                    # As last resort, return input tensor as-is
                    logger.warning("prepare_image_input_for_siglip: unknown image input type, returning as-is")
                    return image_tensor

            # Basic resize/crop/normalize pipeline
            img = img.resize((256, 256))
            img = img.crop((16, 16, 240, 240))  # center crop 224
            arr = np.asarray(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C,H,W]
            # Normalize using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return tensor.unsqueeze(0)
        except Exception as exc:
            logger.warning(f"prepare_image_input_for_siglip fallback failed: {exc}. Returning original tensor")
            return image_tensor

    # If nothing else worked, return as-is
    logger.warning("prepare_image_input_for_siglip: torchvision/PIL not available - returning original tensor")
    return image_tensor


def log_tensor_info(tensor: Union[torch.Tensor, ttnn.Tensor], name: str):
    """
    Log tensor information for debugging.

    Args:
        tensor: Tensor to log
        name: Name for logging
    """
    if isinstance(tensor, torch.Tensor):
        logger.info(f"{name}: PyTorch shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    elif isinstance(tensor, ttnn.Tensor):
        logger.info(f"{name}: TTNN shape={tensor.shape}, dtype={tensor.dtype}")
    else:
        logger.info(f"{name}: Unknown tensor type {type(tensor)}")
