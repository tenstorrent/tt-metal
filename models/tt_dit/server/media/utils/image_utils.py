# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from io import BytesIO
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image


def ndarray_to_b64npy(arr: np.ndarray) -> Dict[str, Any]:
    """Serialize a NumPy array to a base64 ``.npy`` payload (JSON-friendly).

    Used by the staged HTTP endpoints to ship latents / images to ComfyUI.
    """
    buf = BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return {
        "b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def b64npy_to_ndarray(payload: Dict[str, Any]) -> np.ndarray:
    """Deserialize a base64 ``.npy`` payload (as produced by ``ndarray_to_b64npy``)."""
    raw = base64.b64decode(payload["b64"])
    return np.load(BytesIO(raw), allow_pickle=False)


def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL image to base64 string

    Args:
        image: PIL Image
        format: Output format (JPEG, PNG)

    Returns:
        Base64-encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def base64_to_pil(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL image

    Args:
        base64_str: Base64-encoded image string

    Returns:
        PIL Image
    """
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes))


def tensor_to_pil(tensor: torch.Tensor, image_processor) -> Image.Image:
    """
    Convert tensor to PIL image using pipeline's processor

    Args:
        tensor: PyTorch tensor from SDXL pipeline
        image_processor: Image processor from DiffusionPipeline

    Returns:
        PIL Image
    """
    tensor = tensor.unsqueeze(0)
    return image_processor.postprocess(tensor, output_type="pil")[0]
