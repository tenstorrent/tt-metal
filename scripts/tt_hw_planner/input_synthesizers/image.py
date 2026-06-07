# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image input synthesizer — used by VLM / classification / segmentation /
diffusion task templates. Phase 2 minimum-viable emission."""

from __future__ import annotations

from ..task_templates._base import TemplateContext


def emit_source(ctx: TemplateContext) -> str:
    """Return source code for ``demo/image_loader.py``."""
    return '''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image loader + standard ImageNet preprocessing for the generated demo."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def load_image(path) -> "PIL.Image.Image":
    """PIL.Image.open with RGB conversion."""
    from PIL import Image
    return Image.open(str(path)).convert("RGB")


def synthesize_image(size: Tuple[int, int] = (224, 224)) -> "PIL.Image.Image":
    """Deterministic random PIL image for smoke tests."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype="uint8")
    return Image.fromarray(arr)


def to_tensor_imagenet(img: "PIL.Image.Image", resize: int = 256, crop: int = 224) -> torch.Tensor:
    """torchvision: Resize -> CenterCrop -> ToTensor -> Normalize(ImageNet)."""
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize(resize),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)   # (1, 3, H, W)


__all__ = ["load_image", "synthesize_image", "to_tensor_imagenet"]
'''
