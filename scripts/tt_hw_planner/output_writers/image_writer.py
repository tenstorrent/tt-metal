# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Image (PNG) output writer — for diffusion task templates."""

from __future__ import annotations


def emit_helper_snippet() -> str:
    """Source for a write_image helper."""
    return '''
def write_image(image, output_path):
    """Save a PIL.Image or HWC numpy array as PNG."""
    from pathlib import Path
    if hasattr(image, "save"):
        image.save(str(output_path))
        return
    import numpy as np
    from PIL import Image
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(output_path))
'''
