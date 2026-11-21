# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""FLUX.3 Image klein's served sizes, steps and reference limits, and the request check over them.

Sizes are (width, height). This module imports no device code at load, so a server process can import it to
validate requests before any worker opens a device.
"""

from __future__ import annotations

# Square outputs at 1K, 2K and 4K.
FLUX3_KLEIN_SERVING_SIZES = ((1024, 1024), (2048, 2048), (4096, 4096))
FLUX3_KLEIN_DEFAULT_SIZE = (2048, 2048)

# The distilled checkpoint's step count.
FLUX3_KLEIN_NUM_INFERENCE_STEPS = 16

# A reference is fitted, aspect kept and never cropped, to a canvas of at most 4 MP with sides of 64-4096 px.
FLUX3_KLEIN_MAX_REFERENCES = 1
FLUX3_KLEIN_REFERENCE_MAX_PIXELS = 4 * 1024 * 1024
FLUX3_KLEIN_REFERENCE_MIN_SIDE = 64
FLUX3_KLEIN_REFERENCE_MAX_SIDE = 4096

# With the prompt upsampler resident on the mesh, its DRAM holds on every chip leave no room for the 4096 activations.
_UPSAMPLER_RESIDENT_MAX_SIDE = 2048


def served_resolutions(*, upsampler_resident: bool) -> tuple[tuple[int, int], ...]:
    if upsampler_resident:
        return tuple(size for size in FLUX3_KLEIN_SERVING_SIZES if max(size) <= _UPSAMPLER_RESIDENT_MAX_SIDE)
    return FLUX3_KLEIN_SERVING_SIZES


def validate_size(*, width: int, height: int, upsampler_resident: bool) -> None:
    served = served_resolutions(upsampler_resident=upsampler_resident)
    if (width, height) not in served:
        where = " with the prompt upsampler resident" if upsampler_resident else ""
        msg = f"unsupported size {width}x{height}{where}; this deployment serves {list(served)}"
        raise ValueError(msg)
