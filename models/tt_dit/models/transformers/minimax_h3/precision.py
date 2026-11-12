# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Default-off float32 precision switches for the denoise loop and the DiT.

Each switch is `0` / unset (today's code, bit-identical) or `1`; any other value raises at import.

- MINIMAX_H3_FP32_LATENTS=1: the latent stream between denoise steps is float32 on device: the noise / condition
  latent uploads, the Euler update `x += dt * v` and the readback. The input projections still read bfloat16 rows
  (cast from the float32 stream).
"""

from __future__ import annotations

import os

import ttnn


def _flag(name: str) -> bool:
    value = os.environ.get(name, "").strip()
    if value not in ("", "0", "1"):
        msg = f"{name}: expected 0 or 1, got {value!r}"
        raise ValueError(msg)
    return value == "1"


FP32_LATENTS = _flag("MINIMAX_H3_FP32_LATENTS")

if FP32_LATENTS:
    print("[MINIMAX_H3_FP32_LATENTS] latent stream, Euler update and readback in float32")


def latent_dtype() -> ttnn.DataType:
    """The device dtype of the denoise latent streams."""
    return ttnn.float32 if FP32_LATENTS else ttnn.bfloat16


def cast(t: ttnn.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
    """`t` itself when it already has `dtype`, else a typecast copy."""
    return t if t.dtype == dtype else ttnn.typecast(t, dtype)


def bf16_if_fp32(t: ttnn.Tensor) -> ttnn.Tensor:
    """A float32 tensor as bfloat16 (the input of a bfloat16-weight linear); any other tensor itself."""
    return ttnn.typecast(t, ttnn.bfloat16) if t.dtype == ttnn.float32 else t
