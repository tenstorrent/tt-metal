# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 profiling and optimization knobs."""

import os

import ttnn
from loguru import logger


_DTYPE_ALIASES = {
    "bf16": ("bf16", ttnn.bfloat16),
    "bfloat16": ("bf16", ttnn.bfloat16),
    "bfp8": ("bfp8", ttnn.bfloat8_b),
    "bfloat8_b": ("bfp8", ttnn.bfloat8_b),
    "bf8": ("bfp8", ttnn.bfloat8_b),
    "bfp4": ("bfp4", ttnn.bfloat4_b),
    "bfloat4_b": ("bfp4", ttnn.bfloat4_b),
    "bf4": ("bfp4", ttnn.bfloat4_b),
}


def env_weight_dtype(env_name: str, default):
    """Return ``(dtype, cache_suffix)`` for an optional dtype env override."""

    raw = os.getenv(env_name)
    if raw is None or raw == "":
        return default, ""
    key = raw.strip().lower()
    if key not in _DTYPE_ALIASES:
        valid = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"Unsupported {env_name}={raw!r}. Valid values: {valid}")
    canonical, dtype = _DTYPE_ALIASES[key]
    logger.info(f"Gemma4 override: {env_name}={canonical}")
    return dtype, f"_{canonical}"
