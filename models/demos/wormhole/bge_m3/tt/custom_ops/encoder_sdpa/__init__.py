# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Experimental exact-shape BGE-M3 encoder SDPA."""

from .config import EncoderSDPAConfig
from .op import (
    bge_encoder_sdpa_experimental,
    bge_encoder_sdpa_stock,
    build_encoder_sdpa_descriptor,
)

__all__ = [
    "EncoderSDPAConfig",
    "bge_encoder_sdpa_experimental",
    "bge_encoder_sdpa_stock",
    "build_encoder_sdpa_descriptor",
]
