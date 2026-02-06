# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for models/common tests."""

from __future__ import annotations

import zlib


def stable_model_seed(model_name: str) -> int:
    """Stable 32-bit seed derived from model name.

    Python's built-in hash is randomized per process, which breaks reproducibility
    across runs and can mismatch on-disk cached weights. Use CRC32 instead.

    NOTE: Avoid a single hardcoded global seed (e.g., 1234) for all models; a
    per-model stable seed keeps caches distinct and reduces correlated RNG paths.
    """
    return zlib.crc32(model_name.encode("utf-8")) & 0xFFFFFFFF
