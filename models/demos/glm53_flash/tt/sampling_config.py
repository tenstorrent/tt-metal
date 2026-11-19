# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device sampler's ttnn-free constants and knobs (``tt/sampler.py`` holds the device code; the serving adapter
imports these at module level, before ttnn is up)."""

from __future__ import annotations

import os

DEVICE_SAMPLING_ENV = "GLM53_DEVICE_SAMPLING"  # "1": the decode step samples on device (the sampler in the trace)
SAMPLING_SEED_ENV = "GLM53_SAMPLING_SEED"  # pins the adapter's RNG for unseeded rows (reproducible runs)
CANDIDATES = 64  # top-k candidates per user handed to ttnn.sampling: two tiles (never one: a one-tile row hangs)
MAX_TOP_K = 32  # ttnn.sampling walks at most 32 candidates (the reference models' max_device_top_k)
MIN_TOP_K = 2  # k = 1 is untested by the op's unit tests; a greedy row takes the argmax through the mask instead
MAX_USERS = 32  # ttnn.sampling: 1 to 32 users per call (one core each)


def device_sampling_enabled() -> bool:
    return os.environ.get(DEVICE_SAMPLING_ENV, "0") == "1"


def clamp_k(k: int) -> int:
    """vLLM's ``top_k`` to the device's: ``< 1`` (no top-k) and ``> MAX_TOP_K`` become ``MAX_TOP_K`` (sampling from
    the top-32 under top-p: the reference models' approximation), ``1`` becomes ``MIN_TOP_K`` (its row is greedy
    through the mask)."""
    k = int(k)
    if k < 1 or k > MAX_TOP_K:
        return MAX_TOP_K
    return max(k, MIN_TOP_K)
