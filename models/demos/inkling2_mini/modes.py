# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""The weight caches each run mode loads (no ttnn import).

- default: own3-def8, experts GPTQ-rounded with a balanced calibration mix (KL .0067 vs the bf16 model).
- fast: own3-l8, layers 0-19 experts 4-bit (~1.5 ms/token faster decode at 4K, KL .0096).

Pick one with ``$INKLING_MODE`` (``default`` or ``fast``) or ``--fast`` in the demo.
``$INKLING_CACHE`` / ``$INKLING_TILE_CACHE`` still override the decode / prefill cache of either mode.
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE_DIR = Path("/work/inkling-bringup/cache")
# mode: (decode cache, PF expert layout; prefill cache, TILE layout)
CACHES = {"default": ("own3-def8-pf", "own3-def8-tile"), "fast": ("own3-l8-pf", "own3-l8-tile")}


def mode(fast=False) -> str:
    m = "fast" if fast else os.environ.get("INKLING_MODE") or "default"
    if m not in CACHES:
        raise ValueError(f"INKLING_MODE={m!r}: use one of {', '.join(CACHES)}")
    return m


def roots(fast=False) -> tuple[Path, Path]:
    """(decode cache, prefill cache) for the mode; the INKLING_CACHE / INKLING_TILE_CACHE overrides win."""
    pf, tile = CACHES[mode(fast)]
    return (
        Path(os.environ.get("INKLING_CACHE") or CACHE_DIR / pf),
        Path(os.environ.get("INKLING_TILE_CACHE") or CACHE_DIR / tile),
    )
