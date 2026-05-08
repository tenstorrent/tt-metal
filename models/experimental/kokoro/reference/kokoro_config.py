# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KokoroConfig:
    """Reference configuration for Kokoro bring-up in tt-metal.

    The upstream `kokoro` pip package defaults to repo_id='hexgrad/Kokoro-82M'.
    Keeping this explicit here makes it easy to pin/override for future variants.
    """

    repo_id: str = "hexgrad/Kokoro-82M"
    sample_rate_hz: int = 24000
    max_phoneme_chars: int = 510  # upstream hard limit (excludes BOS/EOS)
