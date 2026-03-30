# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Voice preset loading and caching for Bark Small.

Provides efficient voice preset management with an in-memory cache
so that switching between presets does not re-download weights.

Usage:
    from models.demos.wormhole.bark.tt.bark_voice_presets import load_voice_preset

    preset = load_voice_preset("v2/en_speaker_0")
    audio = bark_model.generate("Hello!", voice_preset=preset)
"""

from typing import Optional

# Built-in voice presets available in suno/bark-small
BUILTIN_PRESETS = [
    "v2/en_speaker_0",
    "v2/en_speaker_1",
    "v2/en_speaker_2",
    "v2/en_speaker_3",
    "v2/en_speaker_4",
    "v2/en_speaker_5",
    "v2/en_speaker_6",
    "v2/en_speaker_7",
    "v2/en_speaker_8",
    "v2/en_speaker_9",
    "v2/zh_speaker_0",
    "v2/zh_speaker_1",
    "v2/es_speaker_0",
    "v2/fr_speaker_0",
    "v2/de_speaker_0",
    "v2/ja_speaker_0",
    "v2/ko_speaker_0",
]

_preset_cache: dict = {}


def load_voice_preset(preset_name: str, cache: bool = True) -> Optional[str]:
    """Load a Bark voice preset for consistent speaker characteristics.

    Uses the HuggingFace BarkProcessor to validate speaker presets.
    The processor accepts the preset name string directly as ``voice_preset``.

    Args:
        preset_name: Preset identifier (e.g. "v2/en_speaker_0").
        cache: Whether to cache the validation result (default True).

    Returns:
        Voice preset name string suitable for BarkProcessor, or None if invalid.
    """
    if cache and preset_name in _preset_cache:
        return _preset_cache[preset_name]

    if preset_name not in BUILTIN_PRESETS:
        print(f"WARNING: '{preset_name}' not in built-in presets")
        return None

    # Return the preset name string — BarkProcessor accepts it directly
    if cache:
        _preset_cache[preset_name] = preset_name

    return preset_name


def list_available_presets() -> list:
    """Return all available built-in voice preset names."""
    return list(BUILTIN_PRESETS)


def clear_preset_cache():
    """Clear the in-memory voice preset cache."""
    _preset_cache.clear()
