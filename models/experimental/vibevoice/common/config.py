# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared paths and dependency pins for VibeVoice-1.5B reference / TT ports."""

import os
from pathlib import Path

VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = VIBEVOICE_ROOT / "reference"
RESOURCES_DIR = VIBEVOICE_ROOT / "resources"
VOICES_DIR = RESOURCES_DIR / "voices"
TEXT_EXAMPLES_DIR = RESOURCES_DIR / "text"
DEFAULT_TXT_PATH = TEXT_EXAMPLES_DIR / "1p_short.txt"

_LEGACY_DEFAULT_MODEL_PATH = "/home/iguser/devstral2/VibeVoice/VibeVoice-1.5B"
_HF_REPO_ID = "microsoft/VibeVoice-1.5B"


def _hf_hub_snapshot_path() -> Path | None:
    """Newest local HF hub snapshot for VibeVoice-1.5B, if downloaded."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots = hf_home / "hub" / f"models--{_HF_REPO_ID.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [p for p in snapshots.iterdir() if p.is_dir() and (p / "config.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_model_path() -> str:
    """Model weights directory: env override, legacy default, then HF hub cache."""
    if env := os.environ.get("VIBEVOICE_MODEL_PATH"):
        return env
    legacy = Path(_LEGACY_DEFAULT_MODEL_PATH)
    if legacy.is_dir():
        return str(legacy)
    if hf_snap := _hf_hub_snapshot_path():
        return str(hf_snap)
    return _LEGACY_DEFAULT_MODEL_PATH


MODEL_PATH = resolve_model_path()

# Processor loads Qwen tokenizer from HF cache (not bundled in VibeVoice-1.5B weights).
QWEN_TOKENIZER = "Qwen/Qwen2.5-1.5B"

# transformers>=4.57 changes generate() KV-cache API; pin for reference parity.
TRANSFORMERS_VERSION = "4.51.3"

DEFAULT_DEVICE = os.environ.get("VIBEVOICE_DEVICE", "cpu")

# Only VibeVoice checkpoint is supported for PCC tests; do not load bare Qwen weights.
LM_WEIGHT_SOURCE = "vibevoice"
