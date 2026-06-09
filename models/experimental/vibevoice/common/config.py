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

HF_REPO_ID = "microsoft/VibeVoice-1.5B"
WEIGHTS_DIR = VIBEVOICE_ROOT / "weights"
DEFAULT_MODEL_PATH = WEIGHTS_DIR / "VibeVoice-1.5B"
MODEL_PATH_ENV_VAR = "VIBEVOICE_MODEL_PATH"

# Updated at runtime by ensure_model_weights() in tests and entry-point scripts.
MODEL_PATH = os.environ.get(MODEL_PATH_ENV_VAR, str(DEFAULT_MODEL_PATH))

# Processor loads Qwen tokenizer from HF cache (not bundled in VibeVoice-1.5B weights).
QWEN_TOKENIZER = "Qwen/Qwen2.5-1.5B"

# transformers>=4.57 changes generate() KV-cache API; pin for reference parity.
TRANSFORMERS_VERSION = "4.51.3"

DEFAULT_DEVICE = os.environ.get("VIBEVOICE_DEVICE", "cpu")

# Only VibeVoice checkpoint is supported for PCC tests; do not load bare Qwen weights.
LM_WEIGHT_SOURCE = "vibevoice"
