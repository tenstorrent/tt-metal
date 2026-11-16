# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared paths and dependency pins for VibeVoice-1.5B reference / TT ports."""

import os
from pathlib import Path

VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = VIBEVOICE_ROOT / "resources"
VOICES_DIR = RESOURCES_DIR / "voices"
TEXT_EXAMPLES_DIR = RESOURCES_DIR / "text"

# Upstream demo assets: https://github.com/vibevoice-community/VibeVoice/tree/main/demo
GITHUB_DEMO_REPO = "vibevoice-community/VibeVoice"
GITHUB_DEMO_BRANCH = "main"
DEFAULT_TXT_PATH = TEXT_EXAMPLES_DIR / "1p_vibevoice.txt"
DEFAULT_VOICE_PATH = VOICES_DIR / "en-Alice_woman.wav"

HF_REPO_ID = "microsoft/VibeVoice-1.5B"
WEIGHTS_DIR = VIBEVOICE_ROOT / "weights"
DEFAULT_MODEL_PATH = WEIGHTS_DIR / "VibeVoice-1.5B"
MODEL_PATH_ENV_VAR = "VIBEVOICE_MODEL_PATH"

# Updated at runtime by ensure_model_weights() in tests and entry-point scripts.
MODEL_PATH = os.environ.get(MODEL_PATH_ENV_VAR, str(DEFAULT_MODEL_PATH))

# The reference processor loads the Qwen/Qwen2.5-1.5B tokenizer from the Hugging Face
# cache; it is not bundled in the VibeVoice-1.5B checkpoint.
#
# Reference parity requires transformers 4.51.3: 4.57 changed the generate() KV-cache API.
