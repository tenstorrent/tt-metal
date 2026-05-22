# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for VibeVoice-1.5B reference tests.

Prepends reference/ so vendored `vibevoice` imports resolve, and tt-metal root
so `models.experimental.vibevoice` imports work when running from repo root.
"""

import sys
from pathlib import Path

_VIBEVOICE_ROOT = Path(__file__).resolve().parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
_TT_METAL_ROOT = _VIBEVOICE_ROOT.parent.parent.parent

for path in (_REFERENCE_DIR, _TT_METAL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
