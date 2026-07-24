# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint directory resolution without the ACE-Step ``acestep`` package."""

from __future__ import annotations

import os
from pathlib import Path


def get_checkpoints_dir() -> Path:
    """Match ACE-Step ``get_checkpoints_dir`` env-based resolution (no project root)."""
    env_dir = os.environ.get("ACESTEP_CHECKPOINTS_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(os.path.expanduser("~/.cache/acestep/checkpoints"))
