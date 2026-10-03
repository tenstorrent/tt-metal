# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Locate the vendored amazon-science/chronos-forecasting submodule."""

from __future__ import annotations

import sys
from pathlib import Path

CHRONOS_SUBMODULE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "chronos-forecasting"
CHRONOS_SRC = CHRONOS_SUBMODULE_ROOT / "src"

_INIT_ERROR = (
    "Chronos submodule not found at {path}. From the tt-metal repo root run:\n"
    "  git submodule update --init models/experimental/chronos_forecast/third_party/chronos-forecasting"
)


def ensure_chronos_on_path() -> Path:
    """Put the submodule `src/` on sys.path so `import chronos` uses the pinned checkout."""
    if not (CHRONOS_SRC / "chronos" / "__init__.py").is_file():
        raise FileNotFoundError(_INIT_ERROR.format(path=CHRONOS_SUBMODULE_ROOT))
    src = str(CHRONOS_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return CHRONOS_SRC
