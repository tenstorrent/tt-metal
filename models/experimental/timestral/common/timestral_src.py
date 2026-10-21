# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Pinned upstream mistralai/timestral sources (the ``reference/`` submodule) and checkpoint paths."""

from __future__ import annotations

import sys
from pathlib import Path

TIMESTRAL_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SRC = TIMESTRAL_ROOT / "reference" / "src"
REFERENCE_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
TIMESTRAL_VERSION = "1.0.0"
WEIGHTS_DIR = TIMESTRAL_ROOT / "weights"

_SUBMODULE_HINT = (
    "From the tt-metal repo root, run:\n" "  git submodule update --init models/experimental/timestral/reference"
)


def require_timestral():
    """Import the upstream ``timestral`` package from the reference submodule, failing loudly if it is absent.

    The submodule's ``src/`` goes first on ``sys.path`` so a separately installed
    ``timestral`` cannot shadow the pinned sources.
    """
    if not (REFERENCE_SRC / "timestral" / "__init__.py").is_file():
        raise ImportError(f"the Timestral reference submodule is not checked out at {REFERENCE_SRC}. {_SUBMODULE_HINT}")
    src = str(REFERENCE_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    import timestral

    origin = Path(timestral.__file__).resolve().parents[1]
    if origin != REFERENCE_SRC:
        raise ImportError(f"timestral was imported from {origin}, not the pinned reference at {REFERENCE_SRC}")
    if timestral.__version__ != TIMESTRAL_VERSION:
        raise ImportError(
            f"the reference submodule has timestral {timestral.__version__}, expected {TIMESTRAL_VERSION}. "
            f"{_SUBMODULE_HINT}"
        )
    return timestral


def checkpoint_path(name: str) -> Path:
    """Local directory of a downloaded checkpoint, e.g. ``checkpoint_path("Timestral-209M")``."""
    return WEIGHTS_DIR / name


def checkpoint_ready(name: str) -> bool:
    path = checkpoint_path(name)
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def download_hint(name: str) -> str:
    return (
        f"Missing {checkpoint_path(name)}. Download with:\n"
        f"  hf download mistralai/{name} --local-dir models/experimental/timestral/weights/{name}"
    )
