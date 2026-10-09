# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Pinned upstream amazon-science/chronos-forecasting package and test fixtures."""

from __future__ import annotations

from pathlib import Path

CHRONOS_VERSION = "2.3.2"
DUMMY_MODEL_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dummy-chronos2-model"

_INSTALL_HINT = (
    "From the tt-metal repo root, with python_env active, run:\n"
    "  uv pip install -r models/experimental/chronos_forecast/requirements.txt"
)


def require_chronos():
    """Import the upstream `chronos` package, failing loudly if it is missing or not the pinned version."""
    try:
        import chronos
    except ImportError as e:
        raise ImportError(f"chronos-forecasting=={CHRONOS_VERSION} is not installed. {_INSTALL_HINT}") from e
    if chronos.__version__ != CHRONOS_VERSION:
        raise ImportError(
            f"chronos-forecasting {chronos.__version__} is installed but the reference is pinned to "
            f"{CHRONOS_VERSION}. {_INSTALL_HINT}"
        )
    return chronos
