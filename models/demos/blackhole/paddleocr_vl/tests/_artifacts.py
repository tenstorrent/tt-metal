# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Skip on a missing golden artifact, or fail when PADDLEOCR_VL_REQUIRE_ARTIFACTS=1."""
from __future__ import annotations

import os

import pytest


def require_artifacts() -> bool:
    """True when missing artifacts must fail instead of skip (set for unattended runs)."""
    return os.environ.get("PADDLEOCR_VL_REQUIRE_ARTIFACTS", "").strip().lower() in ("1", "true", "yes")


def missing_artifact(msg: str) -> None:
    """Skip on a dev box; fail under ``PADDLEOCR_VL_REQUIRE_ARTIFACTS=1``.

    Regenerate with ``python models/demos/blackhole/paddleocr_vl/tests/generate_goldens.py``.
    """
    if require_artifacts():
        pytest.fail(f"{msg} [PADDLEOCR_VL_REQUIRE_ARTIFACTS=1]")
    pytest.skip(msg)
