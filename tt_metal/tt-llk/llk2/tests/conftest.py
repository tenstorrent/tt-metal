# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 tests: an out-of-tree consumer of the tt-llk test harness.

Run from this directory: ``pytest`` (add ``--run-simulator`` for ttsim). The
harness comes from ``tt-llk/tests/python_tests``; the LLK 2.0 headers are put on
the include path as ``inc/common`` plus ``inc/<arch>``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LLK2_ROOT = Path(__file__).resolve().parent.parent  # tt-llk/llk2
LLK_HOME = Path(os.environ.get("LLK_HOME", LLK2_ROOT.parent)).resolve()  # tt-llk

os.environ.setdefault("LLK_HOME", str(LLK_HOME))
sys.path.insert(0, str(LLK_HOME / "tests" / "python_tests"))

pytest_plugins = ["tt_llk_harness.plugin"]

import tt_llk_harness  # noqa: E402

tt_llk_harness.require_version(1, 0)
