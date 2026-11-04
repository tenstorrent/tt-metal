# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""EXTRA_MODELS_DIR shim: append the repo root to sys.path (never insert(0)) and re-export the adapter."""

from __future__ import annotations

import sys
from pathlib import Path

# models/demos/blackhole/paddleocr_vl/vllm_bundle/paddleocr_vl/this_file.py
#   parents[0] paddleocr_vl (bundle)   parents[3] blackhole
#   parents[1] vllm_bundle             parents[4] demos
#   parents[2] paddleocr_vl (model)    parents[5] models
_REPO_ROOT = Path(__file__).resolve().parents[6]

if not (_REPO_ROOT / "models" / "demos" / "blackhole").is_dir():
    raise RuntimeError(
        f"{__name__}: expected the tt-metal root at {_REPO_ROOT}, but it has no "
        "models/demos/blackhole. The bundle was moved to a different depth, or "
        "the shipped path does not include everything from models/ down."
    )

if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from models.demos.blackhole.paddleocr_vl.tt.generator_vllm import PaddleOCRVLForConditionalGeneration  # noqa: E402

__all__ = ["PaddleOCRVLForConditionalGeneration"]
