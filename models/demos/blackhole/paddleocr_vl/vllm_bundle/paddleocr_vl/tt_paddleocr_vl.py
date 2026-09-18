# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bundle entry point for ``EXTRA_MODELS_DIR``.

``vllm_tt_plugin.platform.register_tt_models()`` runs
``_register_models_from_extra_dir(ModelRegistry)`` as its first action, so a
distributed bundle can supply a model without editing the plugin. That hook
appends *this folder* to ``sys.path`` and lazily registers the ``main_class``
named in ``vllm_metadata.json`` under the plugin's ``TT``-prefixed convention,
which is how ``PaddleOCRVLForConditionalGeneration`` becomes
``TTPaddleOCRVLForConditionalGeneration`` with no plugin change.

Registration is lazy: vLLM resolves the ``"module:Class"`` string later, in the
API-server process and again in each EngineCore worker. This module therefore has
to be importable on its own and cannot assume the tt-metal checkout is already on
``sys.path`` -- a worker's working directory is not guaranteed. It appends the
repository root (never ``insert(0)``, matching the hook's own rule that an
installed package of the same name must still win) and re-exports the real
adapter, which lives with the model it adapts.

Same shape as the GLM-4.7-Flash bundle, one directory deeper: this model sits at
``models/demos/blackhole/<model>/`` rather than ``models/autoports/<model>/``.
The guard below turns a wrong depth into an explanatory failure rather than a
``ModuleNotFoundError`` from inside a worker.
"""

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
