# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bundle entry point for ``EXTRA_MODELS_DIR``.

``vllm_tt_plugin.platform.register_tt_models()`` runs
``_register_models_from_extra_dir(ModelRegistry)`` as its **first** action, so a
distributed bundle can supply a model without touching the plugin. That hook
appends this folder to ``sys.path`` and lazily registers ``vllm_metadata.json``'s
``main_class`` under the plugin's ``TT``-prefixed convention, which is how
``Glm4MoeLiteForCausalLM`` becomes ``TTGlm4MoeLiteForCausalLM`` with no edit to
the plugin checkout.

Registration is lazy: vLLM resolves the ``"module:Class"`` string later, in the
API-server process and again in each EngineCore worker. This module therefore has
to be importable on its own, which means it cannot assume the tt-metal checkout is
already on ``sys.path`` -- an EngineCore worker's working directory is not
guaranteed. It appends the repository root (never ``insert(0)``, matching the
hook's own rule that an installed package of the same name must still win) and
re-exports the real adapter, which lives with the model it adapts.

This is the same shape as the qwen3-coder bundle, with one difference that matters:
an autoport sits at ``models/autoports/<model>/`` rather than
``models/demos/<arch>/<model>/``, so the repository root is one directory closer.
The guard below turns a wrong depth into an immediate, explanatory failure instead
of a ``ModuleNotFoundError`` from inside an EngineCore worker.
"""

from __future__ import annotations

import sys
from pathlib import Path

# models/autoports/zai_org_glm_4_7_flash/vllm_bundle/glm_4_7_flash/this_file.py
#   parents[0] glm_4_7_flash      parents[3] autoports
#   parents[1] vllm_bundle        parents[4] models
#   parents[2] zai_org_glm_4_7_flash
_REPO_ROOT = Path(__file__).resolve().parents[5]

if not (_REPO_ROOT / "models" / "autoports").is_dir():
    raise RuntimeError(
        f"{__name__}: expected the tt-metal root at {_REPO_ROOT}, but it has no "
        "models/autoports. The bundle was moved to a different depth, or the "
        "source.code allowlist did not ship the whole path from models/ down."
    )

if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm import GLM47FlashForCausalLM  # noqa: E402

__all__ = ["GLM47FlashForCausalLM"]
