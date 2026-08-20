# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bundle entry point for ``EXTRA_MODELS_DIR``.

``vllm_tt_plugin.platform.register_tt_models()`` runs
``_register_models_from_extra_dir(ModelRegistry)`` as its **first** action --
"so a distributed bundle can supply a model without touching this file". That
hook appends this folder to ``sys.path`` and lazily registers
``vllm_metadata.json``'s ``main_class`` under the plugin's ``TT``-prefixed
convention, so the model ends up registered *by* ``register_tt_models()`` with
no edit to the plugin checkout.

Registration is lazy: vLLM resolves the ``"module:Class"`` string later, in the
API-server process and again in each EngineCore worker. This module therefore
has to be importable on its own, which means it cannot assume the tt-metal
checkout is already on ``sys.path`` -- an EngineCore worker's working directory
is not guaranteed. It appends the repository root (never ``insert(0)``, matching
the hook's own rule that an installed package of the same name must still win)
and re-exports the real adapter, which lives with the model it adapts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../models/demos/blackhole/<model>/vllm_bundle/<bundle>/this_file.py
_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from models.demos.blackhole.qwen3_coder_30b_a3b.tt.generator_vllm import Qwen3CoderForCausalLM  # noqa: E402

__all__ = ["Qwen3CoderForCausalLM"]
