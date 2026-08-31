# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-metal-owned vLLM general plugin for Muse-Glimmer-30B.

Registered via the ``vllm.general_plugins`` entry point (see pyproject.toml), so
vLLM imports and calls ``register()`` in every process that loads plugins -
including the API-server frontend, which is where tool-call parsing runs.

Two registrations, neither of which needs a patch to vllm-tt-plugin:

1. **The architecture.** The checkpoint declares
   ``MuseGlimmerForConditionalGeneration`` with nested text/vision configs, so
   ``hf_config != hf_text_config`` and upstream's resolver falls back to
   ``TransformersMultiModalForCausalLM`` inside ``ModelConfig.__post_init__`` -
   which runs *before* the plugin's ``TT``-prefix rewrite.  Registering the plain
   HF names here wins that resolution; the ``TT``-prefixed aliases satisfy the
   plugin's later ``check_and_update_config`` check.  Load order makes this work:
   ``load_general_plugins()`` runs in ``EngineArgs.__post_init__``
   (``arg_utils.py:757``) and ``ModelConfig(...)`` is built later in
   ``create_model_config()`` (``:1598``).

2. **The tool parser.** See :mod:`tool_parser` - the checkpoint's function-call
   grammar is its own, and no stock vLLM parser reads it.

Both are idempotent, mirroring the plugin's ``_register_model_if_missing``.
"""

import logging

logger = logging.getLogger(__name__)

_TARGET = "models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm" ":MuseGlimmerForConditionalGeneration"

# Plain HF names win upstream resolution; TT-prefixed satisfy check_and_update_config.
_ARCHS = (
    "MuseGlimmerForConditionalGeneration",
    "MuseGlimmerForCausalLM",
    "TTMuseGlimmerForConditionalGeneration",
    "TTMuseGlimmerForCausalLM",
)


def _register_architectures() -> None:
    from vllm.model_executor.models.registry import ModelRegistry

    supported = ModelRegistry.get_supported_archs()
    for arch in _ARCHS:
        if arch not in supported:
            ModelRegistry.register_model(arch, _TARGET)
            logger.info("Registered TT model %s -> %s", arch, _TARGET)


def _register_tool_parser() -> None:
    # Importing the module runs its @ToolParserManager.register_module decorator.
    from . import tool_parser  # noqa: F401

    logger.info("Registered tool parser 'muse_glimmer'")


def register() -> None:
    for step in (_register_architectures, _register_tool_parser):
        try:
            step()
        except Exception:  # pragma: no cover - never break plugin loading
            logger.warning("muse_glimmer_vllm_ext: %s failed", step.__name__, exc_info=True)
