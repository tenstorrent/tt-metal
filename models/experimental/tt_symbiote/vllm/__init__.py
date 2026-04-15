# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Central registry of all tt_symbiote models exposed to vLLM.
#
# To add a new symbiote model:
#   1. Create an adapter class in this package (e.g. generator_vllm_<model>.py)
#   2. Add an entry below mapping the TT architecture name -> module:class path
#   3. Add a ModelSpecTemplate in tt-inference-server/workflows/model_spec.py
#
# The TT architecture name is the HuggingFace architecture prefixed with "TT"
# (applied by TTPlatform.check_and_update_config).
#
# All three registration sites (tt-vllm-plugin, run_vllm_api_server.py, and
# vllm/platforms/tt.py) import this dict so there is exactly one place to update.

SYMBIOTE_MODEL_REGISTRY: dict[str, str] = {
    "TTGemma4ForConditionalGeneration": (
        "models.experimental.tt_symbiote.vllm.generator_vllm:SymbioteGemma4ForCausalLM"
    ),
}
