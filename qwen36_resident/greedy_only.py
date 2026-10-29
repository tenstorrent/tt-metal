# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy-only serving classes (a class attribute greedy_only = True: MTP, DFlash2): one warning in the server log the
first time a request that samples (temperature > 0) is served greedily.

These classes mask their prefill logits to the token the device decodes from, so the plugin's temperature-1 sample of
a block-output request's first token returns exactly that token and every request is served greedily, whatever it
asks. The engine cannot tell: the plugin resets a block-output request's sampling controls on its per-request clone
before the engine sees it. The API server's vLLM input processor is the last place that sees what the request asked,
so install() wraps InputProcessor.process_inputs once (logging only; the request goes on unchanged). vLLM imports the
served class in the API server process (the platform reads its capabilities), which is where install() runs.
"""
from loguru import logger

_state = {"installed": False, "warned": False}


def _served_class_is_greedy_only(processor):
    """whether the class this server runs declares greedy_only (resolved once per input processor)"""
    cached = getattr(processor, "_resident_greedy_only", None)
    if cached is None:
        from vllm.model_executor.model_loader.utils import get_model_architecture

        cls, _ = get_model_architecture(processor.model_config)
        cached = bool(getattr(cls, "greedy_only", False))
        processor._resident_greedy_only = cached
    return cached


def install():
    if _state["installed"]:
        return
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor

    original = InputProcessor.process_inputs

    def process_inputs(self, request_id, prompt, params, *args, **kwargs):
        if (
            not _state["warned"]
            and isinstance(params, SamplingParams)
            and params.temperature > 0
            and _served_class_is_greedy_only(self)
        ):
            _state["warned"] = True
            logger.warning(
                f"This profile is greedy-only: request {request_id} asks for temperature {params.temperature} "
                f"(top_k {params.top_k}, top_p {params.top_p}) and is served greedily, as is every sampling request "
                f"(logged once)"
            )
        return original(self, request_id, prompt, params, *args, **kwargs)

    InputProcessor.process_inputs = process_inputs
    _state["installed"] = True
