# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from unittest import mock

# The models unit-test image has no vLLM, but generator_vllm imports it at module level for the
# multimodal registry. These tests only use initialize_vllm_text_transformer, so stub the names.
# The two bases are real classes because Generator subclasses mix them in.
if importlib.util.find_spec("vllm") is None:
    for name in (
        "vllm",
        "vllm.inputs",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.model_executor.models.gemma3_mm",
        "vllm.model_executor.models.interfaces",
        "vllm.model_executor.models.mistral3",
        "vllm.multimodal",
        "vllm.multimodal.inputs",
        "vllm.multimodal.processing",
    ):
        sys.modules[name] = mock.MagicMock()
    sys.modules["vllm.model_executor.models.interfaces"].SupportsMultiModal = type("SupportsMultiModal", (), {})
    sys.modules["vllm.multimodal.processing"].BaseDummyInputsBuilder = type("BaseDummyInputsBuilder", (), {})
