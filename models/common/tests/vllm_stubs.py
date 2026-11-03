# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stand-ins for the vLLM names the TT vLLM adapters import at module scope.

The tt-metal unit-test images have no vLLM, so a host test that imports an adapter
(``generator_vllm``, ``qwen36_vllm``) calls ``stub_vllm_if_missing`` first. Each stub is a
distinct empty class, so an adapter class can inherit any two of them.
"""

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

_STUBBED_CLASSES = {
    "vllm.inputs": ["MultiModalDataDict"],
    "vllm.model_executor.models.gemma3_mm": [
        "Gemma3DummyInputsBuilder",
        "Gemma3MultiModalProcessor",
        "Gemma3ProcessingInfo",
    ],
    "vllm.model_executor.models.interfaces": ["SupportsMultiModal"],
    "vllm.model_executor.models.mistral3": [
        "Mistral3DummyInputsBuilder",
        "Mistral3MultiModalProcessor",
        "Mistral3ProcessingInfo",
    ],
    "vllm.model_executor.models.qwen3_5": [
        "Qwen3_5ProcessingInfo",
        "Qwen3VLDummyInputsBuilder",
        "Qwen3VLMultiModalProcessor",
    ],
    "vllm.multimodal.processing": ["BaseDummyInputsBuilder"],
}


def stub_vllm_if_missing(monkeypatch):
    """Install the stubs in ``sys.modules`` for one test; a real vLLM install is used as is."""
    if "vllm" in sys.modules or importlib.util.find_spec("vllm") is not None:
        return
    registry = SimpleNamespace(register_processor=lambda *args, **kwargs: (lambda cls: cls))
    attrs_by_module = {
        "vllm": {},
        "vllm.model_executor": {},
        "vllm.model_executor.models": {},
        "vllm.multimodal": {"MULTIMODAL_REGISTRY": registry},
    }
    for module_name, class_names in _STUBBED_CLASSES.items():
        attrs_by_module[module_name] = {name: type(name, (), {}) for name in class_names}
    for module_name, attrs in attrs_by_module.items():
        module = ModuleType(module_name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module_name, module)
