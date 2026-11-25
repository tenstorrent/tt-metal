"""Register NVIDIA-Nemotron-3.5-Lightning-30B-A3B's vLLM architecture without opening devices."""
import importlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
METADATA = json.loads((PHASE / 'vllm_metadata.json').read_text())
#: vllm-tt-plugin resolves architectures under a TT prefix. The checkpoint's own hybrid arch is kept (vLLM's
#: NemotronH hybrid config handling keys on it); serving selects this class with
#: TT_MODEL_CLASS_OVERRIDES='TTNemotronHForCausalLM=<main_class>'.
ARCH = 'TT' + METADATA['arch']


def register():
    """Register the adapter class with vLLM, overwriting any earlier entry. Idempotent.

    Unconditional, like the plugin's TT_MODEL_CLASS_OVERRIDES: an if-missing registration would silently keep
    whatever class already holds the architecture.
    """
    from vllm.model_executor.models.registry import ModelRegistry
    module, _, cls = METADATA['main_class'].partition(':')
    getattr(importlib.import_module(module), cls)  # native/runtime import check
    ModelRegistry.register_model(ARCH, METADATA['main_class'])
    return ARCH
