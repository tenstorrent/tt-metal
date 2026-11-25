"""Register Qwen3.6-27B's vLLM architecture without opening devices."""
import importlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
METADATA = json.loads((PHASE / 'vllm_metadata.json').read_text())
#: vllm-tt-plugin resolves the checkpoint's architecture under a TT prefix. The plugin also ships a builtin
#: TTQwen3_5ForConditionalGeneration (the tt-metal demo), so serving selects this adapter with
#: TT_MODEL_CLASS_OVERRIDES='TTQwen3_5ForConditionalGeneration=<main_class>'.
ARCH = 'TT' + METADATA['arch']


def register():
    """Register the adapter class with vLLM, replacing any earlier registration (the plugin builtin included).

    Registering only if missing would silently keep the builtin class for this architecture.
    """
    from vllm.model_executor.models.registry import ModelRegistry
    module, _, cls = METADATA['main_class'].partition(':')
    getattr(importlib.import_module(module), cls)  # native/runtime import check
    ModelRegistry.register_model(ARCH, METADATA['main_class'])
    return ARCH
