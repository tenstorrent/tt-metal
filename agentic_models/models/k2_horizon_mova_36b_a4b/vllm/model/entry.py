"""Register K2-Horizon-MoVA-36B-A4B's vLLM architecture without opening devices."""
import importlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
METADATA = json.loads((PHASE / 'vllm_metadata.json').read_text())
#: vllm-tt-plugin resolves architectures under a TT prefix; the server selects this one with --hf-overrides.
ARCH = 'TT' + METADATA['arch']


def register():
    """Register the adapter class with vLLM. Idempotent; the plugin's EXTRA_MODELS_DIR scan registers the same target."""
    from vllm.model_executor.models.registry import ModelRegistry
    module, _, cls = METADATA['main_class'].partition(':')
    getattr(importlib.import_module(module), cls)  # native/runtime import check
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, METADATA['main_class'])
    return ARCH
