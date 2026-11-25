"""Register Qwen3.6-35B-A3B's vLLM architecture without opening devices."""
import importlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
METADATA = json.loads((PHASE / 'vllm_metadata.json').read_text())
#: vllm-tt-plugin resolves architectures under a TT prefix. The checkpoint's own architecture is kept (no
#: --hf-overrides), so vLLM's hybrid handling (mamba "align" mode under prefix caching) is the validated one.
ARCH = 'TT' + METADATA['arch']
#: How the server selects this adapter (applied by the plugin before model resolution).
TT_MODEL_CLASS_OVERRIDES = f"{ARCH}={METADATA['main_class']}"


def register():
    """Register the adapter class with vLLM. Idempotent.

    Always overwrites, like the plugin's TT_MODEL_CLASS_OVERRIDES: an if-missing registration would silently keep
    whatever registered the Qwen3.5-family architecture first (a plugin built-in or another bundle).
    """
    from vllm.model_executor.models.registry import ModelRegistry
    module, _, cls = METADATA['main_class'].partition(':')
    getattr(importlib.import_module(module), cls)  # native/runtime import check
    ModelRegistry.register_model(ARCH, METADATA['main_class'])
    return ARCH
