"""Register Ornith-1.5-35B-A3B's vLLM architecture without opening devices."""
import importlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
METADATA = json.loads((PHASE / 'vllm_metadata.json').read_text())
#: vllm-tt-plugin resolves architectures under a TT prefix. This is the checkpoint's own (hybrid VL) architecture, which
#: the plugin also maps to its in-tree Qwen3.5 demo; serving selects this package with TT_MODEL_CLASS_OVERRIDES.
ARCH = 'TT' + METADATA['arch']


def register():
    """Register the adapter class with vLLM, replacing any earlier registration of the architecture. Idempotent."""
    from vllm.model_executor.models.registry import ModelRegistry
    module, _, cls = METADATA['main_class'].partition(':')
    getattr(importlib.import_module(module), cls)  # native/runtime import check
    # Overwrite: keeping an existing entry would silently serve the plugin's builtin TTQwen3_5Moe* demo instead.
    ModelRegistry.register_model(ARCH, METADATA['main_class'])
    return ARCH
