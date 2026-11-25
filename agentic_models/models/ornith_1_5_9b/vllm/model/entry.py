"""Register Ornith-1.5-9B's vLLM architecture without opening devices."""
import importlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
METADATA = json.loads((PHASE / 'vllm_metadata.json').read_text())
#: vllm-tt-plugin resolves architectures under a TT prefix. The checkpoint's own hybrid architecture is kept, so the
#: server selects this adapter with TT_MODEL_CLASS_OVERRIDES (see README.md), not --hf-overrides.
ARCH = 'TT' + METADATA['arch']


def register():
    """Register the adapter class with vLLM, replacing any earlier registration of ARCH.

    The plugin registers its own in-tree Qwen3.5 port under the same name, so registering only when the name is
    missing would silently keep that port. Idempotent.
    """
    from vllm.model_executor.models.registry import ModelRegistry
    module, _, cls = METADATA['main_class'].partition(':')
    getattr(importlib.import_module(module), cls)  # native/runtime import check
    ModelRegistry.register_model(ARCH, METADATA['main_class'])
    return ARCH
