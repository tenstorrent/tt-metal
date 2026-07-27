"""Import vLLM-Omni's Voxtral-TTS model classes on a CPU box with no vllm installed.

Installs permissive stub modules for every vllm / mistral_common / transformers import the two
model files make at module scope, then execs the real source unmodified. The classes we care
about (FlowMatchingAudioTransformer, VoxtralTTSAudioTokenizer's decode path) never touch those
imports at runtime, so this preserves their math exactly. einops and torch are REAL.

flash_attn is deliberately absent -> upstream falls back to its own _native_attention SDPA path,
which is the CPU-correct reference for the ALiBi + causal + sliding-window bias.
"""

import os
import sys
import types

SRC = os.environ.get("VOXTRAL_UPSTREAM_SRC",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream_src"))

GEN_MOD = "vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation"


class _Any:
    """Stands in for any vllm class: subclassable, subscriptable, callable, decorator."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return a[0] if a and callable(a[0]) else self

    def __class_getitem__(cls, item):
        return cls

    def __getattr__(self, name):
        return _Any()

    @classmethod
    def register_processor(cls, *a, **k):
        return lambda x: x


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name == "init_logger":
            import logging

            return lambda *a, **k: logging.getLogger("stub")
        if name == "MULTIMODAL_REGISTRY":
            return _Any
        return _Any


_STUBS = [
    "apex", "apex.normalization", "flash_attn",  # absent on purpose -> torch fallbacks
    "vllm", "vllm.config", "vllm.inputs", "vllm.logger", "vllm.sequence",
    "vllm.model_executor", "vllm.model_executor.model_loader",
    "vllm.model_executor.model_loader.weight_utils", "vllm.model_executor.models",
    "vllm.model_executor.models.interfaces", "vllm.model_executor.models.utils",
    "vllm.multimodal", "vllm.multimodal.inputs", "vllm.multimodal.parse",
    "vllm.multimodal.processing", "vllm.multimodal.processing.processor",
    "vllm.tokenizers", "vllm.tokenizers.mistral",
    "vllm_omni", "vllm_omni.quantization", "vllm_omni.quantization.component_config",
    "vllm_omni.platforms",
    "mistral_common", "mistral_common.protocol", "mistral_common.protocol.instruct",
    "mistral_common.protocol.instruct.chunk", "mistral_common.tokens",
    "mistral_common.tokens.tokenizers", "mistral_common.tokens.tokenizers.audio",
    "transformers", "transformers.tokenization_utils_base",
    "regex",  # only used for prompt remapping regexes we never call
]


def install_stubs():
    for name in _STUBS:
        if name in ("apex", "apex.normalization", "flash_attn"):
            continue  # leave absent so the try/except ImportError fallbacks fire
        sys.modules.setdefault(name, _StubModule(name))
    # `regex` is genuinely needed by re.fullmatch-style calls only in load_weights; use stdlib re
    import re as _re

    sys.modules["regex"] = _re


def _exec_source(path, module_name):
    mod = types.ModuleType(module_name)
    mod.__file__ = path
    sys.modules[module_name] = mod
    with open(path) as f:
        code = compile(f.read(), path, "exec")
    exec(code, mod.__dict__)
    return mod


def load_generation():
    """The module holding FlowMatchingAudioTransformer (Block 2)."""
    install_stubs()
    # parent packages must exist for the codec file's `from ... import` to resolve
    for p in ("vllm_omni.model_executor", "vllm_omni.model_executor.models",
              "vllm_omni.model_executor.models.voxtral_tts"):
        sys.modules.setdefault(p, _StubModule(p))
    return _exec_source(os.path.join(SRC, "voxtral_tts_audio_generation.py"), GEN_MOD)


def load_tokenizer_module():
    """The module holding VoxtralTTSAudioTokenizer (Block 3)."""
    gen = load_generation()
    sys.modules[GEN_MOD] = gen
    name = "vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_tokenizer"
    return _exec_source(os.path.join(SRC, "voxtral_tts_audio_tokenizer.py"), name)
