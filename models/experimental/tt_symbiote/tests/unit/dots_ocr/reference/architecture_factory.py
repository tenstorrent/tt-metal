# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HF architecture factories — dummy random weights, config only.

Per User decision #3 in :doc:`/PLAN_dots_ocr_bottomup_tests` we instantiate
the architectures only — no checkpoint is loaded. Each factory:

1. Loads :class:`DotsOCRConfig` from the HF cache (config-only).
2. Constructs the matching ``nn.Module`` instance.
3. Re-initializes every parameter with ``torch.nn.init.normal_(p, std=0.02)``
   under a fixed ``torch.manual_seed(seed)`` so successive calls with the same
   seed produce the **same** state_dict.
4. Returns the module on CPU in ``bfloat16``.

The factories are LRU-cached on ``(layer_idx, seed)`` so a test running 28
decoder layers does not re-instantiate 28 separate Attention modules.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config loader (config-only — does NOT download checkpoint weights)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_dots_config():
    """Load :class:`DotsOCRConfig` from the HF cache.

    ``snapshot_download`` is invoked with ``allow_patterns=['*.json','*.py']``
    which downloads only the config/source files — no model checkpoints
    (which would be several GB). Subsequent calls hit the local cache.
    """
    from transformers import AutoConfig
    from huggingface_hub import snapshot_download

    model_id = os.environ.get("DOTS_OCR_MODEL_ID", "rednote-hilab/dots.ocr")
    cached = snapshot_download(model_id, allow_patterns=["*.json", "*.py"])
    return AutoConfig.from_pretrained(cached, trust_remote_code=True)


def _seed_init_(module: nn.Module, seed: int = 0, std: float = 0.02) -> nn.Module:
    """Re-initialize every parameter in ``module`` deterministically.

    Each parameter gets its own seeded generator so the init order across
    modules with the same architecture is invariant to module-tree iteration
    order (which can shift between transformers versions).
    """
    g = torch.Generator()
    for i, p in enumerate(module.parameters()):
        g.manual_seed(seed + i)
        with torch.no_grad():
            tmp = torch.empty_like(p, dtype=torch.float32)
            tmp.normal_(mean=0.0, std=std, generator=g)
            p.copy_(tmp.to(p.dtype))
    return module.to(torch.bfloat16).eval()


# ---------------------------------------------------------------------------
# Qwen2 text factories
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def build_random_qwen2_attention(layer_idx: int = 7, seed: int = 0):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    cfg = _get_dots_config()
    # ``layer_idx`` is required by some transformers versions to set the
    # rope-section and head config. Pass it through.
    try:
        mod = Qwen2Attention(cfg, layer_idx=layer_idx)
    except TypeError:
        mod = Qwen2Attention(cfg)
    return _seed_init_(mod, seed=seed)


@functools.lru_cache(maxsize=64)
def build_random_qwen2_mlp(layer_idx: int = 7, seed: int = 0):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    cfg = _get_dots_config()
    mod = Qwen2MLP(cfg)
    return _seed_init_(mod, seed=seed)


@functools.lru_cache(maxsize=8)
def build_random_qwen2_rmsnorm(seed: int = 0):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    cfg = _get_dots_config()
    mod = Qwen2RMSNorm(cfg.hidden_size, eps=getattr(cfg, "rms_norm_eps", 1e-6))
    return _seed_init_(mod, seed=seed)


@functools.lru_cache(maxsize=8)
def build_random_qwen2_decoder_layer(layer_idx: int = 7, seed: int = 0):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    cfg = _get_dots_config()
    try:
        mod = Qwen2DecoderLayer(cfg, layer_idx=layer_idx)
    except TypeError:
        mod = Qwen2DecoderLayer(cfg)
    return _seed_init_(mod, seed=seed)


# ---------------------------------------------------------------------------
# Vision factories (HF dots-vision sources live in the model cache, not in
# transformers itself, so we import lazily via the trust-remote-code path.)
# ---------------------------------------------------------------------------


def _get_dots_vision_module(class_name: str):
    """Lookup a named class from the cached ``modeling_dots_vision`` source.

    The HF source file uses relative imports
    (``from .configuration_dots import DotsVisionConfig``) so we install
    a synthetic package first, then load the file as
    ``<pkg>.modeling_dots_vision``. The same approach is used by
    ``AutoConfig.from_pretrained(..., trust_remote_code=True)``.
    """
    from huggingface_hub import snapshot_download
    import importlib.util
    import sys
    import types

    model_id = os.environ.get("DOTS_OCR_MODEL_ID", "rednote-hilab/dots.ocr")
    cached = snapshot_download(model_id, allow_patterns=["*.json", "*.py"])

    pkg_name = "_dots_ocr_remote_code"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [cached]
        sys.modules[pkg_name] = pkg

    # Load configuration_dots first so the vision module's relative import resolves.
    for sub in ("configuration_dots", "modeling_dots_vision", "modeling_dots_ocr"):
        full = f"{pkg_name}.{sub}"
        if full in sys.modules:
            continue
        src = os.path.join(cached, f"{sub}.py")
        if not os.path.exists(src):
            continue
        spec = importlib.util.spec_from_file_location(full, src)
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        assert spec.loader is not None
        spec.loader.exec_module(m)

    return getattr(sys.modules[f"{pkg_name}.modeling_dots_vision"], class_name)


@functools.lru_cache(maxsize=8)
def build_random_dots_vision_attention(seed: int = 0):
    cfg = _get_dots_config().vision_config
    cls = _get_dots_vision_module("VisionAttention")
    dim = getattr(cfg, "hidden_size", 1536)
    num_heads = getattr(cfg, "num_attention_heads", 12)
    try:
        instance = cls(cfg, dim=dim, num_heads=num_heads)
    except TypeError:
        instance = cls(cfg)
    return _seed_init_(instance, seed=seed)


@functools.lru_cache(maxsize=8)
def build_random_dots_vision_mlp(seed: int = 0):
    cfg = _get_dots_config().vision_config
    cls = _get_dots_vision_module("DotsSwiGLUFFN")
    return _seed_init_(cls(cfg), seed=seed)


@functools.lru_cache(maxsize=8)
def build_random_dots_vision_block(seed: int = 0):
    cfg = _get_dots_config().vision_config
    cls = _get_dots_vision_module("DotsVisionBlock")
    return _seed_init_(cls(cfg), seed=seed)


@functools.lru_cache(maxsize=8)
def build_random_dots_vision_patch_embed(seed: int = 0):
    cfg = _get_dots_config().vision_config
    cls = _get_dots_vision_module("DotsPatchEmbed")
    return _seed_init_(cls(cfg), seed=seed)


@functools.lru_cache(maxsize=8)
def build_random_dots_vision_patch_merger(seed: int = 0):
    cfg = _get_dots_config().vision_config
    cls = _get_dots_vision_module("PatchMerger")
    # PatchMerger takes (dim, context_dim) -- read those off the config.
    dim = getattr(cfg, "embed_dim", cfg.hidden_size)
    ctx_dim = getattr(cfg, "embed_dim", cfg.hidden_size)
    try:
        instance = cls(dim=dim, context_dim=ctx_dim, spatial_merge_size=getattr(cfg, "spatial_merge_size", 2))
    except TypeError:
        instance = cls(dim, ctx_dim)
    return _seed_init_(instance, seed=seed)
